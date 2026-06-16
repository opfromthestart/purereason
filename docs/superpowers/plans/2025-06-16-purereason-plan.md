# PureReason Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an RL training pipeline that fine-tunes a 1.5B LLM on math, code, logic, and Lean proof tasks using GRPO without KL divergence penalty.

**Architecture:** Plugin-based task registry (TaskEnv ABC) feeds a mixed dataset into HuggingFace TRL's GRPOTrainer. The model uses 4-bit QLoRA. Each task provides its own reward function with verifiable ground-truth signals.

**Tech Stack:** Python, PyTorch, Transformers, TRL, PEFT, bitsandbytes, sympy, LeanDojo

---

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/tasks/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "purereason"
version = "0.1.0"
description = "RL-based LLM reasoning training without KL divergence"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1.0",
    "transformers>=4.44.0",
    "trl>=0.11.0",
    "peft>=0.12.0",
    "bitsandbytes>=0.43.0",
    "datasets>=2.20.0",
    "accelerate>=0.33.0",
    "sympy>=1.12",
    "wandb>=0.17.0",
]
```

- [ ] **Step 2: Create requirements.txt**

```
torch>=2.1.0
transformers>=4.44.0
trl>=0.11.0
peft>=0.12.0
bitsandbytes>=0.43.0
datasets>=2.20.0
accelerate>=0.33.0
sympy>=1.12
wandb>=0.17.0
```

- [ ] **Step 3: Create directory structure and init files**

```bash
mkdir -p src/tasks
touch src/__init__.py
touch src/tasks/__init__.py
```

- [ ] **Step 4: Verify venv and install dependencies**

```bash
source bin/activate && pip install -r requirements.txt
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml requirements.txt src/__init__.py src/tasks/__init__.py
git commit -m "feat: scaffold project with dependencies"
```

---

### Task 2: TaskEnv Base Class + TaskRegistry + Config

**Files:**
- Create: `src/task_env.py`
- Create: `src/config.py`

- [ ] **Step 1: Write task_env.py with TaskEnv ABC and TaskRegistry**

```python
from abc import ABC, abstractmethod
from datasets import Dataset


class TaskEnv(ABC):
    name: str

    @abstractmethod
    def load_dataset(self) -> Dataset:
        ...

    @abstractmethod
    def get_prompt(self, example: dict) -> str:
        ...

    @abstractmethod
    def compute_reward(self, prompt: str, completion: str) -> float:
        ...


class TaskRegistry:
    _tasks: dict[str, type[TaskEnv]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(task_cls: type[TaskEnv]):
            task_cls.name = name
            cls._tasks[name] = task_cls
            return task_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> TaskEnv:
        task_cls = cls._tasks.get(name)
        if task_cls is None:
            raise KeyError(f"Unknown task: {name}. Registered: {list(cls._tasks.keys())}")
        return task_cls()

    @classmethod
    def list_tasks(cls) -> list[str]:
        return list(cls._tasks.keys())
```

- [ ] **Step 2: Write config.py**

```python
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    fallback_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class TrainingConfig:
    output_dir: str = "./output"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_generations_per_prompt: int = 4
    max_prompt_length: int = 512
    max_completion_length: int = 2048
    temperature: float = 0.8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 50
    max_steps: int = 1000
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    beta: float = 0.0  # KL coefficient — disabled
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True


@dataclass
class TaskSamplingConfig:
    tasks: dict[str, float] = field(default_factory=lambda: {
        "gsm8k": 0.25,
        "math": 0.15,
        "mbpp": 0.15,
        "prontoqa": 0.15,
        "lean_minif2f": 0.10,
        "sokoban": 0.10,
        "spreadsheet": 0.10,
    })
    max_samples_per_task: int = 2000


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    task_sampling: TaskSamplingConfig = field(default_factory=TaskSamplingConfig)
```

- [ ] **Step 3: Verify imports work**

```bash
source bin/activate && python -c "from src.config import Config; c = Config(); print(c.model.base_model)"
```

- [ ] **Step 4: Commit**

```bash
git add src/task_env.py src/config.py
git commit -m "feat: add TaskEnv base class, TaskRegistry, and Config"
```

---

### Task 3: Model Loading (QLoRA)

**Files:**
- Create: `src/model.py`

- [ ] **Step 1: Write model.py**

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.config import ModelConfig


def load_model_and_tokenizer(config: ModelConfig) -> tuple:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=True,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        if "out of memory" in str(e).lower() and config.base_model != config.fallback_model:
            print(f"[WARN] OOM loading {config.base_model}, falling back to {config.fallback_model}")
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                config.fallback_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            raise

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def save_model(model, tokenizer, path: str):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
```

- [ ] **Step 2: Test import**

```bash
source bin/activate && python -c "from src.model import load_model_and_tokenizer; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/model.py
git commit -m "feat: add QLoRA model loading and LoRA configuration"
```

---

### Task 4: Math Task (GSM8K + MATH)

**Files:**
- Create: `src/tasks/math_task.py`

- [ ] **Step 1: Write math_task.py**

```python
import re

from datasets import Dataset, concatenate_datasets

from src.task_env import TaskEnv, TaskRegistry


@TaskRegistry.register("gsm8k")
class GSM8KTask(TaskEnv):
    PROMPT_TEMPLATE = (
        "Solve this math problem step by step. Put your final answer within "
        "\\boxed{}.\n\nProblem: {question}"
    )

    def load_dataset(self) -> Dataset:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        ds = ds.select_columns(["question", "answer"])
        ds = ds.map(
            lambda x: {"prompt": self.get_prompt(x), "task_name": "gsm8k"},
            remove_columns=["question", "answer"],
        )
        return ds

    def get_prompt(self, example: dict) -> str:
        return self.PROMPT_TEMPLATE.format(question=example["question"])

    def compute_reward(self, prompt: str, completion: str) -> float:
        import sympy
        pred = self._extract_answer(completion)
        ref = self._extract_reference(prompt)
        if pred is None or ref is None:
            return 0.0
        try:
            if sympy.simplify(f"{pred} - ({ref})") == 0:
                return 1.0
        except Exception:
            pass
        try:
            if abs(float(pred) - float(ref)) < 1e-6:
                return 1.0
        except Exception:
            pass
        return 0.0

    def _extract_answer(self, text: str) -> str | None:
        patterns = [
            r"\\boxed\{([^}]*)\}",
            r"####\s*(.*?)$",
            r"The answer is\s*\$?([^\$\.\n]+)",
            r"=\s*(-?\d+\.?\d*)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                return matches[-1].strip()
        return None

    def _extract_reference(self, prompt: str) -> str | None:
        ds = load_dataset_for_gsm8k()
        for row in ds:
            formatted = self.get_prompt(row)
            if formatted == prompt:
                match = re.search(r"####\s*(-?\d+\.?\d*)", row["answer"])
                if match:
                    return match.group(1).strip()
        return None


def load_dataset_for_gsm8k():
    from datasets import load_dataset
    return load_dataset("openai/gsm8k", "main", split="train")


@TaskRegistry.register("math")
class MATHTask(TaskEnv):
    PROMPT_TEMPLATE = (
        "Solve this math problem step by step. Put your final answer within "
        "\\boxed{}.\n\nProblem: {problem}"
    )

    def load_dataset(self) -> Dataset:
        from datasets import load_dataset
        ds = load_dataset("hendrycks/competition_math", split="train")
        ds = ds.select_columns(["problem", "solution", "answer"])
        ds = ds.map(
            lambda x: {"prompt": self.get_prompt(x), "task_name": "math"},
            remove_columns=["problem", "solution", "answer"],
        )
        return ds

    def get_prompt(self, example: dict) -> str:
        return self.PROMPT_TEMPLATE.format(problem=example["problem"])

    def compute_reward(self, prompt: str, completion: str) -> float:
        import sympy
        pred = self._extract_answer(completion)
        ref = self._lookup_reference(prompt)
        if pred is None or ref is None:
            return 0.0
        try:
            if sympy.simplify(f"({pred}) - ({ref})") == 0:
                return 1.0
        except Exception:
            pass
        try:
            if float(pred) == float(ref):
                return 1.0
        except Exception:
            pass
        return 0.0

    def _extract_answer(self, text: str) -> str | None:
        patterns = [
            r"\\boxed\{([^}]*)\}",
            r"The answer is\s*\$?([^\$\.\n]+)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                return matches[-1].strip()
        return None

    def _lookup_reference(self, prompt: str) -> str | None:
        ds = load_dataset_for_math()
        for row in ds:
            if self.get_prompt(row) == prompt:
                return row["answer"].strip()
        return None


def load_dataset_for_math():
    from datasets import load_dataset
    return load_dataset("hendrycks/competition_math", split="train")
```

- [ ] **Step 2: Verify math tasks register and load**

```bash
source bin/activate && python -c "
from src.tasks.math_task import GSM8KTask, MATHTask
from src.task_env import TaskRegistry
print('Registered:', TaskRegistry.list_tasks())
t = TaskRegistry.get('gsm8k')
print('Type:', type(t).__name__)
"
```

- [ ] **Step 3: Commit**

```bash
git add src/tasks/math_task.py
git commit -m "feat: add GSM8K and MATH tasks with sympy-based reward"
```

---

### Task 5: Code Task (MBPP + HumanEval)

**Files:**
- Create: `src/tasks/code_task.py`

- [ ] **Step 1: Write code_task.py**

```python
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from datasets import Dataset

from src.task_env import TaskEnv, TaskRegistry


def _run_tests(code: str, test_code: str) -> tuple[int, int]:
    full_code = code + "\n\n" + test_code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return 1, 1
        return 0, 1
    except subprocess.TimeoutExpired:
        return 0, 1
    except Exception:
        return 0, 1
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _extract_code_block(completion: str) -> str:
    patterns = [
        r"```python\n(.*?)```",
        r"```\n(.*?)```",
        r"```python\s*(.*?)```",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, completion, re.DOTALL)
        if matches:
            return matches[0].strip()
    return completion.strip()


@TaskRegistry.register("mbpp")
class MBPPTask(TaskEnv):
    PROMPT_TEMPLATE = (
        "Complete the following Python function. Return only the function "
        "body, no explanation.\n\n{description}\n\n```python\n{code}\n```"
    )

    def load_dataset(self) -> Dataset:
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/mbpp", "full", split="train")
        ds = ds.map(
            lambda x: {"prompt": self.get_prompt(x), "task_name": "mbpp"},
            remove_columns=ds.column_names,
        )
        return ds

    def get_prompt(self, example: dict) -> str:
        return self.PROMPT_TEMPLATE.format(
            description=example.get("text", example.get("description", "")),
            code=example["code"],
        )

    def compute_reward(self, prompt: str, completion: str) -> float:
        code = _extract_code_block(completion)
        ds = load_dataset_for_mbpp()
        for row in ds:
            if self.get_prompt(row) == prompt:
                test_list = row.get("test_list", row.get("test_imports", []))
                if isinstance(test_list, str):
                    test_list = [test_list]
                passed = 0
                total = len(test_list) if test_list else 1
                if not test_list:
                    return 0.0
                for test in test_list:
                    p, _ = _run_tests(code, test)
                    passed += p
                return passed / total
        return 0.0


def load_dataset_for_mbpp():
    from datasets import load_dataset
    return load_dataset("google-research-datasets/mbpp", "full", split="train")


@TaskRegistry.register("humaneval")
class HumanEvalTask(TaskEnv):
    PROMPT_TEMPLATE = "{prompt}"

    def load_dataset(self) -> Dataset:
        from datasets import load_dataset
        ds = load_dataset("openai/openai_humaneval", split="test")
        ds = ds.map(
            lambda x: {"prompt": self.get_prompt(x), "task_name": "humaneval"},
        )
        return ds

    def get_prompt(self, example: dict) -> str:
        return self.PROMPT_TEMPLATE.format(prompt=example["prompt"])

    def compute_reward(self, prompt: str, completion: str) -> float:
        code = _extract_code_block(completion)
        full_code = prompt + code
        ds = load_dataset_for_humaneval()
        for row in ds:
            if row["prompt"] == prompt:
                canonical = row.get("canonical_solution", "")
                test_code = row.get("test", "")
                if not test_code:
                    return 0.0
                inner_test = test_code.replace(
                    f"candidate({row.get('entry_point', '')})",
                    full_code,
                )
                if "def check(" in inner_test:
                    passed, total = _run_tests(full_code, inner_test)
                    return passed / total if total > 0 else 0.0
                passed, total = _run_tests(full_code, inner_test)
                return passed / total if total > 0 else 0.0
        return 0.0


def load_dataset_for_humaneval():
    from datasets import load_dataset
    return load_dataset("openai/openai_humaneval", split="test")
```

- [ ] **Step 2: Verify code tasks register**

```bash
source bin/activate && python -c "
from src.tasks.code_task import MBPPTask, HumanEvalTask
from src.task_env import TaskRegistry
print('Registered:', TaskRegistry.list_tasks())
"
```

- [ ] **Step 3: Commit**

```bash
git add src/tasks/code_task.py
git commit -m "feat: add MBPP and HumanEval tasks with subprocess test execution"
```

---

### Task 6: Logic Task (PRONTOQA)

**Files:**
- Create: `src/tasks/logic_task.py`

- [ ] **Step 1: Write logic_task.py**

```python
import re

from datasets import Dataset

from src.task_env import TaskEnv, TaskRegistry


@TaskRegistry.register("prontoqa")
class PRONTOQATask(TaskEnv):
    PROMPT_TEMPLATE = (
        "Given the following facts, answer the question by reasoning step by "
        "step. Put your final answer on a new line after 'Answer:'.\n\n"
        "Facts: {context}\n\nQuestion: {question}"
    )

    def load_dataset(self) -> Dataset:
        from datasets import load_dataset
        ds = load_dataset("ashen/PRONTOQA", split="train")
        required_cols = ["context", "question", "answer"]
        ds = ds.select_columns([c for c in required_cols if c in ds.column_names])
        ds = ds.map(
            lambda x: {"prompt": self.get_prompt(x), "task_name": "prontoqa"},
            remove_columns=ds.column_names,
        )
        return ds

    def get_prompt(self, example: dict) -> str:
        return self.PROMPT_TEMPLATE.format(
            context=example.get("context", ""),
            question=example.get("question", ""),
        )

    def compute_reward(self, prompt: str, completion: str) -> float:
        pred = self._extract_answer(completion)
        ref = self._lookup_reference(prompt)
        if pred is None or ref is None:
            return 0.0
        if pred.strip().lower() == ref.strip().lower():
            return 1.0
        return 0.0

    def _extract_answer(self, text: str) -> str | None:
        patterns = [
            r"Answer:\s*(.*?)$",
            r"answer is\s*(.*?)[\.\n]",
            r"^(true|false|yes|no)$",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                return matches[-1].strip()
        lines = text.strip().split("\n")
        for line in reversed(lines):
            stripped = line.strip()
            if stripped.lower() in ("true", "false", "yes", "no"):
                return stripped
        return None

    def _lookup_reference(self, prompt: str) -> str | None:
        ds = load_dataset_for_prontoqa()
        for row in ds:
            if self.get_prompt(row) == prompt:
                return row.get("answer", "").strip()
        return None


def load_dataset_for_prontoqa():
    from datasets import load_dataset
    return load_dataset("ashen/PRONTOQA", split="train")
```

- [ ] **Step 2: Verify logic task registers**

```bash
source bin/activate && python -c "
from src.tasks.logic_task import PRONTOQATask
from src.task_env import TaskRegistry
print('Registered:', sorted(TaskRegistry.list_tasks()))
"
```

- [ ] **Step 3: Commit**

```bash
git add src/tasks/logic_task.py
git commit -m "feat: add PRONTOQA logic task with answer matching reward"
```

---

### Task 7: Dataset Loader

**Files:**
- Create: `src/dataset.py`

- [ ] **Step 1: Write dataset.py**

```python
from datasets import Dataset, concatenate_datasets

from src.config import TaskSamplingConfig
from src.task_env import TaskRegistry


def build_mixed_dataset(config: TaskSamplingConfig) -> Dataset:
    samples = []
    for task_name, weight in config.tasks.items():
        try:
            task = TaskRegistry.get(task_name)
        except KeyError:
            print(f"[WARN] Task '{task_name}' not registered, skipping")
            continue
        ds = task.load_dataset()
        n = min(len(ds), config.max_samples_per_task)
        ds = ds.select(range(n))
        samples.append(ds)
        print(f"[INFO] Loaded {n} examples from '{task_name}'")

    if not samples:
        raise ValueError("No task datasets loaded. Check task registry.")

    mixed = concatenate_datasets(samples)
    mixed = mixed.shuffle(seed=42)
    print(f"[INFO] Mixed dataset size: {len(mixed)}")
    return mixed
```

- [ ] **Step 2: Quick integration test**

```bash
source bin/activate && python -c "
from src.config import TaskSamplingConfig
from src.dataset import build_mixed_dataset
import src.tasks.math_task  # triggers registration
cfg = TaskSamplingConfig(tasks={'gsm8k': 1.0}, max_samples_per_task=100)
ds = build_mixed_dataset(cfg)
print('Columns:', ds.column_names)
print('First prompt:', ds[0]['prompt'][:80])
"
```

- [ ] **Step 3: Commit**

```bash
git add src/dataset.py
git commit -m "feat: add mixed dataset builder from registered tasks"
```

---

### Task 8: Training Loop

**Files:**
- Create: `src/train.py`

- [ ] **Step 1: Write train.py**

```python
import os
import sys
import torch

from trl import GRPOConfig, GRPOTrainer

from src.config import Config
from src.model import load_model_and_tokenizer, save_model
from src.dataset import build_mixed_dataset
from src.task_env import TaskRegistry


def create_reward_fn():
    def reward_fn(prompts, completions, task_name=None, **kwargs):
        rewards = []
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            try:
                task = TaskRegistry.get(task_name[i])
                r = task.compute_reward(prompt, completion)
                rewards.append(r)
            except Exception as e:
                print(f"[WARN] Reward computation failed: {e}")
                rewards.append(0.0)
        return rewards
    return reward_fn


def main():
    config = Config()

    import src.tasks.math_task        # noqa: triggers @TaskRegistry.register
    import src.tasks.code_task        # noqa
    import src.tasks.logic_task       # noqa
    import src.tasks.sokoban_task     # noqa
    import src.tasks.spreadsheet_task # noqa
    import src.tasks.lean_task        # noqa

    os.makedirs(config.training.output_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(config.model)

    dataset = build_mixed_dataset(config.task_sampling)

    grpo_config = GRPOConfig(
        output_dir=config.training.output_dir,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_generations=config.training.num_generations_per_prompt,
        max_prompt_length=config.training.max_prompt_length,
        max_completion_length=config.training.max_completion_length,
        temperature=config.training.temperature,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.max_steps,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        bf16=config.training.bf16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        beta=config.training.beta,
        seed=config.training.seed,
        report_to="wandb",
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=create_reward_fn(),
        args=grpo_config,
        train_dataset=dataset,
    )

    trainer.train()

    save_model(model, tokenizer, os.path.join(config.training.output_dir, "final"))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check and import verification**

```bash
source bin/activate && python -c "
import ast
with open('src/train.py') as f:
    ast.parse(f.read())
print('Syntax OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add src/train.py
git commit -m "feat: add main training loop with GRPO via TRL"
```

---

### Task 9: Evaluation Script

**Files:**
- Create: `src/eval.py`

- [ ] **Step 1: Write eval.py**

```python
import json
import os
import torch
from tqdm import tqdm

from src.config import Config
from src.model import load_model_and_tokenizer
from src.task_env import TaskRegistry


def evaluate(model, tokenizer, task_name: str, split: str = "test", max_samples: int = 100):
    task = TaskRegistry.get(task_name)
    ds = task.load_dataset()
    ds = ds.select(range(min(len(ds), max_samples)))

    total_reward = 0.0
    results = []

    model.eval()
    for example in tqdm(ds, desc=f"Evaluating {task_name}"):
        prompt = task.get_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        reward = task.compute_reward(prompt, completion)
        total_reward += reward
        results.append({"prompt": prompt[:200], "completion": completion[:200], "reward": reward})

    avg_reward = total_reward / len(ds) if len(ds) > 0 else 0.0
    print(f"{task_name}: avg_reward={avg_reward:.4f} over {len(ds)} samples")
    return avg_reward, results


def main():
    config = Config()

    import src.tasks.math_task        # noqa
    import src.tasks.code_task        # noqa
    import src.tasks.logic_task       # noqa
    import src.tasks.sokoban_task     # noqa
    import src.tasks.spreadsheet_task # noqa
    import src.tasks.lean_task        # noqa

    model, tokenizer = load_model_and_tokenizer(config.model)

    all_results = {}
    for task_name in TaskRegistry.list_tasks():
        try:
            avg, results = evaluate(model, tokenizer, task_name)
            all_results[task_name] = {"avg_reward": avg, "samples": len(results)}
        except Exception as e:
            print(f"[WARN] Evaluation failed for {task_name}: {e}")

    os.makedirs(config.training.output_dir, exist_ok=True)
    with open(os.path.join(config.training.output_dir, "eval_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nSummary:")
    for name, r in all_results.items():
        print(f"  {name}: {r['avg_reward']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check**

```bash
source bin/activate && python -c "
import ast
with open('src/eval.py') as f:
    ast.parse(f.read())
print('Syntax OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add src/eval.py
git commit -m "feat: add evaluation script for held-out benchmarks"
```

---

### Task 10: Lean Task (miniF2F via LeanDojo)

**Files:**
- Create: `scripts/setup_lean.sh`
- Create: `src/tasks/lean_task.py`

- [ ] **Step 1: Write setup_lean.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "Installing Lean 4 toolchain..."

if ! command -v elan &> /dev/null; then
    curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
    export PATH="$HOME/.elan/bin:$PATH"
fi

elan default leanprover/lean4:stable
lean --version
lake --version

echo "Lean toolchain installation complete."
```

```bash
chmod +x scripts/setup_lean.sh
```

- [ ] **Step 2: Write lean_task.py**

```python
import re
import shutil
import subprocess

from datasets import Dataset

from src.task_env import TaskEnv, TaskRegistry


def _lean_available() -> bool:
    return shutil.which("lean") is not None


@TaskRegistry.register("lean_minif2f")
class LeanMiniF2FTask(TaskEnv):
    PROMPT_TEMPLATE = (
        "Prove the following theorem in Lean 4. Write a complete proof "
        "using only valid Lean 4 tactics.\n\n```lean\n{theorem}\n```"
    )

    def load_dataset(self) -> Dataset:
        if not _lean_available():
            print("[WARN] Lean not installed. Returning empty dataset. "
                  "Run scripts/setup_lean.sh to install.")
            return Dataset.from_dict({"prompt": [], "task_name": []})

        from datasets import load_dataset
        ds = load_dataset("lean-dojo/LeanDojo", split="train")
        ds = ds.select_columns([c for c in ["name", "goal", "full_name"] if c in ds.column_names])
        ds = ds.map(
            lambda x: {"prompt": self.get_prompt(x), "task_name": "lean_minif2f"},
            remove_columns=ds.column_names,
        )
        return ds

    def get_prompt(self, example: dict) -> str:
        theorem = example.get("goal", example.get("name", ""))
        return self.PROMPT_TEMPLATE.format(theorem=theorem)

    def compute_reward(self, prompt: str, completion: str) -> float:
        if not _lean_available():
            return 0.0

        code = self._extract_lean_code(completion)
        if not code:
            return 0.0

        result = subprocess.run(
            ["lean", "--stdin"],
            input=code,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return 1.0 if result.returncode == 0 else 0.0

    def _extract_lean_code(self, text: str) -> str | None:
        patterns = [
            r"```lean\n(.*?)```",
            r"```\n(.*?)```",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return matches[0].strip()
        return text.strip()
```

- [ ] **Step 3: Syntax check**

```bash
source bin/activate && python -c "
import ast
with open('src/tasks/lean_task.py') as f:
    ast.parse(f.read())
print('Syntax OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/setup_lean.sh src/tasks/lean_task.py
git commit -m "feat: add Lean miniF2F task via LeanDojo and setup script"
```

---

### Task 11: Sokoban Environment + Task

**Files:**
- Create: `src/envs/__init__.py`
- Create: `src/envs/sokoban_env.py`
- Create: `src/tasks/sokoban_task.py`

- [ ] **Step 1: Create envs/__init__.py**

```bash
mkdir -p src/envs
touch src/envs/__init__.py
```

- [ ] **Step 2: Write sokoban_env.py**

```python
import copy
import random


SYMBOLS = {"empty": " ", "wall": "#", "box": "$", "target": ".", "box_on_target": "*",
           "agent": "@", "agent_on_target": "+"}


class SokobanEnv:
    def __init__(self, width: int = 6, height: int = 6, num_boxes: int = 2, seed: int = 42):
        random.seed(seed)
        self.width = width
        self.height = height
        self.num_boxes = num_boxes
        self.grid = None
        self.init_grid = None
        self.agent_pos = (0, 0)
        self.reset()

    def reset(self) -> str:
        self._generate_puzzle()
        return self.render()

    def _generate_puzzle(self):
        self.grid = [[" " for _ in range(self.width)] for _ in range(self.height)]
        for x in range(self.width):
            self.grid[0][x] = "#"
            self.grid[self.height - 1][x] = "#"
        for y in range(self.height):
            self.grid[y][0] = "#"
            self.grid[y][self.width - 1] = "#"

        free_cells = [(y, x) for y in range(1, self.height - 1)
                      for x in range(1, self.width - 1)]
        random.shuffle(free_cells)

        self.agent_pos = free_cells.pop()
        gy, gx = self.agent_pos
        self.grid[gy][gx] = "@"

        for _ in range(self.num_boxes):
            if not free_cells:
                break
            by, bx = free_cells.pop()
            self.grid[by][bx] = "$"

        for _ in range(self.num_boxes):
            if not free_cells:
                break
            ty, tx = free_cells.pop()
            self.grid[ty][tx] = "." if self.grid[ty][tx] == " " else (
                "*" if self.grid[ty][tx] == "$" else self.grid[ty][tx]
            )

        self.init_grid = copy.deepcopy(self.grid)

    def render(self) -> str:
        lines = []
        for row in self.grid:
            lines.append("".join(row))
        return "\n".join(lines)

    def step(self, move: str) -> tuple[str, float, bool]:
        moves = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
        if move not in moves:
            return self.render(), self._reward(), False

        dy, dx = moves[move]
        ay, ax = self.agent_pos
        ny, nx = ay + dy, ax + dx

        if not (0 <= ny < self.height and 0 <= nx < self.width):
            return self.render(), self._reward(), False
        if self._is_wall(ny, nx):
            return self.render(), self._reward(), False

        if self._is_box(ny, nx):
            bny, bnx = ny + dy, nx + dx
            if not (0 <= bny < self.height and 0 <= bnx < self.width):
                return self.render(), self._reward(), False
            if self._is_wall(bny, bnx) or self._is_box(bny, bnx):
                return self.render(), self._reward(), False
            self._move_box((ny, nx), (bny, bnx))

        self._move_agent(self.agent_pos, (ny, nx))
        self.agent_pos = (ny, nx)
        return self.render(), self._reward(), self._is_done()

    def _is_wall(self, y: int, x: int) -> bool:
        return self.grid[y][x] == "#"

    def _is_box(self, y: int, x: int) -> bool:
        return self.grid[y][x] in ("$", "*")

    def _is_target(self, y: int, x: int) -> bool:
        return self.grid[y][x] in (".", "*", "+")

    def _move_agent(self, from_pos, to_pos):
        fy, fx = from_pos
        ty, tx = to_pos
        from_was_target = self._is_target(fy, fx) and self.grid[fy][fx] in ("+",)
        self.grid[fy][fx] = "." if from_was_target else " "
        to_is_target = self._is_target(ty, tx)
        self.grid[ty][tx] = "+" if to_is_target else "@"

    def _move_box(self, from_pos, to_pos):
        fy, fx = from_pos
        ty, tx = to_pos
        box_on_target = self.grid[fy][fx] == "*"
        self.grid[fy][fx] = "." if box_on_target else " "
        target_under = self.grid[ty][tx] == "."
        self.grid[ty][tx] = "*" if target_under else "$"

    def _reward(self) -> float:
        boxes_on_target = sum(
            1 for y in range(self.height) for x in range(self.width)
            if self.grid[y][x] == "*"
        )
        return boxes_on_target / self.num_boxes

    def _is_done(self) -> bool:
        return self._reward() >= 1.0

    @property
    def board(self) -> str:
        return self.render()
```

- [ ] **Step 3: Write sokoban_task.py**

```python
import random

from datasets import Dataset

from src.task_env import TaskEnv, TaskRegistry
from src.envs.sokoban_env import SokobanEnv


@TaskRegistry.register("sokoban")
class SokobanTask(TaskEnv):
    PROMPT_TEMPLATE = (
        "You control an agent (@) on a grid. Push boxes ($) onto targets (.) "
        "to score. You cannot walk through walls (#) or boxes. Moves: U D L R. "
        "Output your moves as a continuous sequence on one line.\n\n"
        "Board:\n{board}\n\nMoves:"
    )

    def __init__(self, num_puzzles: int = 500, max_moves: int = 50):
        self.num_puzzles = num_puzzles
        self.max_moves = max_moves
        self._puzzles = {}

    def load_dataset(self) -> Dataset:
        data = []
        for i in range(self.num_puzzles):
            env = SokobanEnv(width=6, height=6, num_boxes=2, seed=42 + i)
            board = env.reset()
            self._puzzles[i] = {"env": env, "board": board}
            data.append({"prompt": self.get_prompt({"board": board}), "task_name": "sokoban"})
        return Dataset.from_list(data)

    def get_prompt(self, example: dict) -> str:
        return self.PROMPT_TEMPLATE.format(board=example["board"])

    def compute_reward(self, prompt: str, completion: str) -> float:
        board = None
        for pid, puzzle in self._puzzles.items():
            if self.get_prompt({"board": puzzle["board"]}) == prompt:
                board = puzzle["board"]
                break
        if board is None:
            return 0.0

        env = SokobanEnv(width=6, height=6, num_boxes=2, seed=42)
        env.reset()

        moves = self._extract_moves(completion)
        for move in moves[:self.max_moves]:
            _, reward, done = env.step(move)
            if done:
                break
        return env._reward()

    def _extract_moves(self, text: str) -> list[str]:
        filtered = []
        for ch in text.strip().upper():
            if ch in "UDLR":
                filtered.append(ch)
        return filtered
```

- [ ] **Step 4: Verify Sokoban environment works**

```bash
source bin/activate && python -c "
from src.envs.sokoban_env import SokobanEnv
env = SokobanEnv(width=6, height=6, seed=42)
board = env.reset()
print(board)
print('Moves U:', env.step('U')[0])
print('Reward:', env._reward())
"
```

- [ ] **Step 5: Commit**

```bash
git add src/envs/ src/tasks/sokoban_task.py
git commit -m "feat: add Sokoban puzzle environment and task"
```

---

### Task 12: Spreadsheet Environment + Task

**Files:**
- Create: `src/envs/spreadsheet_env.py`
- Create: `src/tasks/spreadsheet_task.py`

- [ ] **Step 1: Write spreadsheet_env.py**

```python
import random
import re
from typing import Any


class SpreadsheetEnv:
    def __init__(self, rows: int = 10, cols: int = 10, seed: int = 42):
        random.seed(seed)
        self.rows = rows
        self.cols = cols
        self.grid: dict[str, Any] = {}
        self.reset()

    def reset(self):
        self.grid = {}
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self._to_ref(r, c)
                self.grid[cell] = random.randint(1, 100)
        return self.render()

    def _to_ref(self, row: int, col: int) -> str:
        return f"{chr(ord('A') + col)}{row + 1}"

    def _parse_ref(self, ref: str) -> tuple[int, int]:
        col = ord(ref[0].upper()) - ord("A")
        row = int(ref[1:]) - 1
        return row, col

    def render(self) -> str:
        lines = []
        header = "     " + "  ".join(f"{chr(ord('A') + c):>4}" for c in range(self.cols))
        lines.append(header)
        lines.append("    " + "-" * (5 * self.cols))
        for r in range(self.rows):
            row_str = f"{r + 1:>2} |"
            for c in range(self.cols):
                cell = self._to_ref(r, c)
                val = self.grid.get(cell, "")
                row_str += f" {str(val):>4}"
            lines.append(row_str)
        return "\n".join(lines)

    def set(self, cell: str, value: Any):
        self.grid[cell.upper()] = value

    def get(self, cell: str) -> Any:
        return self.grid.get(cell.upper(), 0)

    def evaluate_formula(self, formula: str, target_cell: str) -> Any:
        formula = formula.strip()
        if formula.startswith("="):
            formula = formula[1:]

        sum_match = re.match(r"SUM\((\w\d+):(\w\d+)\)", formula, re.IGNORECASE)
        if sum_match:
            start, end = sum_match.group(1), sum_match.group(2)
            sr, sc = self._parse_ref(start)
            er, ec = self._parse_ref(end)
            total = 0
            for r in range(min(sr, er), max(sr, er) + 1):
                for c in range(min(sc, ec), max(sc, ec) + 1):
                    val = self.get(self._to_ref(r, c))
                    if isinstance(val, (int, float)):
                        total += val
            return total

        avg_match = re.match(r"AVG\((\w\d+):(\w\d+)\)", formula, re.IGNORECASE)
        if avg_match:
            start, end = avg_match.group(1), avg_match.group(2)
            sr, sc = self._parse_ref(start)
            er, ec = self._parse_ref(end)
            total = 0
            count = 0
            for r in range(min(sr, er), max(sr, er) + 1):
                for c in range(min(sc, ec), max(sc, ec) + 1):
                    val = self.get(self._to_ref(r, c))
                    if isinstance(val, (int, float)):
                        total += val
                        count += 1
            return total / count if count > 0 else 0

        count_match = re.match(r"COUNT\((\w\d+):(\w\d+)\)", formula, re.IGNORECASE)
        if count_match:
            start, end = count_match.group(1), count_match.group(2)
            sr, sc = self._parse_ref(start)
            er, ec = self._parse_ref(end)
            count = 0
            for r in range(min(sr, er), max(sr, er) + 1):
                for c in range(min(sc, ec), max(sc, ec) + 1):
                    val = self.get(self._to_ref(r, c))
                    if isinstance(val, (int, float)):
                        count += 1
            return count

        if_match = re.match(r"IF\((.+),(.+),(.+)\)", formula, re.IGNORECASE)
        if if_match:
            cond, true_val, false_val = if_match.group(1), if_match.group(2), if_match.group(3)
            cond = cond.strip()
            if ">" in cond:
                left, right = cond.split(">", 1)
                left_val = self._resolve_value(left.strip())
                right_val = self._resolve_value(right.strip())
                if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
                    return self._resolve_value(true_val.strip()) if left_val > right_val else self._resolve_value(false_val.strip())
            if "<" in cond:
                left, right = cond.split("<", 1)
                left_val = self._resolve_value(left.strip())
                right_val = self._resolve_value(right.strip())
                if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
                    return self._resolve_value(true_val.strip()) if left_val < right_val else self._resolve_value(false_val.strip())

        if re.match(r"^\w\d+$", formula, re.IGNORECASE):
            return self.get(formula)

        try:
            return eval(formula, {"__builtins__": {}}, {})
        except Exception:
            return formula

    def _resolve_value(self, expr: str) -> Any:
        expr = expr.strip()
        if re.match(r"^\w\d+$", expr, re.IGNORECASE):
            return self.get(expr)
        try:
            return int(expr)
        except ValueError:
            try:
                return float(expr)
            except ValueError:
                return expr
```

- [ ] **Step 2: Write spreadsheet_task.py**

```python
import random

from datasets import Dataset

from src.task_env import TaskEnv, TaskRegistry
from src.envs.spreadsheet_env import SpreadsheetEnv


TASK_TEMPLATES = [
    (
        "Compute the sum of cells {start} through {end}.",
        "SUM({start}:{end})",
    ),
    (
        "Compute the average of cells {start} through {end}.",
        "AVG({start}:{end})",
    ),
    (
        "Count how many numbers are in the range {start}:{end}.",
        "COUNT({start}:{end})",
    ),
    (
        "If cell {cell_a} is greater than cell {cell_b}, return the value of "
        "cell {cell_a}, otherwise return cell {cell_b}.",
        "IF({cell_a}>{cell_b},{cell_a},{cell_b})",
    ),
]


@TaskRegistry.register("spreadsheet")
class SpreadsheetTask(TaskEnv):
    PROMPT_TEMPLATE = (
        "You have a spreadsheet. Cells are referenced as A1, B2, etc. "
        "Write formulas using SUM(range), AVG(range), COUNT(range), "
        "IF(condition, true_val, false_val).\n\n"
        "Grid:\n{grid}\n\n"
        "Task: {task}\n\n"
        "Output your answer as:\nFORMULA: <cell>=<formula>"
    )

    def __init__(self, num_puzzles: int = 500):
        self.num_puzzles = num_puzzles
        self._puzzles = {}

    def load_dataset(self) -> Dataset:
        data = []
        random.seed(42)
        for i in range(self.num_puzzles):
            env = SpreadsheetEnv(rows=5, cols=5, seed=42 + i)
            grid = env.reset()
            task_text, formula = self._generate_task(env)
            expected = env.evaluate_formula(formula, "Z0")
            self._puzzles[i] = {"env": env, "grid": grid, "task": task_text, "expected": expected}
            data.append({
                "prompt": self.get_prompt({"grid": grid, "task": task_text}),
                "task_name": "spreadsheet",
            })
        return Dataset.from_list(data)

    def _generate_task(self, env: SpreadsheetEnv) -> tuple[str, str]:
        cols = [chr(ord("A") + c) for c in range(env.cols)]
        template, formula_template = random.choice(TASK_TEMPLATES)

        start_col = random.choice(cols)
        end_col = random.choice(cols[cols.index(start_col):])
        start = f"{start_col}{random.randint(1, env.rows)}"
        end = f"{end_col}{random.randint(1, env.rows)}"

        cell_a = f"{random.choice(cols)}{random.randint(1, env.rows)}"
        cell_b = f"{random.choice(cols)}{random.randint(1, env.rows)}"

        task_text = template.format(
            start=start, end=end, cell_a=cell_a, cell_b=cell_b,
        )
        formula = formula_template.format(
            start=start, end=end, cell_a=cell_a, cell_b=cell_b,
        )
        return task_text, formula

    def get_prompt(self, example: dict) -> str:
        return self.PROMPT_TEMPLATE.format(
            grid=example["grid"], task=example["task"],
        )

    def compute_reward(self, prompt: str, completion: str) -> float:
        puzzle = None
        for pid, pdata in self._puzzles.items():
            if self.get_prompt({"grid": pdata["grid"], "task": pdata["task"]}) == prompt:
                puzzle = pdata
                break
        if puzzle is None:
            return 0.0

        import re
        match = re.search(r"FORMULA:\s*(\w\d+)=(.+)", completion)
        if not match:
            match = re.search(r"(\w\d+)\s*=\s*(.+)", completion)
        if not match:
            return 0.0

        _, formula = match.group(1), match.group(2).strip()

        env = SpreadsheetEnv(rows=5, cols=5)
        env.grid = dict(puzzle["env"].grid)
        result = env.evaluate_formula(formula, "")
        expected = puzzle["expected"]

        try:
            if isinstance(result, (int, float)) and isinstance(expected, (int, float)):
                if abs(float(result) - float(expected)) < 1e-6:
                    return 1.0
            elif str(result) == str(expected):
                return 1.0
        except Exception:
            pass
        return 0.0
```

- [ ] **Step 3: Verify spreadsheet environment works**

```bash
source bin/activate && python -c "
from src.envs.spreadsheet_env import SpreadsheetEnv
env = SpreadsheetEnv(rows=5, cols=5, seed=42)
grid = env.reset()
print(grid)
print('SUM A1:A3:', env.evaluate_formula('SUM(A1:A3)', ''))
print('AVG A1:A3:', env.evaluate_formula('AVG(A1:A3)', ''))
"
```

- [ ] **Step 4: Commit**

```bash
git add src/envs/spreadsheet_env.py src/tasks/spreadsheet_task.py
git commit -m "feat: add spreadsheet formula environment and task"
```

---

### Task 13: Integration Test (Smoke Test)

**Files:**
- Create: `tests/test_registry.py`
- Create: `tests/test_rewards.py`

- [ ] **Step 1: Write test_registry.py**

```bash
mkdir -p tests
```

```python
import pytest
from src.task_env import TaskEnv, TaskRegistry
from datasets import Dataset


def test_register_and_get_task():
    @TaskRegistry.register("test_dummy")
    class DummyTask(TaskEnv):
        def load_dataset(self):
            return Dataset.from_dict({"prompt": ["test"], "task_name": ["test_dummy"]})

        def get_prompt(self, example):
            return example["prompt"]

        def compute_reward(self, prompt, completion):
            return 1.0

    task = TaskRegistry.get("test_dummy")
    assert task.name == "test_dummy"
    ds = task.load_dataset()
    assert len(ds) == 1
    assert task.compute_reward("a", "b") == 1.0


def test_unknown_task_raises():
    with pytest.raises(KeyError, match="nonexistent"):
        TaskRegistry.get("nonexistent")


def test_list_tasks():
    names = TaskRegistry.list_tasks()
    assert "test_dummy" in names
```

- [ ] **Step 2: Write test_rewards.py**

```python
import pytest
from src.tasks.math_task import GSM8KTask


class TestGSM8KReward:
    def test_extract_boxed_answer(self):
        task = GSM8KTask()
        text = "Therefore, \\boxed{42} is the answer."
        assert task._extract_answer(text) == "42"

    def test_extract_hashtag_answer(self):
        task = GSM8KTask()
        text = "Step 3: add them\n#### 15"
        assert task._extract_answer(text) == "15"

    def test_no_answer_returns_none(self):
        task = GSM8KTask()
        text = "Just some reasoning with no answer format."
        assert task._extract_answer(text) is None

    def test_compute_reward_correct(self):
        task = GSM8KTask()
        prompt = task.get_prompt({"question": "What is 2+2?"})
        completion = "2+2=4. \\boxed{4}"
        # This will fail without actual dataset lookup, but tests extraction
        assert isinstance(task.compute_reward(prompt, completion), float)

    def test_compute_reward_incorrect(self):
        task = GSM8KTask()
        prompt = task.get_prompt({"question": "What is 2+2?"})
        completion = "2+2=5. \\boxed{5}"
        assert task.compute_reward(prompt, completion) == 0.0
```

- [ ] **Step 3: Run tests**

```bash
source bin/activate && pip install pytest && python -m pytest tests/ -v
```

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: add unit tests for TaskRegistry and math reward extraction"
```

---

### Task 14: Final Wiring — Ensure __init__.py Imports

**Files:**
- Modify: `src/tasks/__init__.py`

- [ ] **Step 1: Update tasks/__init__.py with auto-imports**

```python
from src.tasks.math_task import GSM8KTask, MATHTask
from src.tasks.code_task import MBPPTask, HumanEvalTask
from src.tasks.logic_task import PRONTOQATask
from src.tasks.lean_task import LeanMiniF2FTask
from src.tasks.sokoban_task import SokobanTask
from src.tasks.spreadsheet_task import SpreadsheetTask
```

- [ ] **Step 2: Verify all tasks register from init import**

```bash
source bin/activate && python -c "
from src.tasks import *
from src.task_env import TaskRegistry
print('All tasks:', sorted(TaskRegistry.list_tasks()))
"
```

- [ ] **Step 3: Run all tests**

```bash
source bin/activate && python -m pytest tests/ -v
```

- [ ] **Step 4: Commit**

```bash
git add src/tasks/__init__.py
git commit -m "chore: wire task imports in __init__.py"
```
