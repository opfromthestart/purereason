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
