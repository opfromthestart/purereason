import re
import subprocess
import sys
import tempfile
from pathlib import Path

from datasets import Dataset

from src.task_env import TaskEnv, TaskRegistry


def _run_code(code: str) -> tuple[int, int]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
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


_mbpp_lookup: dict[str, dict] = {}
_humaneval_lookup: dict[str, str] = {}


@TaskRegistry.register("mbpp")
class MBPPTask(TaskEnv):
    PROMPT_TEMPLATE = (
        "Complete the following Python function. Return only the function "
        "body, no explanation.\n\n{description}\n\n```python\n{code}\n```"
    )

    def load_dataset(self) -> Dataset:
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/mbpp", "full", split="train")

        data = []
        for row in ds:
            prompt = self.get_prompt(row)
            test_list = row.get("test_list", [])
            if isinstance(test_list, str):
                test_list = [test_list]
            imports = row.get("test_imports", "")
            if isinstance(imports, list):
                imports = "\n".join(imports)
            _mbpp_lookup[prompt] = {
                "test_list": test_list,
                "test_imports": imports,
            }
            data.append({"prompt": prompt, "task_name": "mbpp"})

        return Dataset.from_list(data)

    def get_prompt(self, example: dict) -> str:
        return self.PROMPT_TEMPLATE.format(
            description=example.get("text", example.get("description", "")),
            code=example["code"],
        )

    def compute_reward(self, prompt: str, completion: str) -> float:
        info = _mbpp_lookup.get(prompt)
        if info is None:
            return 0.0

        code = _extract_code_block(completion)
        test_list = info["test_list"]
        if not test_list:
            return 0.0

        prefix = (info["test_imports"] + "\n\n") if info["test_imports"] else ""
        passed = 0
        for test in test_list:
            full = prefix + code + "\n\n" + test
            p, _ = _run_code(full)
            passed += p
        return passed / len(test_list)


@TaskRegistry.register("humaneval")
class HumanEvalTask(TaskEnv):
    PROMPT_TEMPLATE = "{prompt}"

    def load_dataset(self) -> Dataset:
        from datasets import load_dataset
        ds = load_dataset("openai/openai_humaneval", split="test")

        data = []
        for row in ds:
            prompt = self.get_prompt(row)
            _humaneval_lookup[prompt] = row.get("test", "")
            data.append({"prompt": prompt, "task_name": "humaneval"})

        return Dataset.from_list(data)

    def get_prompt(self, example: dict) -> str:
        return self.PROMPT_TEMPLATE.format(prompt=example["prompt"])

    def compute_reward(self, prompt: str, completion: str) -> float:
        test_code = _humaneval_lookup.get(prompt)
        if test_code is None or not test_code:
            return 0.0

        body = _extract_code_block(completion)
        body_lines = body.strip().split("\n")
        indented = "\n".join("    " + line.lstrip() for line in body_lines)
        full_func = prompt + "\n" + indented
        full_code = full_func + "\n\n" + test_code
        passed, total = _run_code(full_code)
        return passed / total if total > 0 else 0.0
