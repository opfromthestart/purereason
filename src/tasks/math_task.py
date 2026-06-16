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
