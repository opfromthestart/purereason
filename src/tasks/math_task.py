import re

from datasets import Dataset

from src.task_env import TaskEnv, TaskRegistry


@TaskRegistry.register("gsm8k")
class GSM8KTask(TaskEnv):
    PROMPT_TEMPLATE = (
        "Solve this math problem step by step. Put your final answer within "
        "\\boxed{{}}.\n\nProblem: {question}"
    )

    def __init__(self):
        self._refs: dict[str, str] = {}

    def load_dataset(self) -> Dataset:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        ds = ds.select_columns(["question", "answer"])

        data = []
        for row in ds:
            prompt = self.get_prompt(row)
            match = re.search(r"####\s*(-?\d+\.?\d*)", row["answer"])
            if match:
                self._refs[prompt] = match.group(1).strip()
            data.append({"prompt": prompt, "task_name": "gsm8k"})

        return Dataset.from_list(data)

    def get_prompt(self, example: dict) -> str:
        return self.PROMPT_TEMPLATE.format(question=example["question"])

    def compute_reward(self, prompt: str, completion: str) -> float:
        import sympy
        pred = self._extract_answer(completion)
        ref = self._refs.get(prompt)
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


@TaskRegistry.register("math")
class MATHTask(TaskEnv):
    PROMPT_TEMPLATE = (
        "Solve this math problem step by step. Put your final answer within "
        "\\boxed{{}}.\n\nProblem: {problem}"
    )

    def __init__(self):
        self._refs: dict[str, str] = {}

    def load_dataset(self) -> Dataset:
        from datasets import load_dataset
        ds = load_dataset("qwedsacf/competition_math", split="train")
        ds = ds.select_columns(["problem", "solution"])

        data = []
        for row in ds:
            prompt = self.get_prompt(row)
            matches = re.findall(r"\\boxed\{([^}]*)\}", row["solution"])
            answer = matches[-1].strip() if matches else ""
            self._refs[prompt] = answer
            data.append({"prompt": prompt, "task_name": "math"})

        return Dataset.from_list(data)

    def get_prompt(self, example: dict) -> str:
        return self.PROMPT_TEMPLATE.format(problem=example["problem"])

    def compute_reward(self, prompt: str, completion: str) -> float:
        import sympy
        pred = self._extract_answer(completion)
        ref = self._refs.get(prompt)
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
