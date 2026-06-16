import re

from datasets import Dataset

from src.task_env import TaskEnv, TaskRegistry


@TaskRegistry.register("lean_minif2f")
class LeanMiniF2FTask(TaskEnv):
    PROMPT_TEMPLATE = (
        "Predict the next tactic for this Lean 4 proof. Output only the "
        "tactic, nothing else.\n\n"
        "Theorem: {full_name}\n\n"
        "Current state:\n{state}\n\n"
        "Tactic:"
    )

    def load_dataset(self) -> Dataset:
        from datasets import load_dataset
        ds = load_dataset("cat-searcher/leandojo-benchmark-4-random", split="test")

        def _format(row):
            row["prompt"] = self.get_prompt(row)
            row["task_name"] = "lean_minif2f"
            return row

        ds = ds.map(_format)
        return ds

    def get_prompt(self, example: dict) -> str:
        return self.PROMPT_TEMPLATE.format(
            full_name=example.get("full_name", ""),
            state=example.get("state", ""),
        )

    @staticmethod
    def _strip_tags(text: str) -> str:
        return re.sub(r"</?a>", "", text).strip()

    def compute_reward(self, prompt: str, completion: str) -> float:
        ds = self.load_dataset()
        for row in ds:
            if row["prompt"] == prompt:
                expected = self._strip_tags(row.get("tactic", ""))
                predicted = completion.strip()
                if not predicted or not expected:
                    return 0.0
                if predicted.lower() == expected.lower():
                    return 1.0
                if "".join(predicted.split()) == "".join(expected.split()):
                    return 1.0
                return 0.0
        return 0.0

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
