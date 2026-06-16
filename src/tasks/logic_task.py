import re

from datasets import Dataset

from src.task_env import TaskEnv, TaskRegistry


@TaskRegistry.register("prontoqa")
class PRONTOQATask(TaskEnv):
    PROMPT_TEMPLATE = (
        "Given the following facts, answer the question by reasoning step by "
        "step. Put your final answer on a new line after 'Answer:'.\n\n"
        "Facts: {context}\n\nQuestion: {question}\n\nOptions: {options}"
    )

    def load_dataset(self) -> Dataset:
        from datasets import load_dataset
        ds = load_dataset("renma/ProntoQA", split="validation")

        def _resolve_answer(row):
            row["prompt"] = self.get_prompt(row)
            row["task_name"] = "prontoqa"
            letter = row.get("answer", "").strip().upper()
            options = row.get("options", [])
            idx = ord(letter) - ord("A") if letter else -1
            row["answer_text"] = options[idx] if 0 <= idx < len(options) else letter
            return row

        ds = ds.map(_resolve_answer)
        return ds

    def get_prompt(self, example: dict) -> str:
        options = example.get("options", [])
        options_str = "\n".join(options) if options else ""
        return self.PROMPT_TEMPLATE.format(
            context=example.get("context", ""),
            question=example.get("question", ""),
            options=options_str,
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
            r"^(true|false|yes|no|A|B)$",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                return matches[-1].strip()
        lines = text.strip().split("\n")
        for line in reversed(lines):
            stripped = line.strip()
            if stripped.lower() in ("true", "false", "yes", "no", "a", "b"):
                return stripped
        return None

    def _lookup_reference(self, prompt: str) -> str | None:
        ds = load_dataset_for_prontoqa()
        for row in ds:
            if self.get_prompt(row) == prompt:
                return row.get("answer_text", "").strip()
        return None


def load_dataset_for_prontoqa():
    from datasets import load_dataset
    return load_dataset("renma/ProntoQA", split="validation")
