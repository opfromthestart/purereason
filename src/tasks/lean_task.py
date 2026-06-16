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
