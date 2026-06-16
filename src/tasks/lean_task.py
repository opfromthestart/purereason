import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from datasets import Dataset

from src.task_env import TaskEnv, TaskRegistry


@TaskRegistry.register("lean_minif2f")
class LeanMiniF2FTask(TaskEnv):
    PROMPT_TEMPLATE = (
        "Prove the following theorem in Lean 4. Output the complete proof "
        "as a sequence of tactics, one per line. Do not include any "
        "explanation.\n\n"
        "Theorem ({name}):\n{statement}\n\n"
        "Initial proof state:\n{state}\n\n"
        "Proof:"
    )

    def __init__(self):
        self._proofs: dict[str, str] | None = None
        self._statements: dict[str, str] | None = None

    def load_dataset(self) -> Dataset:
        from datasets import load_dataset
        ds = load_dataset(
            "charliemeyer2000/leandojo_benchmark_lean4_17_0",
            split="validation",
        )

        ds = ds.filter(lambda x: x["PROVABLE"] == 1)

        grouped = defaultdict(list)
        for row in ds:
            grouped[row["NAME"]].append(row)

        data = []
        proofs = {}
        statements = {}
        for name, rows in grouped.items():
            rows.sort(key=lambda r: r.get("STATE", ""))
            first = rows[0]
            statement = first["STATEMENT"]
            state = first["STATE"]
            proof = "\n".join(r["TACTIC"].strip() for r in rows)

            prompt = self.get_prompt({
                "name": name, "statement": statement, "state": state,
            })
            data.append({"prompt": prompt, "task_name": "lean_minif2f"})
            proofs[prompt] = proof
            statements[prompt] = statement

        self._proofs = proofs
        self._statements = statements
        return Dataset.from_list(data)

    def get_prompt(self, example: dict) -> str:
        return self.PROMPT_TEMPLATE.format(
            name=example.get("name", ""),
            statement=example.get("statement", ""),
            state=example.get("state", ""),
        )

    def _lean_available(self) -> bool:
        return shutil.which("lean") is not None

    def _verify_lean(self, statement: str, proof: str) -> bool:
        source = f"{statement} := by\n{proof}"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", delete=False,
        ) as f:
            f.write(source)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["lean", tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def compute_reward(self, prompt: str, completion: str) -> float:
        if self._proofs is None:
            self.load_dataset()

        expected = self._proofs.get(prompt)
        statement = self._statements.get(prompt)
        if expected is None:
            return 0.0

        predicted = completion.strip()
        if not predicted:
            return 0.0

        if self._lean_available():
            if self._verify_lean(statement, predicted):
                return 1.0

        pred_norm = " ".join(predicted.lower().split())
        exp_norm = " ".join(expected.lower().split())
        if pred_norm == exp_norm:
            return 1.0
        return 0.0
