import hashlib
import json
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from datasets import Dataset

from src.task_env import TaskEnv, TaskRegistry


MATHLIB_SRC = "/tmp/lean_verify/.lake/packages/mathlib"
MATHLIB_PROJECT = "/tmp/lean_verify"
MATHLIB_CACHE = "/tmp/mathlib_theorems_cache.json"


CORE_THEOREMS = [
    ("add_comm", "theorem add_comm (a b : Nat) : a + b = b + a := by", "omega"),
    ("add_assoc", "theorem add_assoc (a b c : Nat) : (a + b) + c = a + (b + c) := by", "omega"),
    ("add_zero", "theorem add_zero (a : Nat) : a + 0 = a := by", "simp"),
    ("zero_add", "theorem zero_add (a : Nat) : 0 + a = a := by", "simp"),
    ("mul_comm", "theorem mul_comm (a b : Nat) : a * b = b * a := by",
     "induction a generalizing b with\n  | zero => simp\n  | succ a ih => simp [Nat.succ_mul, Nat.mul_succ, ih]"),
    ("mul_assoc", "theorem mul_assoc (a b c : Nat) : (a * b) * c = a * (b * c) := by",
     "induction a generalizing b c with\n  | zero => simp\n  | succ a ih => simp [Nat.succ_mul, Nat.add_mul, ih]"),
    ("mul_add", "theorem mul_add (a b c : Nat) : a * (b + c) = a * b + a * c := by",
     "induction a generalizing b c with\n  | zero => simp\n  | succ a ih => simp [Nat.succ_mul, ih, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]"),
    ("add_comm_rev", "theorem add_comm_rev (a b c : Nat) : a + b + c = a + c + b := by", "omega"),
    ("add_shift", "theorem add_shift (a b c : Nat) : (a + b) + c = (a + c) + b := by", "omega"),
    ("mul_one", "theorem mul_one (a : Nat) : a * 1 = a := by", "simp"),
    ("one_mul", "theorem one_mul (a : Nat) : 1 * a = a := by", "simp"),
    ("mul_zero", "theorem mul_zero (a : Nat) : a * 0 = 0 := by", "simp"),
    ("zero_mul", "theorem zero_mul (a : Nat) : 0 * a = 0 := by", "simp"),
    ("add_left_cancel", "theorem add_left_cancel (a b c : Nat) (h : a + b = a + c) : b = c := by", "omega"),
    ("add_right_cancel", "theorem add_right_cancel (a b c : Nat) (h : b + a = c + a) : b = c := by", "omega"),
    ("and_comm", "theorem and_comm (a b : Bool) : (a && b) = (b && a) := by", "cases a <;> cases b <;> rfl"),
    ("or_comm", "theorem or_comm (a b : Bool) : (a || b) = (b || a) := by", "cases a <;> cases b <;> rfl"),
    ("not_not", "theorem not_not (a : Bool) : (!(!a)) = a := by", "cases a <;> rfl"),
    ("and_true", "theorem and_true (a : Bool) : (a && true) = a := by", "cases a <;> rfl"),
    ("or_false", "theorem or_false (a : Bool) : (a || false) = a := by", "cases a <;> rfl"),
    ("length_append", "theorem length_append (xs ys : List α) : (xs ++ ys).length = xs.length + ys.length := by", "induction xs <;> simp [*]; omega"),
    ("reverse_reverse", "theorem reverse_reverse (xs : List α) : xs.reverse.reverse = xs := by", "induction xs <;> simp [*]"),
    ("map_id", "theorem map_id (xs : List α) : xs.map id = xs := by", "induction xs <;> simp [*]"),
    ("list_append_nil", "theorem append_nil (xs : List α) : xs ++ [] = xs := by",
     "induction xs with\n  | nil => rfl\n  | cons x xs ih => simp [ih]"),
    ("list_nil_append", "theorem nil_append (xs : List α) : [] ++ xs = xs := by", "rfl"),
    ("list_append_assoc", "theorem append_assoc (xs ys zs : List α) : (xs ++ ys) ++ zs = xs ++ (ys ++ zs) := by", "induction xs <;> simp [*]"),
    ("true_and", "theorem true_and (a : Bool) : (true && a) = a := by", "cases a <;> rfl"),
    ("false_or", "theorem false_or (a : Bool) : (false || a) = a := by", "cases a <;> rfl"),
    ("succ_add", "theorem succ_add (a b : Nat) : (Nat.succ a) + b = Nat.succ (a + b) := by", "simp [Nat.succ_add]"),
    ("add_succ", "theorem add_succ (a b : Nat) : a + (Nat.succ b) = Nat.succ (a + b) := by", "omega"),
    ("pow_zero", "theorem pow_zero (a : Nat) : a ^ 0 = 1 := by", "simp"),
    ("pow_one", "theorem pow_one (a : Nat) : a ^ 1 = a := by", "simp"),
    ("sub_self", "theorem sub_self (a : Nat) : a - a = 0 := by", "omega"),
    ("max_self", "theorem max_self (a : Nat) : max a a = a := by", "simp"),
    ("min_self", "theorem min_self (a : Nat) : min a a = a := by", "simp"),
    ("mod_self", "theorem mod_self (a : Nat) (h : a > 0) : a % a = 0 := by", "simp [*]"),
    ("div_one", "theorem div_one (a : Nat) : a / 1 = a := by", "simp"),
    ("gcd_self", "theorem gcd_self (a : Nat) : Nat.gcd a a = a := by", "simp"),
    ("list_length_singleton", "theorem length_singleton (x : α) : [x].length = 1 := by", "simp"),
    ("list_mem_cons_self", "theorem mem_cons_self (x : α) (xs : List α) : x ∈ x :: xs := by", "simp"),
    ("list_map_append", "theorem map_append (f : α → β) (xs ys : List α) : (xs ++ ys).map f = xs.map f ++ ys.map f := by", "induction xs <;> simp [*]"),
    ("neg_neg", "theorem neg_neg (a : Int) : -(-a) = a := by", "omega"),
    ("add_comm_int", "theorem add_comm_int (a b : Int) : a + b = b + a := by", "omega"),
    ("sub_eq_add_neg", "theorem sub_eq_add_neg (a b : Int) : a - b = a + (-b) := by", "omega"),
]


def _extract_mathlib_theorems() -> list[dict]:
    cache = Path(MATHLIB_CACHE)
    if cache.exists():
        return json.loads(cache.read_text())

    if not Path(MATHLIB_SRC).exists():
        print(f"[WARN] Mathlib source not found at {MATHLIB_SRC}")
        return []

    raw = []
    for fpath in Path(MATHLIB_SRC).rglob("Mathlib/**/*.lean"):
        try:
            content = fpath.read_text()
        except Exception:
            continue

        for m in re.finditer(
            r"^theorem\s+(\S+)\s*(.*?):=", content, re.MULTILINE | re.DOTALL,
        ):
            name = m.group(1)
            rest = m.group(2).strip().rstrip(":")
            full = f"theorem {name} {rest}".strip()
            if len(full) < 30 or len(full) > 400:
                continue
            if any(c in full for c in "⋆≃≅≈₀₁₂₃₄₅₆₇₈₉"):
                continue
            rel = fpath.relative_to(MATHLIB_SRC)
            mod = str(rel.with_suffix("")).replace("/", ".")
            raw.append({"name": name, "statement": full, "module": mod})

        if len(raw) >= 10000:
            break

    random.Random(42).shuffle(raw)
    theorems = raw[:500]

    cache.write_text(json.dumps(theorems))
    print(f"[INFO] Cached {len(theorems)} mathlib theorem statements")
    return theorems


@TaskRegistry.register("lean_minif2f")
class LeanMiniF2FTask(TaskEnv):
    PROMPT_TEMPLATE = (
        "Prove the following theorem in Lean 4. Output only the proof "
        "(the body of `by`), nothing else.\n\n"
        "```lean\n{statement}\n```"
    )

    def __init__(self):
        self._lookup: dict[str, dict] = {}

    def load_dataset(self) -> Dataset:
        data = []

        for name, statement, _ in CORE_THEOREMS:
            prompt = self.get_prompt({"statement": statement})
            self._lookup[prompt] = {"statement": statement, "module": None}
            data.append({"prompt": prompt, "task_name": "lean_minif2f"})

        for t in _extract_mathlib_theorems():
            prompt = self.get_prompt({"statement": t["statement"]})
            self._lookup[prompt] = {"statement": t["statement"], "module": t["module"]}
            data.append({"prompt": prompt, "task_name": "lean_minif2f"})

        return Dataset.from_list(data)

    def get_prompt(self, example: dict) -> str:
        return self.PROMPT_TEMPLATE.format(statement=example["statement"])

    def _lean_available(self) -> bool:
        return shutil.which("lean") is not None

    def _mathlib_available(self) -> bool:
        return Path(MATHLIB_PROJECT, ".lake").exists()

    @staticmethod
    def _is_trivial(proof: str) -> bool:
        stripped = " ".join(proof.lower().split())
        if stripped in ("sorry", "admit", "exact sorry", "apply sorry"):
            return True
        if stripped.startswith("sorry"):
            return True
        return False

    def _verify_lean(self, statement: str, proof: str, module: str | None) -> bool:
        if self._is_trivial(proof):
            return False

        uid = hashlib.sha1(statement.encode()).hexdigest()[:8]
        safe_stmt = statement.replace("theorem ", f"theorem thm_{uid}_", 1)

        if module and self._mathlib_available():
            source = f'import {module}\n\n{safe_stmt} := by\n  {proof}'
        else:
            source = f"{safe_stmt}\n  {proof}"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", delete=False, dir="/tmp",
        ) as f:
            f.write(source)
            tmp_path = f.name

        try:
            if module and self._mathlib_available():
                result = subprocess.run(
                    ["lake", "env", "lean", tmp_path],
                    capture_output=True, text=True, timeout=30,
                    cwd=MATHLIB_PROJECT,
                )
            else:
                result = subprocess.run(
                    ["lean", tmp_path],
                    capture_output=True, text=True, timeout=10,
                )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def compute_reward(self, prompt: str, completion: str) -> float:
        predicted = completion.strip()
        if not predicted:
            return 0.0

        info = self._lookup.get(prompt)
        if info is None:
            return 0.0

        if not self._lean_available():
            return 0.0

        if self._verify_lean(info["statement"], predicted, info.get("module")):
            return 1.0
        return 0.0
