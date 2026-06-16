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
