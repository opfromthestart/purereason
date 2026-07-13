import json
import os
import urllib.request

import numpy as np
from datasets import Dataset

from src.task_env import TaskEnv, TaskRegistry

_ARCGI_ONE_URL = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data"
_ARCGI_TWO_URL = "https://raw.githubusercontent.com/arcprize/ARC-AGI-2/main/data"

_puzzle_outputs: dict[str, list[list[int]]] = {}


def _task_ids_cache_path(key: str) -> str:
    d = os.path.join(os.path.dirname(__file__), "..", "..", ".cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{key}_task_ids.json")


def _load_task_ids(data_url: str, cache_key: str) -> list[str]:
    cp = _task_ids_cache_path(cache_key)
    if os.path.exists(cp):
        with open(cp) as f:
            return json.load(f)

    for base, api_path in [
        ("https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data",
         "https://api.github.com/repos/fchollet/ARC-AGI/contents/data/training"),
        ("https://raw.githubusercontent.com/arcprize/ARC-AGI-2/main/data",
         "https://api.github.com/repos/arcprize/ARC-AGI-2/contents/data/training"),
    ]:
        if data_url.rstrip("/") == base.rstrip("/"):
            api_url = api_path
            break
    else:
        raise ValueError(f"Unknown data_url: {data_url}")

    req = urllib.request.Request(api_url, headers={"User-Agent": "python"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        items = json.loads(resp.read())
    ids = sorted([it["name"].replace(".json", "") for it in items if it["name"].endswith(".json")])
    with open(cp, "w") as f:
        json.dump(ids, f)
    return ids


def _ensure_data(data_url: str, cache_key: str) -> list[dict]:
    ids = _load_task_ids(data_url, cache_key)
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", ".cache", cache_key)
    os.makedirs(cache_dir, exist_ok=True)
    tasks = []
    for task_id in ids:
        fpath = os.path.join(cache_dir, f"{task_id}.json")
        if not os.path.exists(fpath):
            try:
                url = f"{data_url}/training/{task_id}.json"
                with urllib.request.urlopen(url, timeout=10) as resp:
                    data_b = resp.read()
                with open(fpath, "wb") as f:
                    f.write(data_b)
            except Exception:
                continue
        try:
            with open(fpath) as f:
                tasks.append(json.load(f))
        except Exception:
            continue
    return tasks


def _render_grid(grid: list[list[int]]) -> str:
    lines = []
    for row in grid:
        line = ""
        for v in row:
            line += str(v) if v < 10 else chr(ord("A") + v - 10)
        lines.append(line)
    return "\n".join(lines)


def _grids_equal(a, b) -> bool:
    return np.array_equal(np.array(a), np.array(b))


def _build_arc_task_class(data_url: str, cache_key: str, task_name: str):
    class _Task(TaskEnv):
        PROMPT_TEMPLATE = (
            "You are given input-output grid pairs that demonstrate a pattern transformation. "
            "The grids use digits 0-9 (and A-F for 10-15) to represent colors. "
            "Study the examples to infer the transformation rule, then apply it to the test input. "
            "Output ONLY the test output grid in the same format as the examples.\n\n"
            "{examples}\n\nTest Input:\n{test_input}\n\nTest Output:"
        )

        def __init__(self, num_tasks: int = 400, train_examples: int = 3):
            self.num_tasks = num_tasks
            self.train_examples = train_examples

        def load_dataset(self) -> Dataset:
            tasks = _ensure_data(data_url, cache_key)
            if self.num_tasks > 0:
                tasks = tasks[:self.num_tasks]
            data = []
            for task in tasks:
                train_pairs = task["train"]
                test_pairs = task["test"]
                if not test_pairs:
                    continue
                test_pair = test_pairs[0]

                n_examples = min(self.train_examples, len(train_pairs) - 1)
                if n_examples < 1:
                    n_examples = len(train_pairs)

                examples_text = ""
                for j in range(n_examples):
                    pair = train_pairs[j]
                    examples_text += (
                        f"Example {j + 1} Input:\n{_render_grid(pair['input'])}\n"
                        f"Example {j + 1} Output:\n{_render_grid(pair['output'])}\n\n"
                    )

                prompt = self.PROMPT_TEMPLATE.format(
                    examples=examples_text.strip(),
                    test_input=_render_grid(test_pair["input"]),
                )

                data.append({"prompt": prompt, "task_name": task_name})
                _puzzle_outputs[prompt] = test_pair["output"]
            return Dataset.from_list(data)

        def get_prompt(self, example: dict) -> str:
            return example["prompt"]

        def compute_reward(self, prompt: str, completion: str) -> float:
            expected = _puzzle_outputs.get(prompt)
            if expected is None:
                return 0.0
            h = len(expected)
            w = len(expected[0])
            predicted = _extract_grid(completion, h, w)
            if predicted is None:
                return 0.0
            return 1.0 if _grids_equal(predicted, expected) else 0.0

    return _Task


def _extract_grid(text: str, target_h: int, target_w: int) -> list[list[int]] | None:
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    grid = []
    for line in lines:
        row = []
        for ch in line:
            if ch.isdigit():
                row.append(int(ch))
            elif ch.upper() in "ABCDEF":
                row.append(10 + ord(ch.upper()) - ord("A"))
            elif ch in " \t,":
                continue
            else:
                break
        if not row:
            continue
        if len(row) != target_w:
            continue
        grid.append(row)
        if len(grid) == target_h:
            break
    if len(grid) == target_h and all(len(r) == target_w for r in grid):
        return grid
    return None


ArcAGIOneTask = _build_arc_task_class(_ARCGI_ONE_URL, "arc1", "arc_agi_one")
TaskRegistry.register("arc_agi_one")(ArcAGIOneTask)

ArcAGITwoTask = _build_arc_task_class(_ARCGI_TWO_URL, "arc2", "arc_agi_two")
TaskRegistry.register("arc_agi_two")(ArcAGITwoTask)
