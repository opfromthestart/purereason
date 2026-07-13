from datasets import Dataset, concatenate_datasets

from src.config import TaskSamplingConfig
from src.task_env import TaskRegistry


def _wrap_chatml(tokenizer, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def _remap_prompt_keys(unused, old_to_new: dict[str, str]):
    to_remap = [
        ("src.tasks.math_task", ["_gsm8k_refs", "_math_refs"]),
        ("src.tasks.code_task", ["_mbpp_lookup", "_humaneval_lookup"]),
        ("src.tasks.logic_task", ["_prontoqa_refs"]),
        ("src.tasks.sokoban_task", ["_sokoban_puzzles"]),
        ("src.tasks.spreadsheet_task", ["_spreadsheet_puzzles"]),
        ("src.tasks.lean_task", ["_lean_lookup"]),
        ("src.tasks.knights_knaves_task", ["_knights_knaves_refs"]),
        ("src.tasks.block_world_task", ["_block_world_puzzles"]),
    ]
    for mod_name, attr_names in to_remap:
        try:
            mod = __import__(mod_name, fromlist=attr_names)
        except ImportError:
            continue
        for attr_name in attr_names:
            d = getattr(mod, attr_name, None)
            if isinstance(d, dict):
                for old_k, new_k in old_to_new.items():
                    if old_k in d:
                        d[new_k] = d.pop(old_k)


def build_mixed_dataset(config: TaskSamplingConfig, tokenizer=None) -> Dataset:
    samples = []
    for task_name, weight in config.tasks.items():
        try:
            task = TaskRegistry.get(task_name)
        except KeyError:
            print(f"[WARN] Task '{task_name}' not registered, skipping")
            continue
        if task.is_interactive:
            continue
        ds = task.load_dataset()
        n = min(len(ds), config.max_samples_per_task)
        ds = ds.select(range(n))

        if tokenizer is not None:
            old_to_new = {}
            for row in ds:
                old_to_new[row["prompt"]] = _wrap_chatml(tokenizer, row["prompt"])

            ds = ds.map(
                lambda x, t=tokenizer: {"prompt": _wrap_chatml(t, x["prompt"])},
                fn_kwargs={},
            )
            _remap_prompt_keys(task, old_to_new)

        samples.append(ds)
        print(f"[INFO] Loaded {n} examples from '{task_name}'")

    if not samples:
        raise ValueError("No task datasets loaded. Check task registry.")

    mixed = concatenate_datasets(samples)
    mixed = mixed.shuffle(seed=42)
    print(f"[INFO] Mixed dataset size: {len(mixed)}")
    return mixed
