from datasets import Dataset, concatenate_datasets

from src.config import TaskSamplingConfig
from src.task_env import TaskRegistry


def build_mixed_dataset(config: TaskSamplingConfig) -> Dataset:
    samples = []
    for task_name, weight in config.tasks.items():
        try:
            task = TaskRegistry.get(task_name)
        except KeyError:
            print(f"[WARN] Task '{task_name}' not registered, skipping")
            continue
        ds = task.load_dataset()
        n = min(len(ds), config.max_samples_per_task)
        ds = ds.select(range(n))
        samples.append(ds)
        print(f"[INFO] Loaded {n} examples from '{task_name}'")

    if not samples:
        raise ValueError("No task datasets loaded. Check task registry.")

    mixed = concatenate_datasets(samples)
    mixed = mixed.shuffle(seed=42)
    print(f"[INFO] Mixed dataset size: {len(mixed)}")
    return mixed
