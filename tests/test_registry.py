import pytest
from src.task_env import TaskEnv, TaskRegistry
from datasets import Dataset


def test_register_and_get_task():
    @TaskRegistry.register("test_dummy")
    class DummyTask(TaskEnv):
        def load_dataset(self):
            return Dataset.from_dict({"prompt": ["test"], "task_name": ["test_dummy"]})

        def get_prompt(self, example):
            return example["prompt"]

        def compute_reward(self, prompt, completion):
            return 1.0

    task = TaskRegistry.get("test_dummy")
    assert task.name == "test_dummy"
    ds = task.load_dataset()
    assert len(ds) == 1
    assert task.compute_reward("a", "b") == 1.0


def test_unknown_task_raises():
    with pytest.raises(KeyError, match="nonexistent"):
        TaskRegistry.get("nonexistent")


def test_list_tasks():
    names = TaskRegistry.list_tasks()
    assert "test_dummy" in names
