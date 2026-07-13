from abc import ABC, abstractmethod
from typing import Any

from datasets import Dataset


class TaskEnv(ABC):
    name: str
    is_interactive: bool = False

    @abstractmethod
    def load_dataset(self) -> Dataset:
        ...

    @abstractmethod
    def get_prompt(self, example: Any) -> str:
        ...

    @abstractmethod
    def compute_reward(self, prompt: str, completion: str) -> float:
        ...

    def get_initial_state(self, idx: int) -> dict:
        raise NotImplementedError

    def get_initial_prompt(self, state: dict) -> str:
        raise NotImplementedError

    def process_action(self, state: dict, action_text: str) -> dict:
        raise NotImplementedError

    def compute_episode_reward(self, final_state: dict) -> float:
        raise NotImplementedError


class TaskRegistry:
    _tasks: dict[str, type[TaskEnv]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(task_cls: type[TaskEnv]):
            task_cls.name = name
            cls._tasks[name] = task_cls
            return task_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> TaskEnv:
        task_cls = cls._tasks.get(name)
        if task_cls is None:
            raise KeyError(f"Unknown task: {name}. Registered: {list(cls._tasks.keys())}")
        return task_cls()

    @classmethod
    def list_tasks(cls) -> list[str]:
        return list(cls._tasks.keys())
