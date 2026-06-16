from abc import ABC, abstractmethod
from datasets import Dataset


class TaskEnv(ABC):
    name: str

    @abstractmethod
    def load_dataset(self) -> Dataset:
        ...

    @abstractmethod
    def get_prompt(self, example: dict) -> str:
        ...

    @abstractmethod
    def compute_reward(self, prompt: str, completion: str) -> float:
        ...


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
