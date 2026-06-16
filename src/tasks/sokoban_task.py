
from datasets import Dataset

from src.task_env import TaskEnv, TaskRegistry
from src.envs.sokoban_env import SokobanEnv


@TaskRegistry.register("sokoban")
class SokobanTask(TaskEnv):
    PROMPT_TEMPLATE = (
        "You control an agent (@) on a grid. Push boxes ($) onto targets (.) "
        "to score. You cannot walk through walls (#) or boxes. Moves: U D L R. "
        "Output your moves as a continuous sequence on one line.\n\n"
        "Board:\n{board}\n\nMoves:"
    )

    def __init__(self, num_puzzles: int = 500, max_moves: int = 50):
        self.num_puzzles = num_puzzles
        self.max_moves = max_moves
        self._puzzles = {}

    def load_dataset(self) -> Dataset:
        data = []
        for i in range(self.num_puzzles):
            env = SokobanEnv(width=6, height=6, num_boxes=2, seed=42 + i)
            board = env.reset()
            self._puzzles[i] = {"env": env, "board": board}
            data.append({"prompt": self.get_prompt({"board": board}), "task_name": "sokoban"})
        return Dataset.from_list(data)

    def get_prompt(self, example: dict) -> str:
        return self.PROMPT_TEMPLATE.format(board=example["board"])

    def compute_reward(self, prompt: str, completion: str) -> float:
        board = None
        for pid, puzzle in self._puzzles.items():
            if self.get_prompt({"board": puzzle["board"]}) == prompt:
                board = puzzle["board"]
                break
        if board is None:
            return 0.0

        env = SokobanEnv(width=6, height=6, num_boxes=2, seed=42)
        env.reset()

        moves = self._extract_moves(completion)
        for move in moves[:self.max_moves]:
            _, reward, done = env.step(move)
            if done:
                break
        return env._reward()

    def _extract_moves(self, text: str) -> list[str]:
        filtered = []
        for ch in text.strip().upper():
            if ch in "UDLR":
                filtered.append(ch)
        return filtered
