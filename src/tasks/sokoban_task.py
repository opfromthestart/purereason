import copy

from datasets import Dataset

from src.task_env import TaskEnv, TaskRegistry
from src.envs.sokoban_env import SokobanEnv

_sokoban_puzzles: dict[str, dict] = {}


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

    def load_dataset(self) -> Dataset:
        data = []
        for i in range(self.num_puzzles):
            env = SokobanEnv(width=6, height=6, num_boxes=2, seed=42 + i)
            board = env.reset()
            prompt = self.get_prompt({"board": board})
            _sokoban_puzzles[prompt] = {"env": env, "board": board}
            data.append({"prompt": prompt, "task_name": "sokoban"})
        return Dataset.from_list(data)

    def get_prompt(self, example: dict) -> str:
        return self.PROMPT_TEMPLATE.format(board=example["board"])

    def compute_reward(self, prompt: str, completion: str) -> float:
        puzzle = _sokoban_puzzles.get(prompt)
        if puzzle is None:
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


@TaskRegistry.register("sokoban_interactive")
class SokobanInteractiveTask(TaskEnv):
    is_interactive = True

    def __init__(self, num_puzzles: int = 500, width: int = 6, height: int = 6,
                 num_boxes: int = 2, max_moves: int = 30):
        self.num_puzzles = num_puzzles
        self.width = width
        self.height = height
        self.num_boxes = num_boxes
        self.max_moves = max_moves

    def load_dataset(self) -> Dataset:
        return Dataset.from_list([])

    def get_prompt(self, example: dict) -> str:
        return example.get("prompt", "")

    def compute_reward(self, prompt: str, completion: str) -> float:
        return 0.0

    def get_initial_state(self, idx: int) -> dict:
        env = SokobanEnv(width=self.width, height=self.height,
                         num_boxes=self.num_boxes, seed=1000 + idx)
        board = env.reset()
        return {
            "grid": _serialize_grid(env.grid),
            "init_grid": _serialize_grid(env.init_grid),
            "agent_pos": env.agent_pos,
            "num_boxes": self.num_boxes,
            "width": self.width,
            "height": self.height,
            "seed": 1000 + idx,
            "moves_taken": 0,
            "boxes_on_target": 0,
        }

    def get_initial_prompt(self, state: dict) -> str:
        env = _env_from_state(state)
        return (
            "You control an agent (@) on a grid. Push boxes ($) onto targets (.) "
            "to score. You cannot walk through walls (#) or boxes.\n"
            "Output one move at a time: U D L R.\n\n"
            f"Board:\n{env.render()}\n\nMove:"
        )

    def process_action(self, state: dict, action_text: str) -> dict:
        env = _env_from_state(state)
        boxes_before = env._reward() * self.num_boxes

        move = self._extract_move(action_text)
        if move is None:
            return {
                "observation": "Invalid move. Output one of: U D L R",
                "reward": 0.0,
                "done": False,
                "state": state,
            }

        new_board, _step_reward, done = env.step(move)
        boxes_after = env._reward() * self.num_boxes

        new_state = {
            "grid": _serialize_grid(env.grid),
            "init_grid": state["init_grid"],
            "agent_pos": env.agent_pos,
            "num_boxes": state["num_boxes"],
            "width": state["width"],
            "height": state["height"],
            "seed": state["seed"],
            "moves_taken": state["moves_taken"] + 1,
            "boxes_on_target": int(boxes_after),
        }

        progress = (boxes_after - boxes_before) / max(self.num_boxes, 1)
        reward = max(0.0, progress * 0.1)

        if done:
            reward = 1.0 / max(1, new_state["moves_taken"])
            observation = f"Move {move}:\n{new_board}\n\nALL BOXES ON TARGETS! You win!"
        elif new_state["moves_taken"] >= self.max_moves:
            done = True
            observation = (f"Move {move}:\n{new_board}\n\nOut of moves. "
                          f"Boxes on target: {int(boxes_after)}/{self.num_boxes}")
        else:
            observation = (f"Move {move}:\n{new_board}\n\n"
                          f"Boxes on target: {int(boxes_after)}/{self.num_boxes}")

        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "state": new_state,
        }

    def compute_episode_reward(self, final_state: dict) -> float:
        if final_state["boxes_on_target"] < self.num_boxes:
            return 0.0
        return 1.0 / max(1, final_state.get("moves_taken", 1))

    def _extract_move(self, text: str) -> str | None:
        for ch in text.strip().upper():
            if ch in "UDLR":
                return ch
        return None


def _serialize_grid(grid: list[list[str]]) -> list[str]:
    return ["".join(row) for row in grid]


def _env_from_state(state: dict) -> SokobanEnv:
    env = SokobanEnv(width=state["width"], height=state["height"],
                     num_boxes=state["num_boxes"], seed=state["seed"])
    env.reset()
    h, w = state["height"], state["width"]
    for y in range(h):
        for x in range(w):
            env.grid[y][x] = state["grid"][y][x]
    env.init_grid = [list(s) for s in state["init_grid"]]
    env.agent_pos = tuple(state["agent_pos"])
    return env
