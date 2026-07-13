import json
from collections import deque

from datasets import Dataset

from src.task_env import TaskEnv, TaskRegistry
from src.envs.block_world_env import BlockWorldEnv

_block_world_puzzles: dict[str, dict] = {}
_block_world_interactive_puzzles: dict[str, dict] = {}


def _state_key(on: dict[str, str], holding: str | None = None) -> frozenset:
    items = list(on.items())
    if holding:
        items.append(("__holding__", holding))
    return frozenset(items)


def _find_shortest_plan(env: BlockWorldEnv, max_steps: int = 15) -> list[dict] | None:
    start = _state_key(env.on, env.holding)
    goal = _state_key(env.goal_on)
    if start == goal:
        return []

    queue: deque = deque([(start, [])])
    visited: set[frozenset] = {start}

    while queue:
        state_fs, plan = queue.popleft()
        if len(plan) >= max_steps:
            continue

        items = dict(state_fs)
        holding = items.pop("__holding__", None)
        on_dict = items

        supported = set(on_dict.values()) - {"table"}
        placed = set(on_dict.keys())
        clear = placed - supported
        if holding:
            clear.discard(holding)

        if holding is not None:
            for target in sorted(clear | {"TABLE"}):
                new_on = dict(on_dict)
                if target == "TABLE":
                    new_on[holding] = "table"
                else:
                    new_on[holding] = target
                new_key = _state_key(new_on)
                if new_key == goal:
                    return plan + [{"action": "place", "block": target}]
                if new_key not in visited:
                    visited.add(new_key)
                    queue.append((new_key, plan + [{"action": "place", "block": target}]))
        else:
            for block in sorted(clear):
                new_on = dict(on_dict)
                del new_on[block]
                new_state = _state_key(new_on, block)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, plan + [{"action": "pick", "block": block}]))

    return None


def _blocks_correct(on: dict[str, str], goal_on: dict[str, str]) -> int:
    count = 0
    for block, target in goal_on.items():
        if block in on and on[block] == target:
            count += 1
    return count


def _optimal_next_actions(env: BlockWorldEnv) -> set[str]:
    plan = _find_shortest_plan(env)
    if plan is None or len(plan) == 0:
        return set()
    return {json.dumps(plan[0], sort_keys=True)}


def _execute_plan(env: BlockWorldEnv, actions: list[dict]) -> bool:
    for action in actions:
        _, _, done = env.step(action)
        if done:
            return True
    return env._is_goal()


def _parse_json_actions(text: str) -> list[dict]:
    actions: list[dict] = []
    decoder = json.JSONDecoder()
    pos = 0
    text_stripped = text.strip()
    while pos < len(text_stripped):
        while pos < len(text_stripped) and text_stripped[pos] in (" ", "\n", "\r", "\t", ","):
            pos += 1
        if pos >= len(text_stripped):
            break
        if text_stripped[pos] == "{":
            try:
                obj, end = decoder.raw_decode(text_stripped, pos)
                if isinstance(obj, dict) and "action" in obj:
                    actions.append(obj)
                pos = end
            except json.JSONDecodeError:
                pos += 1
        else:
            pos += 1
    return actions


@TaskRegistry.register("block_world")
class BlockWorldTask(TaskEnv):
    PROMPT_TEMPLATE = (
        "You have a robot arm that can move blocks on a table. "
        "Blocks are labeled A, B, C, etc. "
        "The arm can pick up one clear block (nothing on top) and place it on "
        "another clear block or on the table.\n\n"
        "Output your plan as a sequence of JSON actions, one per line:\n"
        '{{"action": "pick", "block": "A"}}\n'
        '{{"action": "place", "block": "B"}}  (or "TABLE" to place on table)\n\n'
        "{initial_state}\n\n{goal_state}\n\nPlan:"
    )

    def __init__(self, num_puzzles: int = 500, num_blocks: int = 4, max_steps: int = 15):
        self.num_puzzles = num_puzzles
        self.num_blocks = num_blocks
        self.max_steps = max_steps

    def load_dataset(self) -> Dataset:
        data = []
        for i in range(self.num_puzzles):
            seed = 100 + i
            env = BlockWorldEnv(num_blocks=self.num_blocks, seed=seed)
            env.reset()
            prompt = self.get_prompt({
                "initial_state": env.render(),
                "goal_state": env.render_goal(),
            })
            _block_world_puzzles[prompt] = {
                "initial_on": dict(env.on),
                "goal_on": dict(env.goal_on),
                "num_blocks": self.num_blocks,
                "seed": seed,
            }
            data.append({"prompt": prompt, "task_name": "block_world"})
        return Dataset.from_list(data)

    def get_prompt(self, example: dict) -> str:
        return self.PROMPT_TEMPLATE.format(
            initial_state=example["initial_state"],
            goal_state=example["goal_state"],
        )

    def compute_reward(self, prompt: str, completion: str) -> float:
        puzzle = _block_world_puzzles.get(prompt)
        if puzzle is None:
            return 0.0

        env = BlockWorldEnv(num_blocks=puzzle["num_blocks"], seed=puzzle["seed"])
        env.reset()
        env.load_state(puzzle["initial_on"])

        actions = _parse_json_actions(completion)[:self.max_steps]
        if not actions:
            return 0.0

        return 1.0 if _execute_plan(env, actions) else 0.0


@TaskRegistry.register("block_world_interactive")
class BlockWorldInteractiveTask(TaskEnv):
    is_interactive = True

    PROMPT_TEMPLATE = (
        "You have a robot arm that can move blocks on a table. "
        "Blocks are labeled A, B, C, etc. "
        "The arm can pick up one clear block (nothing on top) and place it on "
        "another clear block or on the table.\n\n"
        "Output your next action as JSON:\n"
        '{{"action": "pick", "block": "A"}}\n'
        '{{"action": "place", "block": "B"}}  (or "TABLE" to place on table)\n\n'
        "{state_section}\n\nAction:"
    )

    def __init__(self, num_puzzles: int = 500, num_blocks: int = 4, max_steps: int = 15):
        self.num_puzzles = num_puzzles
        self.num_blocks = num_blocks
        self.max_steps = max_steps

    def load_dataset(self) -> Dataset:
        return Dataset.from_list([])

    def get_prompt(self, example: dict) -> str:
        return self.PROMPT_TEMPLATE.format(state_section=example.get("state_section", ""))

    def compute_reward(self, prompt: str, completion: str) -> float:
        puzzle = _block_world_interactive_puzzles.get(prompt)
        if puzzle is None:
            return 0.0
        actions = _parse_json_actions(completion)
        if not actions:
            return 0.0
        action_key = json.dumps(actions[0], sort_keys=True)
        return 1.0 if action_key in puzzle["optimal_actions"] else 0.0

    def get_initial_state(self, idx: int) -> dict:
        seed = 300 + idx
        env = BlockWorldEnv(num_blocks=self.num_blocks, seed=seed)
        env.reset()
        return {
            "on": dict(env.on),
            "goal_on": dict(env.goal_on),
            "holding": env.holding,
            "num_blocks": self.num_blocks,
            "seed": seed,
            "steps_taken": 0,
        }

    def get_initial_prompt(self, state: dict) -> str:
        env = self._env_from_state(state)
        state_text = f"{env.render()}\n\n{env.render_goal()}"
        return self.PROMPT_TEMPLATE.format(state_section=state_text)

    def process_action(self, state: dict, action_text: str) -> dict:
        env = self._env_from_state(state)
        initial_on = dict(state["on"])
        goal_on = dict(state["goal_on"])

        actions = _parse_json_actions(action_text)
        if not actions:
            observation = "Invalid action format. Output JSON with action and block."
            return {
                "observation": observation,
                "reward": 0.0,
                "done": False,
                "state": state,
            }

        action = actions[0]
        _, _, done = env.step(action)

        new_state = {
            "on": dict(env.on),
            "goal_on": goal_on,
            "holding": env.holding,
            "num_blocks": state["num_blocks"],
            "seed": state["seed"],
            "steps_taken": state["steps_taken"] + 1,
        }

        observation = f"Result:\n{env.render()}\n\nGoal:\n{env.render_goal()}"

        if done:
            reward = 1.0
        elif new_state["steps_taken"] >= self.max_steps:
            done = True
            reward = 0.0
        else:
            before_env = self._env_from_state({"on": initial_on, "goal_on": goal_on,
                                                "holding": None, "num_blocks": state["num_blocks"],
                                                "seed": state["seed"], "steps_taken": 0})
            before_env.holding = None
            plan_before = _find_shortest_plan(before_env, self.max_steps)
            plan_after = _find_shortest_plan(env, self.max_steps)
            dist_before = len(plan_before) if plan_before else self.max_steps * 2
            dist_after = len(plan_after) if plan_after else self.max_steps * 2
            progress = max(0.0, (dist_before - dist_after) / max(dist_before, 1))
            reward = progress * 0.1

        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "state": new_state,
        }

    def compute_episode_reward(self, final_state: dict) -> float:
        env = self._env_from_state(final_state)
        return 1.0 if env._is_goal() else 0.0

    def _env_from_state(self, state: dict) -> BlockWorldEnv:
        env = BlockWorldEnv(num_blocks=state["num_blocks"], seed=state["seed"])
        env.reset()
        env.load_state(state["on"])
        env.goal_on = dict(state["goal_on"])
        env.holding = state.get("holding")
        if env.holding:
            env._update_clear()
        return env
