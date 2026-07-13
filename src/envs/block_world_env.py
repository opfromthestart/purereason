import copy
import json
import random
import re


class BlockWorldEnv:
    def __init__(self, num_blocks: int = 4, seed: int = 42):
        random.seed(seed)
        self.num_blocks = num_blocks
        self.block_names = [chr(ord("A") + i) for i in range(num_blocks)]
        self.on: dict[str, str] = {}
        self.clear: set[str] = set()
        self.holding: str | None = None
        self.goal_on: dict[str, str] = {}
        self.steps_taken: int = 0

    def reset(self) -> str:
        self._generate_puzzle()
        self.steps_taken = 0
        return self.render()

    def _generate_puzzle(self):
        for _ in range(200):
            goal = self._random_state()
            initial = self._random_state()
            if initial != goal:
                self.goal_on = goal
                self.on = initial
                break
        else:
            self.goal_on = {b: "table" for b in self.block_names}
            self.on = {b: "table" for b in self.block_names}
        self._update_clear()
        self.holding = None

    def _random_state(self) -> dict[str, str]:
        blocks = list(self.block_names)
        random.shuffle(blocks)
        state: dict[str, str] = {}
        supported: set[str] = set()
        for block in blocks:
            candidates = ["table"]
            for other in blocks:
                if other != block and other not in supported:
                    candidates.append(other)
            target = random.choice(candidates)
            state[block] = target
            if target != "table":
                supported.add(target)
        return state

    def _update_clear(self):
        supported = set(self.on.values()) - {"table"}
        placed = set(self.on.keys())
        held = {self.holding} if self.holding else set()
        self.clear = placed - supported - held

    def render(self) -> str:
        return self._render_state(self.on, "Current state")

    def render_goal(self) -> str:
        return self._render_state(self.goal_on, "Goal state")

    def _render_state(self, state: dict[str, str], label: str) -> str:
        stacks: list[list[str]] = []
        placed: set[str] = set()
        for block in self.block_names:
            if state.get(block) == "table" and block not in placed:
                stack = [block]
                placed.add(block)
                current = block
                while True:
                    found = False
                    for b, on in state.items():
                        if on == current and b not in placed:
                            stack.append(b)
                            placed.add(b)
                            current = b
                            found = True
                            break
                    if not found:
                        break
                stacks.append(stack)

        if not stacks:
            return f"{label}:\n  (empty)\n=========="

        max_h = max(len(s) for s in stacks)
        lines = [f"{label}:"]
        for row in range(max_h - 1, -1, -1):
            line = " "
            for stack in stacks:
                if row < len(stack):
                    line += f"[{stack[row]}]"
                else:
                    line += "   "
            lines.append(line)
        table_width = len(stacks) * 3
        lines.append("=" * table_width)
        return "\n".join(lines)

    def step(self, action: dict) -> tuple[str, float, bool]:
        act = action.get("action", "")
        block = action.get("block", "").upper()

        if act == "pick":
            if self.holding is not None or block not in self.clear:
                return self.render(), 0.0, False
            self.holding = block
            del self.on[block]
            self._update_clear()
            self.steps_taken += 1
            return self.render(), 0.0, False

        elif act == "place":
            if self.holding is None:
                return self.render(), 0.0, False
            target = block
            if target == "TABLE":
                self.on[self.holding] = "table"
            elif target in self.clear:
                self.on[self.holding] = target
            else:
                return self.render(), 0.0, False
            self.holding = None
            self._update_clear()
            self.steps_taken += 1
            return self.render(), float(self._is_goal()), self._is_goal()

        return self.render(), 0.0, False

    def _is_goal(self) -> bool:
        if self.holding is not None:
            return False
        return self.on == self.goal_on

    def get_clear_blocks(self) -> list[str]:
        return sorted(self.clear)

    def load_state(self, state: dict[str, str]):
        self.on = copy.deepcopy(state)
        self.holding = None
        self._update_clear()
        self.steps_taken = 0
