import copy
import random


SYMBOLS = {"empty": " ", "wall": "#", "box": "$", "target": ".", "box_on_target": "*",
           "agent": "@", "agent_on_target": "+"}


class SokobanEnv:
    def __init__(self, width: int = 6, height: int = 6, num_boxes: int = 2, seed: int = 42):
        random.seed(seed)
        self.width = width
        self.height = height
        self.num_boxes = num_boxes
        self.grid = None
        self.init_grid = None
        self.agent_pos = (0, 0)
        self.reset()

    def reset(self) -> str:
        self._generate_puzzle()
        return self.render()

    def _generate_puzzle(self):
        self.grid = [[" " for _ in range(self.width)] for _ in range(self.height)]
        for x in range(self.width):
            self.grid[0][x] = "#"
            self.grid[self.height - 1][x] = "#"
        for y in range(self.height):
            self.grid[y][0] = "#"
            self.grid[y][self.width - 1] = "#"

        free_cells = [(y, x) for y in range(1, self.height - 1)
                      for x in range(1, self.width - 1)]
        random.shuffle(free_cells)

        self.agent_pos = free_cells.pop()
        gy, gx = self.agent_pos
        self.grid[gy][gx] = "@"

        for _ in range(self.num_boxes):
            if not free_cells:
                break
            by, bx = free_cells.pop()
            self.grid[by][bx] = "$"

        for _ in range(self.num_boxes):
            if not free_cells:
                break
            ty, tx = free_cells.pop()
            self.grid[ty][tx] = "." if self.grid[ty][tx] == " " else (
                "*" if self.grid[ty][tx] == "$" else self.grid[ty][tx]
            )

        self.init_grid = copy.deepcopy(self.grid)

    def render(self) -> str:
        lines = []
        for row in self.grid:
            lines.append("".join(row))
        return "\n".join(lines)

    def step(self, move: str) -> tuple[str, float, bool]:
        moves = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
        if move not in moves:
            return self.render(), self._reward(), False

        dy, dx = moves[move]
        ay, ax = self.agent_pos
        ny, nx = ay + dy, ax + dx

        if not (0 <= ny < self.height and 0 <= nx < self.width):
            return self.render(), self._reward(), False
        if self._is_wall(ny, nx):
            return self.render(), self._reward(), False

        if self._is_box(ny, nx):
            bny, bnx = ny + dy, nx + dx
            if not (0 <= bny < self.height and 0 <= bnx < self.width):
                return self.render(), self._reward(), False
            if self._is_wall(bny, bnx) or self._is_box(bny, bnx):
                return self.render(), self._reward(), False
            self._move_box((ny, nx), (bny, bnx))

        self._move_agent(self.agent_pos, (ny, nx))
        self.agent_pos = (ny, nx)
        return self.render(), self._reward(), self._is_done()

    def _is_wall(self, y: int, x: int) -> bool:
        return self.grid[y][x] == "#"

    def _is_box(self, y: int, x: int) -> bool:
        return self.grid[y][x] in ("$", "*")

    def _is_target(self, y: int, x: int) -> bool:
        return self.grid[y][x] in (".", "*", "+")

    def _move_agent(self, from_pos, to_pos):
        fy, fx = from_pos
        ty, tx = to_pos
        from_was_target = self._is_target(fy, fx) and self.grid[fy][fx] in ("+",)
        self.grid[fy][fx] = "." if from_was_target else " "
        to_is_target = self._is_target(ty, tx)
        self.grid[ty][tx] = "+" if to_is_target else "@"

    def _move_box(self, from_pos, to_pos):
        fy, fx = from_pos
        ty, tx = to_pos
        box_on_target = self.grid[fy][fx] == "*"
        self.grid[fy][fx] = "." if box_on_target else " "
        target_under = self.grid[ty][tx] == "."
        self.grid[ty][tx] = "*" if target_under else "$"

    def _reward(self) -> float:
        boxes_on_target = sum(
            1 for y in range(self.height) for x in range(self.width)
            if self.grid[y][x] == "*"
        )
        return boxes_on_target / self.num_boxes

    def _is_done(self) -> bool:
        return self._reward() >= 1.0

    @property
    def board(self) -> str:
        return self.render()
