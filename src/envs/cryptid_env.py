import random
from dataclasses import dataclass, field
from enum import Enum


class Terrain(Enum):
    DESERT = "D"
    FOREST = "F"
    WATER = "W"
    MOUNTAIN = "M"
    SWAMP = "S"


class Territory(Enum):
    BEAR = "bear"
    COUGAR = "cougar"


class StructureType(Enum):
    STANDING_STONE = "stone"
    ABANDONED_SHACK = "shack"


class StructureColor(Enum):
    WHITE = "white"
    GREEN = "green"
    BLUE = "blue"
    BLACK = "black"


@dataclass
class Clue:
    clue_type: str
    params: dict = field(default_factory=dict)

    def matches(self, x: int, y: int, board: "Board") -> bool:
        t = self.clue_type
        p = self.params
        if t == "on_terrain":
            return board.get_terrain(x, y) in p["terrains"]
        elif t == "within_n_of_terrain":
            return board.min_dist_to_terrain(x, y, p["terrain"]) <= p["n"]
        elif t == "within_n_of_territory":
            return board.min_dist_to_territory(x, y, p["territory"]) <= p["n"]
        elif t == "within_n_of_structure":
            return board.min_dist_to_structure(x, y, p["structure_type"]) <= p["n"]
        elif t == "not_within_n_of_terrain":
            return board.min_dist_to_terrain(x, y, p["terrain"]) > p["n"]
        elif t == "not_within_n_of_structure":
            return board.min_dist_to_structure(x, y, p["structure_type"]) > p["n"]
        elif t == "on_one_of_two_terrains":
            return board.get_terrain(x, y) in [p["t1"], p["t2"]]
        return False

    def describe(self) -> str:
        t = self.clue_type
        p = self.params
        if t == "on_terrain":
            return f"The habitat is on {p['terrains'][0].value} terrain."
        elif t == "on_one_of_two_terrains":
            return f"The habitat is on {p['t1'].value} or {p['t2'].value}."
        elif t == "within_n_of_terrain":
            return f"The habitat is within {p['n']} space(s) of {p['terrain'].value}."
        elif t == "within_n_of_territory":
            return f"The habitat is within {p['n']} space(s) of {p['territory'].value} territory."
        elif t == "within_n_of_structure":
            return f"The habitat is within {p['n']} space(s) of a {p['structure_type'].value}."
        elif t == "not_within_n_of_terrain":
            return f"The habitat is NOT within {p['n']} space(s) of {p['terrain'].value}."
        elif t == "not_within_n_of_structure":
            return f"The habitat is NOT within {p['n']} space(s) of a {p['structure_type'].value}."
        return "Unknown clue."


@dataclass
class Cell:
    terrain: Terrain = Terrain.DESERT
    bear_territory: bool = False
    cougar_territory: bool = False
    structure_type: StructureType | None = None
    structure_color: StructureColor | None = None


class Board:
    def __init__(self, width: int = 5, height: int = 5, seed: int = 42):
        random.seed(seed)
        self.width = width
        self.height = height
        self.cells: list[list[Cell]] = []
        self._generate()

    def _generate(self):
        terrains = list(Terrain)
        self.cells = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                cell = Cell(terrain=random.choice(terrains))
                if random.random() < 0.3:
                    cell.bear_territory = True
                if random.random() < 0.3 and not cell.bear_territory:
                    cell.cougar_territory = True
                if random.random() < 0.25:
                    cell.structure_type = random.choice(list(StructureType))
                    cell.structure_color = random.choice(list(StructureColor))
                row.append(cell)
            self.cells.append(row)

    def get_terrain(self, x: int, y: int) -> Terrain:
        return self.cells[y][x].terrain

    def get_cell(self, x: int, y: int) -> Cell:
        return self.cells[y][x]

    def min_dist_to_terrain(self, x: int, y: int, terrain: Terrain) -> int:
        best = 999
        for cy in range(self.height):
            for cx in range(self.width):
                if self.cells[cy][cx].terrain == terrain:
                    d = abs(x - cx) + abs(y - cy)
                    if d < best:
                        best = d
        return best

    def min_dist_to_territory(self, x: int, y: int, territory: Territory) -> int:
        best = 999
        for cy in range(self.height):
            for cx in range(self.width):
                cell = self.cells[cy][cx]
                if (territory == Territory.BEAR and cell.bear_territory) or \
                   (territory == Territory.COUGAR and cell.cougar_territory):
                    d = abs(x - cx) + abs(y - cy)
                    if d < best:
                        best = d
        return best

    def min_dist_to_structure(self, x: int, y: int, stype: StructureType) -> int:
        best = 999
        for cy in range(self.height):
            for cx in range(self.width):
                if self.cells[cy][cx].structure_type == stype:
                    d = abs(x - cx) + abs(y - cy)
                    if d < best:
                        best = d
        return best

    def render_cell(self, x: int, y: int, cubes: set, discs: set, my_idx: int) -> str:
        cell = self.cells[y][x]
        parts = [cell.terrain.value]
        if cell.bear_territory:
            parts.append("B")
        if cell.cougar_territory:
            parts.append("C")
        if cell.structure_type:
            st = "S" if cell.structure_type == StructureType.STANDING_STONE else "A"
            sc = cell.structure_color.value[0] if cell.structure_color else "?"
            parts.append(f"{st}{sc}")
        for pi in range(4):
            if (pi, x, y) in cubes:
                parts.append(f"X{pi}")
            if (pi, x, y) in discs:
                parts.append(f"O{pi}")
        return " ".join(parts)

    def all_spaces(self) -> list[tuple[int, int]]:
        return [(x, y) for y in range(self.height) for x in range(self.width)]


def _generate_clue(habitat: tuple[int, int], board: Board) -> Clue:
    x, y = habitat
    clue_types = [
        "on_terrain",
        "on_one_of_two_terrains",
        "within_n_of_terrain",
        "within_n_of_territory",
        "within_n_of_structure",
        "not_within_n_of_terrain",
        "not_within_n_of_structure",
    ]
    for _ in range(100):
        ct = random.choice(clue_types)
        if ct == "on_terrain":
            t = board.get_terrain(x, y)
            return Clue(clue_type=ct, params={"terrains": [t]})
        elif ct == "on_one_of_two_terrains":
            others = [t for t in Terrain if t != board.get_terrain(x, y)]
            t2 = random.choice(others)
            return Clue(clue_type=ct, params={"t1": board.get_terrain(x, y), "t2": t2})
        elif ct == "within_n_of_terrain":
            terrains = list(Terrain)
            random.shuffle(terrains)
            for t in terrains:
                for n in [1, 2]:
                    c = Clue(clue_type=ct, params={"terrain": t, "n": n})
                    if c.matches(x, y, board) and not _all_match(c, board):
                        return c
        elif ct == "within_n_of_territory":
            territories = list(Territory)
            random.shuffle(territories)
            for territory in territories:
                for n in [1, 2]:
                    c = Clue(clue_type=ct, params={"territory": territory, "n": n})
                    if c.matches(x, y, board) and not _all_match(c, board):
                        return c
        elif ct == "within_n_of_structure":
            stypes = list(StructureType)
            random.shuffle(stypes)
            for st in stypes:
                for n in [1, 2]:
                    c = Clue(clue_type=ct, params={"structure_type": st, "n": n})
                    if c.matches(x, y, board) and not _all_match(c, board):
                        return c
        elif ct == "not_within_n_of_terrain":
            for t in Terrain:
                for n in [1, 2]:
                    c = Clue(clue_type=ct, params={"terrain": t, "n": n})
                    if c.matches(x, y, board) and not _all_match(c, board):
                        return c
        elif ct == "not_within_n_of_structure":
            for st in StructureType:
                for n in [1, 2]:
                    c = Clue(clue_type=ct, params={"structure_type": st, "n": n})
                    if c.matches(x, y, board) and not _all_match(c, board):
                        return c
    return Clue(clue_type="on_terrain", params={"terrains": [board.get_terrain(x, y)]})


def _all_match(clue: Clue, board: Board) -> bool:
    return all(clue.matches(x, y, board) for x, y in board.all_spaces())


def _generate_all_clues(board: Board, habitat: tuple[int, int], num_players: int) -> list[Clue]:
    clues: list[Clue] = []
    attempts = 0
    while len(clues) < num_players and attempts < 500:
        clue = _generate_clue(habitat, board)
        combined = clues + [clue]
        matches = _count_joint_matches(combined, board)
        if 1 <= matches <= max(10, board.width * board.height // 3):
            if clue not in clues:
                clues.append(clue)
        attempts += 1
    while len(clues) < num_players:
        clues.append(_generate_clue(habitat, board))
    return clues


def _count_joint_matches(clues: list[Clue], board: Board) -> int:
    count = 0
    for x, y in board.all_spaces():
        if all(c.matches(x, y, board) for c in clues):
            count += 1
    return count
