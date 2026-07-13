import re

import numpy as np
from dataclasses import dataclass, field


_CELL_CHARS = "0123456789ABCDEF"


@dataclass
class ArcAGIGrid:
    cells: np.ndarray
    available_actions: list[int] = field(default_factory=list)

    @property
    def height(self) -> int:
        return self.cells.shape[0]

    @property
    def width(self) -> int:
        return self.cells.shape[1]

    def render(self) -> str:
        w = self.width
        h = self.height
        rows = []

        tens = "   " + "".join(str(x // 10) if x % 10 == 0 else " " for x in range(w))
        ones = "   " + "".join(str(x % 10) for x in range(w))
        rows.append(tens)
        rows.append(ones)

        for y in range(h):
            line = f"{y:02d} "
            for x in range(w):
                v = int(self.cells[y, x])
                line += _CELL_CHARS[min(v, 15)]
            rows.append(line)
        return "\n".join(rows)


def parse_action(text: str, available_actions: list[int]) -> tuple[int, dict | None]:
    text = text.strip().upper()
    text_clean = re.sub(r'\s+', ' ', text)

    m = re.search(r'ACTION\s*6.*?(\d+).*?(\d+)', text_clean)
    if m and 6 in available_actions:
        x = int(m.group(1))
        y = int(m.group(2))
        if 0 <= x <= 63 and 0 <= y <= 63:
            return (6, {"x": x, "y": y})

    m = re.search(r'ACTION\s*(\d)', text_clean)
    if m:
        action_id = int(m.group(1))
        if action_id in available_actions:
            return (action_id, None)

    return (0, None)


def create_arcade(environments_dir: str | None = None):
    import arc_agi
    if environments_dir:
        return arc_agi.Arcade(environments_dir=environments_dir)
    return arc_agi.Arcade()


def get_available_game_ids(arc) -> list[str]:
    games = arc.get_environments()
    return [g.game_id for g in games]
