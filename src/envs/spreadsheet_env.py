import random
import re
from typing import Any


class SpreadsheetEnv:
    def __init__(self, rows: int = 10, cols: int = 10, seed: int = 42):
        random.seed(seed)
        self.rows = rows
        self.cols = cols
        self.grid: dict[str, Any] = {}
        self.reset()

    def reset(self):
        self.grid = {}
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self._to_ref(r, c)
                self.grid[cell] = random.randint(1, 100)
        return self.render()

    def _to_ref(self, row: int, col: int) -> str:
        return f"{chr(ord('A') + col)}{row + 1}"

    def _parse_ref(self, ref: str) -> tuple[int, int]:
        col = ord(ref[0].upper()) - ord("A")
        row = int(ref[1:]) - 1
        return row, col

    def render(self) -> str:
        lines = []
        header = "     " + "  ".join(f"{chr(ord('A') + c):>4}" for c in range(self.cols))
        lines.append(header)
        lines.append("    " + "-" * (5 * self.cols))
        for r in range(self.rows):
            row_str = f"{r + 1:>2} |"
            for c in range(self.cols):
                cell = self._to_ref(r, c)
                val = self.grid.get(cell, "")
                row_str += f" {str(val):>4}"
            lines.append(row_str)
        return "\n".join(lines)

    def set(self, cell: str, value: Any):
        self.grid[cell.upper()] = value

    def get(self, cell: str) -> Any:
        return self.grid.get(cell.upper(), 0)

    def evaluate_formula(self, formula: str, target_cell: str) -> Any:
        formula = formula.strip()
        if formula.startswith("="):
            formula = formula[1:]

        sum_match = re.match(r"SUM\((\w\d+):(\w\d+)\)", formula, re.IGNORECASE)
        if sum_match:
            start, end = sum_match.group(1), sum_match.group(2)
            sr, sc = self._parse_ref(start)
            er, ec = self._parse_ref(end)
            total = 0
            for r in range(min(sr, er), max(sr, er) + 1):
                for c in range(min(sc, ec), max(sc, ec) + 1):
                    val = self.get(self._to_ref(r, c))
                    if isinstance(val, (int, float)):
                        total += val
            return total

        avg_match = re.match(r"AVG\((\w\d+):(\w\d+)\)", formula, re.IGNORECASE)
        if avg_match:
            start, end = avg_match.group(1), avg_match.group(2)
            sr, sc = self._parse_ref(start)
            er, ec = self._parse_ref(end)
            total = 0
            count = 0
            for r in range(min(sr, er), max(sr, er) + 1):
                for c in range(min(sc, ec), max(sc, ec) + 1):
                    val = self.get(self._to_ref(r, c))
                    if isinstance(val, (int, float)):
                        total += val
                        count += 1
            return total / count if count > 0 else 0

        count_match = re.match(r"COUNT\((\w\d+):(\w\d+)\)", formula, re.IGNORECASE)
        if count_match:
            start, end = count_match.group(1), count_match.group(2)
            sr, sc = self._parse_ref(start)
            er, ec = self._parse_ref(end)
            count = 0
            for r in range(min(sr, er), max(sr, er) + 1):
                for c in range(min(sc, ec), max(sc, ec) + 1):
                    val = self.get(self._to_ref(r, c))
                    if isinstance(val, (int, float)):
                        count += 1
            return count

        if_match = re.match(r"IF\((.+),(.+),(.+)\)", formula, re.IGNORECASE)
        if if_match:
            cond, true_val, false_val = if_match.group(1), if_match.group(2), if_match.group(3)
            cond = cond.strip()
            if ">" in cond:
                left, right = cond.split(">", 1)
                left_val = self._resolve_value(left.strip())
                right_val = self._resolve_value(right.strip())
                if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
                    return self._resolve_value(true_val.strip()) if left_val > right_val else self._resolve_value(false_val.strip())
            if "<" in cond:
                left, right = cond.split("<", 1)
                left_val = self._resolve_value(left.strip())
                right_val = self._resolve_value(right.strip())
                if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
                    return self._resolve_value(true_val.strip()) if left_val < right_val else self._resolve_value(false_val.strip())

        if re.match(r"^\w\d+$", formula, re.IGNORECASE):
            return self.get(formula)

        try:
            return eval(formula, {"__builtins__": {}}, {})
        except Exception:
            return formula

    def _resolve_value(self, expr: str) -> Any:
        expr = expr.strip()
        if re.match(r"^\w\d+$", expr, re.IGNORECASE):
            return self.get(expr)
        try:
            return int(expr)
        except ValueError:
            try:
                return float(expr)
            except ValueError:
                return expr
