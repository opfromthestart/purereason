import random
import re

from datasets import Dataset

from src.task_env import TaskEnv, TaskRegistry

_knights_knaves_refs: dict[str, dict[str, bool]] = {}


def _eval_truth(statement: dict, types: dict[str, bool]) -> bool:
    t = statement["type"]
    if t == "is_knight":
        return types[statement["x"]]
    elif t == "is_knave":
        return not types[statement["x"]]
    elif t == "i_am_knight":
        return types[statement["speaker"]]
    elif t == "same":
        return types[statement["x"]] == types[statement["y"]]
    elif t == "different":
        return types[statement["x"]] != types[statement["y"]]
    elif t == "if_knight_then_knave":
        return (not types[statement["x"]]) or (not types[statement["y"]])
    return True


def _check_unique_solution(
    statements: list[dict], chars: list[str]
) -> tuple[bool, dict[str, bool] | None]:
    solutions = []
    n = len(chars)
    for mask in range(1 << n):
        types = {}
        for i, c in enumerate(chars):
            types[c] = bool(mask & (1 << i))
        valid = True
        for stmt in statements:
            speaker = stmt["speaker"]
            truth_val = _eval_truth(stmt, types)
            if types[speaker] != truth_val:
                valid = False
                break
        if valid:
            solutions.append(types)
    if len(solutions) == 1:
        return True, solutions[0]
    return False, None


def _format_statement(stmt: dict) -> str:
    speaker = stmt["speaker"]
    t = stmt["type"]
    if t == "is_knight":
        return f"{speaker} says: {stmt['x']} is a knight."
    elif t == "is_knave":
        return f"{speaker} says: {stmt['x']} is a knave."
    elif t == "i_am_knight":
        return f"{speaker} says: I am a knight."
    elif t == "same":
        return f"{speaker} says: {stmt['x']} and {stmt['y']} are the same type."
    elif t == "different":
        return f"{speaker} says: {stmt['x']} and {stmt['y']} are different types."
    elif t == "if_knight_then_knave":
        return f"{speaker} says: If {stmt['x']} is a knight, then {stmt['y']} is a knave."
    return ""


def _generate_statement_consistent(
    speaker: str, chars: list[str], types: dict[str, bool]
) -> dict:
    speaker_is_knight = types[speaker]
    others = [c for c in chars if c != speaker]

    stmt_types = ["is_knight", "is_knave", "i_am_knight", "same", "different"]
    if len(chars) >= 2:
        stmt_types.append("if_knight_then_knave")

    for _ in range(50):
        stmt_type = random.choice(stmt_types)
        stmt: dict = {"speaker": speaker, "type": stmt_type}

        if stmt_type == "i_am_knight":
            return stmt

        elif stmt_type in ("is_knight", "is_knave"):
            candidates = list(chars)
            random.shuffle(candidates)
            for target in candidates:
                stmt["x"] = target
                truth = _eval_truth(stmt, types)
                if truth == speaker_is_knight:
                    return stmt
            stmt["x"] = random.choice(chars)
            truth = _eval_truth(stmt, types)
            if truth != speaker_is_knight:
                stmt["type"] = "is_knave" if stmt_type == "is_knight" else "is_knight"
            return stmt

        elif stmt_type in ("same", "different"):
            x = random.choice(chars)
            y_candidates = [c for c in chars if c != x]
            if not y_candidates:
                continue
            y = random.choice(y_candidates)
            stmt["x"] = x
            stmt["y"] = y
            truth = _eval_truth(stmt, types)
            if truth == speaker_is_knight:
                return stmt
            stmt["type"] = "different" if stmt_type == "same" else "same"
            return stmt

        elif stmt_type == "if_knight_then_knave":
            x = random.choice(chars)
            y = random.choice(chars)
            stmt["x"] = x
            stmt["y"] = y
            truth = _eval_truth(stmt, types)
            if truth == speaker_is_knight:
                return stmt

    return {"speaker": speaker, "type": "i_am_knight"}


def _generate_puzzle(num_characters: int) -> tuple[list[str], dict[str, bool], list[dict]]:
    chars = [chr(ord("A") + i) for i in range(num_characters)]

    for _ in range(2000):
        types = {c: random.choice([True, False]) for c in chars}
        if all(types[c] for c in chars) or not any(types[c] for c in chars):
            continue

        statements = []
        for speaker in chars:
            num_stmts = random.randint(1, 2)
            for _ in range(num_stmts):
                stmt = _generate_statement_consistent(speaker, chars, types)
                statements.append(stmt)

        unique, solution = _check_unique_solution(statements, chars)
        if unique and solution == types:
            return chars, types, statements

    return _generate_simple_puzzle(num_characters)


def _generate_simple_puzzle(num_characters: int) -> tuple[list[str], dict[str, bool], list[dict]]:
    chars = [chr(ord("A") + i) for i in range(num_characters)]
    types = {}
    statements = []
    for i, c in enumerate(chars):
        types[c] = i % 2 == 0
    statements.append({"speaker": "A", "type": "i_am_knight"})
    for i in range(num_characters - 1):
        stmt_type = "is_knave" if types[chars[i + 1]] else "is_knight"
        statements.append({
            "speaker": chars[i],
            "type": stmt_type,
            "x": chars[i + 1],
        })
    return chars, types, statements


@TaskRegistry.register("knights_knaves")
class KnightsKnavesTask(TaskEnv):
    PROMPT_TEMPLATE = (
        "You are on an island where each person is either a knight (always "
        "tells the truth) or a knave (always lies). Given their statements, "
        "determine who is a knight and who is a knave.\n\n"
        "Reason step by step, then output your final answer on a new line as:\n"
        "Answer: A: knight, B: knave, ...\n\n"
        "{statements}"
    )

    def __init__(self, num_puzzles: int = 500, num_characters: int = 3):
        self.num_puzzles = num_puzzles
        self.num_characters = num_characters

    def load_dataset(self) -> Dataset:
        data = []
        for i in range(self.num_puzzles):
            n = random.randint(2, max(2, self.num_characters))
            chars, types, statements = _generate_puzzle(n)
            statements_text = "\n".join(_format_statement(s) for s in statements)
            prompt = self.get_prompt({"statements": statements_text})
            _knights_knaves_refs[prompt] = types
            data.append({"prompt": prompt, "task_name": "knights_knaves"})
        return Dataset.from_list(data)

    def get_prompt(self, example: dict) -> str:
        return self.PROMPT_TEMPLATE.format(statements=example["statements"])

    def compute_reward(self, prompt: str, completion: str) -> float:
        ref = _knights_knaves_refs.get(prompt)
        if ref is None:
            return 0.0
        pred = self._extract_classifications(completion, list(ref.keys()))
        if pred is None:
            return 0.0
        if pred == ref:
            return 1.0
        return 0.0

    def _extract_classifications(
        self, text: str, chars: list[str]
    ) -> dict[str, bool] | None:
        patterns = [
            r"Answer:\s*(.*?)$",
            r"answer:\s*(.*?)$",
        ]
        answer_text = None
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                answer_text = matches[-1].strip()
                break
        if answer_text is None:
            lines = text.strip().split("\n")
            for line in reversed(lines):
                if any(c in line.upper() for c in chars):
                    answer_text = line.strip()
                    break
        if answer_text is None:
            return None
        result: dict[str, bool] = {}
        for c in chars:
            knight_pat = re.search(
                rf"{c}\s*[:=]?\s*(knight|true|k)", answer_text, re.IGNORECASE
            )
            knave_pat = re.search(
                rf"{c}\s*[:=]?\s*(knave|false|f)", answer_text, re.IGNORECASE
            )
            if knight_pat and not knave_pat:
                result[c] = True
            elif knave_pat and not knight_pat:
                result[c] = False
            elif knight_pat and knave_pat:
                if knight_pat.start() < knave_pat.start():
                    result[c] = True
                else:
                    result[c] = False
            else:
                return None
        if len(result) != len(chars):
            return None
        return result
