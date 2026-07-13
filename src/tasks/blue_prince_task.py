import random
import re
from itertools import product

from datasets import Dataset

from src.task_env import TaskEnv, TaskRegistry

_blue_prince_refs: dict[str, dict] = {}


def _generate_statement(num_boxes: int, gem: int, rng: random.Random) -> str:
    gem_types = ["at", "not_at", "in_set", "not_in_set", "if_then"]
    truth_types = ["box_true", "box_false", "same", "different",
                   "at_least_one_true", "at_least_one_false"]

    if rng.random() < 0.4:
        stmt_type = rng.choice(gem_types)
    else:
        stmt_type = rng.choice(truth_types)

    targets = list(range(num_boxes))

    if stmt_type == "at":
        box = rng.choice(targets)
        return f"The gem is in box {box}."
    elif stmt_type == "not_at":
        box = rng.choice(targets)
        return f"The gem is NOT in box {box}."
    elif stmt_type == "in_set":
        n = rng.randint(2, min(3, num_boxes))
        chosen = rng.sample(targets, n)
        return f"The gem is in one of boxes {', '.join(str(b) for b in sorted(chosen))}."
    elif stmt_type == "not_in_set":
        n = rng.randint(2, min(3, num_boxes))
        chosen = rng.sample(targets, n)
        return f"The gem is NOT in boxes {', '.join(str(b) for b in sorted(chosen))}."
    elif stmt_type == "if_then":
        a = rng.choice(targets)
        b = rng.choice([t for t in targets if t != a])
        return f"If the gem is in box {a}, then it is NOT in box {b}."
    elif stmt_type == "box_true":
        box = rng.choice(targets)
        return f"Box {box}'s statement is true."
    elif stmt_type == "box_false":
        box = rng.choice(targets)
        return f"Box {box}'s statement is false (Box {box} is lying)."
    elif stmt_type == "same":
        a = rng.choice(targets)
        b = rng.choice([t for t in targets if t != a])
        return f"Box {a} and box {b} have the same truth value."
    elif stmt_type == "different":
        a = rng.choice(targets)
        b = rng.choice([t for t in targets if t != a])
        return f"Box {a} and box {b} have different truth values."
    elif stmt_type == "at_least_one_true":
        n = rng.randint(2, min(3, num_boxes))
        chosen = rng.sample(targets, n)
        return f"At least one of boxes {', '.join(str(b) for b in sorted(chosen))} is telling the truth."
    elif stmt_type == "at_least_one_false":
        n = rng.randint(2, min(3, num_boxes))
        chosen = rng.sample(targets, n)
        return f"At least one of boxes {', '.join(str(b) for b in sorted(chosen))} is lying."

    return f"The gem is in box {gem}."


def _eval_statement(stmt: str, gem: int, truth: dict[int, bool]) -> bool:
    stmt_lower = stmt.lower()

    at_match = re.match(r'the gem is in box (\d+)', stmt_lower)
    if at_match:
        return gem == int(at_match.group(1))

    not_at_match = re.match(r'the gem is not in box (\d+)', stmt_lower)
    if not_at_match:
        return gem != int(not_at_match.group(1))

    in_set_match = re.match(r'the gem is in one of boxes (.+)', stmt_lower)
    if in_set_match:
        boxes = [int(b.strip()) for b in re.findall(r'\d+', in_set_match.group(1))]
        return gem in boxes

    not_in_set_match = re.match(
        r'the gem is (?:not|NOT) in boxes (.+)', stmt_lower)
    if not_in_set_match:
        boxes = [int(b.strip()) for b in re.findall(r'\d+', not_in_set_match.group(1))]
        return gem not in boxes

    if_then_match = re.match(
        r'if the gem is in box (\d+), then it is not in box (\d+)', stmt_lower)
    if if_then_match:
        a, b = int(if_then_match.group(1)), int(if_then_match.group(2))
        if gem == a:
            return gem != b
        return True

    box_true_match = re.match(r"box (\d+)'s statement is true", stmt_lower)
    if box_true_match:
        return truth.get(int(box_true_match.group(1)), False)

    box_false_match = re.match(
        r"box (\d+)'s statement is false|box (\d+) is lying", stmt_lower)
    if box_false_match:
        box = int(box_false_match.group(1) or box_false_match.group(2))
        return not truth.get(box, False)

    same_match = re.match(
        r'box (\d+) and box (\d+) have the same truth value', stmt_lower)
    if same_match:
        a, b = int(same_match.group(1)), int(same_match.group(2))
        return truth.get(a, False) == truth.get(b, False)

    diff_match = re.match(
        r'box (\d+) and box (\d+) have different truth values', stmt_lower)
    if diff_match:
        a, b = int(diff_match.group(1)), int(diff_match.group(2))
        return truth.get(a, False) != truth.get(b, False)

    at_least_true_match = re.match(
        r'at least one of boxes (.+) is telling the truth', stmt_lower)
    if at_least_true_match:
        boxes = [int(b.strip()) for b in re.findall(r'\d+', at_least_true_match.group(1))]
        return any(truth.get(b, False) for b in boxes)

    at_least_false_match = re.match(
        r'at least one of boxes (.+) is lying', stmt_lower)
    if at_least_false_match:
        boxes = [int(b.strip()) for b in re.findall(r'\d+', at_least_false_match.group(1))]
        return any(not truth.get(b, True) for b in boxes)

    return False


def _find_consistent_truths(statements: list[str], gem: int
                            ) -> list[dict[int, bool]]:
    num_boxes = len(statements)
    results = []
    for bits in range(1 << num_boxes):
        truth = {}
        for i in range(num_boxes):
            truth[i] = bool(bits & (1 << i))
        consistent = True
        for i, stmt in enumerate(statements):
            if truth[i] != _eval_statement(stmt, gem, truth):
                consistent = False
                break
        if consistent:
            results.append(truth)
    return results


def _check_constraint(truth: dict[int, bool], constraint_type: str,
                      k: int | None = None) -> bool:
    n = len(truth)
    true_count = sum(1 for v in truth.values() if v)
    if constraint_type == "exactly_k":
        return true_count == k
    elif constraint_type == "at_least_k":
        return true_count >= (k or 1)
    elif constraint_type == "at_most_k":
        return true_count <= (k or n - 1)
    elif constraint_type == "mixed":
        return 0 < true_count < n
    return True


def _format_constraint(constraint_type: str, k: int | None = None) -> str:
    if constraint_type == "exactly_k":
        return f"Exactly {k} of the statements are true; the rest are lies."
    elif constraint_type == "at_least_k":
        return f"At least {k} of the statements are true."
    elif constraint_type == "at_most_k":
        return f"At most {k} of the statements are true."
    elif constraint_type == "mixed":
        return ("Not all statements have the same truth value "
                "(at least one is true and at least one is false).")
    return ""


def _generate_puzzle(num_boxes: int, constraint_type: str,
                     k: int | None, seed: int
                     ) -> tuple[list[str], int] | None:
    rng = random.Random(seed)

    for _ in range(2000):
        gem = rng.randrange(num_boxes)
        statements = [_generate_statement(num_boxes, gem, rng) for _ in range(num_boxes)]

        valid_gems = []
        for g in range(num_boxes):
            truths = _find_consistent_truths(statements, g)
            for t in truths:
                if _check_constraint(t, constraint_type, k):
                    valid_gems.append(g)
                    break

        if len(set(valid_gems)) == 1 and valid_gems[0] == gem:
            return statements, gem

    return None


@TaskRegistry.register("blue_prince")
class BluePrinceTask(TaskEnv):
    PROMPT_TEMPLATE = (
        "There are {num_boxes} boxes, each with a statement. "
        "{constraint_text} "
        "A gem is hidden in exactly one box.\n"
        "Determine which box contains the gem.\n\n"
        "Statements:\n{statements}\n\n"
        "Reason step by step, then output which box contains the gem:\n"
        "Answer: box N"
    )

    def __init__(self, num_puzzles: int = 500, max_boxes: int = 5):
        self.num_puzzles = num_puzzles
        self.max_boxes = max_boxes

    def load_dataset(self) -> Dataset:
        data = []
        constraint_types = ["mixed", "exactly_k", "at_least_k", "at_most_k"]
        for i in range(self.num_puzzles):
            rng = random.Random(800 + i)
            num_boxes = rng.randint(2, self.max_boxes)
            ctype = rng.choice(constraint_types)
            k = None
            if ctype == "exactly_k":
                k = rng.randint(1, max(1, num_boxes - 1))
            elif ctype in ("at_least_k", "at_most_k"):
                k = rng.randint(1, max(1, num_boxes))

            puzzle = _generate_puzzle(num_boxes, ctype, k, 1000 + i)
            if puzzle is None:
                puzzle = _generate_simple_puzzle(num_boxes, "mixed")

            statements, gem = puzzle
            statements_text = "\n".join(
                f"Box {j}: {s}" for j, s in enumerate(statements)
            )
            constraint_text = _format_constraint(ctype, k)
            prompt = self.get_prompt({
                "num_boxes": num_boxes,
                "constraint_text": constraint_text,
                "statements": statements_text,
            })
            _blue_prince_refs[prompt] = {"gem": gem, "num_boxes": num_boxes}
            data.append({"prompt": prompt, "task_name": "blue_prince"})
        return Dataset.from_list(data)

    def get_prompt(self, example: dict) -> str:
        return self.PROMPT_TEMPLATE.format(
            num_boxes=example["num_boxes"],
            constraint_text=example["constraint_text"],
            statements=example["statements"],
        )

    def compute_reward(self, prompt: str, completion: str) -> float:
        ref = _blue_prince_refs.get(prompt)
        if ref is None:
            return 0.0

        answer = self._extract_answer(completion, ref["num_boxes"])
        if answer is None:
            return 0.0
        return 1.0 if answer == ref["gem"] else 0.0

    def _extract_answer(self, text: str, num_boxes: int) -> int | None:
        patterns = [
            r'Answer:\s*box\s*(\d+)',
            r'answer:\s*box\s*(\d+)',
            r'box\s*(\d+)\s*(?:contains|has|holds)\s*(?:the\s*)?gem',
            r'the gem is in box\s*(\d+)',
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                val = int(match.group(1))
                if 0 <= val < num_boxes:
                    return val

        numbers = re.findall(r'\b(\d+)\b', text)
        for n_str in reversed(numbers):
            val = int(n_str)
            if 0 <= val < num_boxes:
                return val
        return None


def _generate_simple_puzzle(num_boxes: int,
                            constraint_type: str = "mixed"
                            ) -> tuple[list[str], int]:
    gem = 0
    statements = []
    for i in range(num_boxes):
        if i == gem:
            statements.append(f"The gem is in box {gem}.")
        else:
            statements.append(f"The gem is NOT in box {i}.")
    return statements, gem
