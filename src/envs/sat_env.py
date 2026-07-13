import itertools
import random
import re


def _generate_3sat(num_vars: int, num_clauses: int, seed: int) -> list[list[int]]:
    random.seed(seed)
    clauses = []
    for _ in range(num_clauses):
        vars_in_clause = random.sample(range(1, num_vars + 1), min(3, num_vars))
        clause = []
        for v in vars_in_clause:
            clause.append(v if random.random() < 0.5 else -v)
        clauses.append(clause)
    return clauses


def _evaluate(clauses: list[list[int]], assignment: dict[int, bool]) -> bool:
    for clause in clauses:
        satisfied = False
        for lit in clause:
            var = abs(lit)
            val = assignment.get(var, False)
            if (lit > 0 and val) or (lit < 0 and not val):
                satisfied = True
                break
        if not satisfied:
            return False
    return True


def _find_satisfying_assignment(clauses: list[list[int]], num_vars: int,
                                prefer: dict[int, bool] | None = None
                                ) -> dict[int, bool] | None:
    best = None
    for bits in range(1 << num_vars):
        assignment = {}
        for v in range(1, num_vars + 1):
            assignment[v] = bool(bits & (1 << (v - 1)))
        if _evaluate(clauses, assignment):
            if prefer is not None:
                if best is None:
                    best = assignment
                else:
                    old_score = sum(1 for v, val in prefer.items()
                                    if best.get(v) == val)
                    new_score = sum(1 for v, val in prefer.items()
                                    if assignment.get(v) == val)
                    if new_score > old_score:
                        best = assignment
            else:
                return assignment
    return best


def _format_clauses(clauses: list[list[int]]) -> str:
    parts = []
    for i, clause in enumerate(clauses):
        lits = []
        for lit in clause:
            var = f"x{abs(lit)}"
            if lit < 0:
                lits.append(f"NOT {var}")
            else:
                lits.append(var)
        parts.append(f"  ({' OR '.join(lits)})")
    return " AND\n".join(parts)


def _format_assignment(assignment: dict[int, bool]) -> str:
    parts = []
    for v in sorted(assignment):
        parts.append(f"x{v}={assignment[v]}")
    return ", ".join(parts)


class SATEnv:
    def __init__(self, num_vars: int = 4, num_clauses: int = 4, seed: int = 42):
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.seed = seed
        self.clauses_given: list[list[int]] = []
        self.clauses_actual: list[list[int]] = []
        self.removed_clause: list[int] = []
        self.removed_idx: int = -1
        self.satisfying_actual: dict[int, bool] = {}
        self.satisfying_given: dict[int, bool] | None = {}

    def reset(self):
        for attempt in range(500):
            rng = random.Random(self.seed + attempt)
            clauses = _generate_3sat(self.num_vars, self.num_clauses,
                                     self.seed + attempt)

            all_satisfying = []
            for bits in range(1 << self.num_vars):
                assignment = {}
                for v in range(1, self.num_vars + 1):
                    assignment[v] = bool(bits & (1 << (v - 1)))
                if _evaluate(clauses, assignment):
                    all_satisfying.append(assignment)

            if len(all_satisfying) < 2:
                continue

            candidate_indices = list(range(len(clauses)))
            rng.shuffle(candidate_indices)
            for rem_idx in candidate_indices:
                actual = [c for i, c in enumerate(clauses) if i != rem_idx]
                counterfactual = None
                for assignment in all_satisfying:
                    if not _evaluate(clauses, assignment) and _evaluate(actual, assignment):
                        pass
                    if _evaluate(actual, assignment) and not _evaluate(clauses, assignment):
                        counterfactual = assignment
                        break
                if counterfactual is not None:
                    given_sat = all_satisfying[0]
                    self.clauses_given = clauses
                    self.clauses_actual = actual
                    self.removed_clause = clauses[rem_idx]
                    self.removed_idx = rem_idx
                    self.satisfying_actual = given_sat
                    self.satisfying_given = given_sat
                    return

        clauses = _generate_3sat(self.num_vars, self.num_clauses, self.seed)
        self.clauses_given = clauses
        self.clauses_actual = clauses[:-1]
        self.removed_clause = clauses[-1]
        self.removed_idx = len(clauses) - 1
        sat = _find_satisfying_assignment(clauses, self.num_vars)
        self.satisfying_actual = sat or {v: False for v in range(1, self.num_vars + 1)}
        self.satisfying_given = _find_satisfying_assignment(clauses, self.num_vars)

    def render_given_formula(self) -> str:
        return _format_clauses(self.clauses_given)

    def get_variable_value(self, var: int) -> bool | None:
        return self.satisfying_actual.get(var)

    def check_counterfactual(self, assignment: dict[int, bool]) -> bool:
        sat_given = _evaluate(self.clauses_given, assignment)
        sat_actual = _evaluate(self.clauses_actual, assignment)
        return sat_actual and not sat_given
