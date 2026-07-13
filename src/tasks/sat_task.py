import re

from datasets import Dataset

from src.task_env import TaskEnv, TaskRegistry
from src.envs.sat_env import SATEnv, _evaluate, _format_assignment


@TaskRegistry.register("sat_counterfactual")
class SATCounterfactualTask(TaskEnv):
    is_interactive = True

    def __init__(self, num_games: int = 500, num_vars: int = 4, num_clauses: int = 4,
                 max_probes: int = 8, max_turns: int = 10):
        self.num_games = num_games
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.max_probes = max_probes
        self.max_turns = max_turns

    def load_dataset(self) -> Dataset:
        return Dataset.from_list([])

    def get_prompt(self, example: dict) -> str:
        return example.get("prompt", "")

    def compute_reward(self, prompt: str, completion: str) -> float:
        return 0.0

    def get_initial_state(self, idx: int) -> dict:
        env = SATEnv(num_vars=self.num_vars, num_clauses=self.num_clauses,
                     seed=600 + idx)
        env.reset()
        return {
            "clauses_given": env.clauses_given,
            "clauses_actual": env.clauses_actual,
            "satisfying_actual": dict(env.satisfying_actual),
            "num_vars": self.num_vars,
            "seed": 600 + idx,
            "probes_used": 0,
            "turn": 0,
        }

    def get_initial_prompt(self, state: dict) -> str:
        formula_text = self._format_clauses(state["clauses_given"])
        return (
            "You are given a 3SAT formula. One component has been secretly "
            "removed from the actual formula.\n"
            "You can probe variables to learn their values in a satisfying "
            "assignment of the ACTUAL (modified) formula.\n"
            "Your goal: find an assignment that satisfies the actual formula "
            "but NOT the given formula.\n\n"
            f"Given formula:\n{formula_text}\n\n"
            "Commands:\n"
            "  PROBE xN  -- ask for the value of variable N in the actual formula\n"
            "  SOLVE x1=true, x2=false, ...  -- propose a counterfactual assignment\n\n"
            f"Probes remaining: {self.max_probes - state['probes_used']}"
        )

    def process_action(self, state: dict, action_text: str) -> dict:
        probe_match = re.search(r'PROBE\s+x(\d+)', action_text, re.IGNORECASE)
        if probe_match:
            var = int(probe_match.group(1))
            if var < 1 or var > state["num_vars"]:
                obs = f"Invalid variable x{var}. Variables are x1 to x{state['num_vars']}."
                return {"observation": obs, "reward": 0.0, "done": False, "state": state}

            value = state["satisfying_actual"].get(var)
            if value is None:
                obs = f"x{var} has no assigned value."
            else:
                obs = f"x{var} = {value}"

            new_probes = state["probes_used"] + 1
            new_state = {
                **state,
                "probes_used": new_probes,
                "turn": state["turn"] + 1,
            }

            if new_probes >= self.max_probes:
                obs += "\n\nNo probes remaining. You must SOLVE now."

            done = state["turn"] + 1 >= self.max_turns
            return {"observation": obs, "reward": 0.0, "done": done, "state": new_state}

        solve_match = re.search(r'SOLVE\s+(.+)', action_text, re.IGNORECASE)
        if solve_match:
            assignment_str = solve_match.group(1)
            assignment = self._parse_assignment(assignment_str, state["num_vars"])
            if assignment is None or len(assignment) != state["num_vars"]:
                obs = ("Invalid assignment. Use SOLVE x1=true, x2=false, ... "
                       f"for all {state['num_vars']} variables.")
                return {"observation": obs, "reward": 0.0, "done": False, "state": state}

            is_counterfactual = (
                _evaluate(state["clauses_actual"], assignment)
                and not _evaluate(state["clauses_given"], assignment)
            )

            new_state = {**state, "turn": state["turn"] + 1}

            if is_counterfactual:
                obs = (f"Correct! Assignment {_format_assignment(assignment)} "
                        "satisfies the actual formula but not the given formula.")
                actions_taken = state["turn"] + 1
                return {"observation": obs, "reward": 1.0 / max(1, actions_taken), "done": True, "state": new_state}
            else:
                obs = (f"Assignment {_format_assignment(assignment)} is NOT counterfactual. "
                       "It either satisfies both formulas or neither.")
                done = state["turn"] + 1 >= self.max_turns
                return {"observation": obs, "reward": 0.0, "done": done, "state": new_state}

        obs = "Unknown command. Use PROBE xN to query a variable, or SOLVE x1=... to answer."
        return {"observation": obs, "reward": 0.0, "done": False, "state": state}

    def compute_episode_reward(self, final_state: dict) -> float:
        return 0.0

    def _format_clauses(self, clauses: list[list[int]]) -> str:
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

    def _parse_assignment(self, text: str, num_vars: int) -> dict[int, bool] | None:
        result = {}
        pairs = re.findall(r'x(\d+)\s*[=:]\s*(true|false|1|0)', text, re.IGNORECASE)
        for var_str, val_str in pairs:
            var = int(var_str)
            val = val_str.lower() in ("true", "1")
            result[var] = val
        if len(result) == num_vars and set(result.keys()) == set(range(1, num_vars + 1)):
            return result
        pairs = re.findall(r'x(\d+)\s*[=:]\s*(T|F)', text, re.IGNORECASE)
        result = {}
        for var_str, val_str in pairs:
            var = int(var_str)
            result[var] = val_str.upper() == "T"
        if len(result) == num_vars and set(result.keys()) == set(range(1, num_vars + 1)):
            return result
        return None
