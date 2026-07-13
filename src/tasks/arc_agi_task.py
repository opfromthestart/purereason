from datasets import Dataset

from src.task_env import TaskEnv, TaskRegistry


@TaskRegistry.register("arc_agi")
class ArcAGITask(TaskEnv):
    is_interactive = True

    def __init__(self, game_ids: list[str] | None = None, num_episodes: int = 500,
                 max_turns: int = 100):
        self._game_ids = game_ids
        self.num_episodes = num_episodes
        self.max_turns = max_turns
        self._env_cache: dict = {}

    def _get_game_ids(self):
        if self._game_ids is not None:
            return self._game_ids
        from src.envs.arc_agi_env import create_arcade, get_available_game_ids
        arc = create_arcade()
        ids = get_available_game_ids(arc)
        if not ids:
            ids = ["ls20", "ft09", "vc33"]
        return ids

    def load_dataset(self) -> Dataset:
        return Dataset.from_list([])

    def get_prompt(self, example: dict) -> str:
        return example.get("prompt", "")

    def compute_reward(self, prompt: str, completion: str) -> float:
        return 0.0

    def _get_env(self, state: dict):
        key = state["episode_key"]
        if key not in self._env_cache:
            from src.envs.arc_agi_env import create_arcade
            arc = create_arcade()
            short_id = state["game_id"].split("-")[0]
            env = arc.make(short_id)
            if env is None:
                raise RuntimeError(f"Game {short_id} not found")
            env.reset()
            self._env_cache[key] = (arc, env)
        return self._env_cache[key]

    def get_initial_state(self, idx: int) -> dict:
        game_ids = self._get_game_ids()
        game_id = game_ids[idx % len(game_ids)]
        episode_key = f"{game_id}_{idx}"

        from src.envs.arc_agi_env import create_arcade
        arc = create_arcade()
        short_id = game_id.split("-")[0]
        env = arc.make(short_id)
        obs = env.reset()
        self._env_cache[episode_key] = (arc, env)

        cells = obs.frame[0].copy() if obs and obs.frame else None

        return {
            "game_id": game_id,
            "episode_key": episode_key,
            "cells": cells,
            "available_actions": list(obs.available_actions) if obs else [],
            "game_state": obs.state.value if obs else "unknown",
            "levels_completed": obs.levels_completed if obs else 0,
            "last_levels": obs.levels_completed if obs else 0,
            "turn": 0,
            "done": False,
        }

    def get_initial_prompt(self, state: dict) -> str:
        cells = state["cells"]
        from src.envs.arc_agi_env import ArcAGIGrid

        grid = ArcAGIGrid(cells=cells, available_actions=state["available_actions"])
        lines = [grid.render(), ""]
        lines.append(self._status_lines(state))
        return "\n".join(lines)

    def process_action(self, state: dict, action_text: str) -> dict:
        from arcengine import GameAction, GameState
        from src.envs.arc_agi_env import parse_action, ArcAGIGrid

        s = dict(state)

        current_state = GameState(s["game_state"]) if s["game_state"] != "unknown" else None

        if current_state in (GameState.WIN, GameState.GAME_OVER):
            done = True
            reward = (1.0 / max(1, s["turn"] + 1)) if current_state == GameState.WIN else 0.0
            new_state = {**s, "turn": s["turn"] + 1, "done": True}
            cells = s["cells"]
            grid = ArcAGIGrid(cells=cells, available_actions=s["available_actions"]) if cells is not None else None
            obs_lines = grid.render() + "\n\n" if grid else ""
            observation = obs_lines + self._status_lines(new_state)
            return {
                "observation": observation,
                "reward": reward,
                "done": True,
                "state": new_state,
            }

        arc, env = self._get_env(s)
        available = s["available_actions"]

        action_id, action_data = parse_action(action_text, available)

        game_action_map = {
            1: GameAction.ACTION1, 2: GameAction.ACTION2,
            3: GameAction.ACTION3, 4: GameAction.ACTION4,
            5: GameAction.ACTION5, 6: GameAction.ACTION6,
            7: GameAction.ACTION7,
        }
        action = game_action_map.get(action_id)
        if action is not None:
            obs = env.step(action, data=action_data if action_data else None)
        else:
            obs = env.observation_space

        gs = obs.state if obs else GameState.NOT_FINISHED
        done = gs in (GameState.WIN, GameState.GAME_OVER)
        new_levels = obs.levels_completed if obs else 0
        levels_gained = max(0, new_levels - s["last_levels"])

        if gs == GameState.WIN:
            reward = 1.0 / max(1, s["turn"] + 1)
        elif levels_gained > 0:
            reward = 0.1 * levels_gained
        else:
            reward = 0.0

        if s["turn"] + 1 >= self.max_turns and not done:
            done = True

        cells = obs.frame[0].copy() if obs and obs.frame else s["cells"]

        new_state = {
            **s,
            "cells": cells,
            "available_actions": list(obs.available_actions) if obs else s["available_actions"],
            "game_state": gs.value,
            "levels_completed": new_levels,
            "last_levels": new_levels if levels_gained > 0 else s["last_levels"],
            "turn": s["turn"] + 1,
            "done": done,
        }

        grid = ArcAGIGrid(cells=cells, available_actions=list(obs.available_actions) if obs else [])
        observation = grid.render() + "\n\n" + self._status_lines(new_state)

        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "state": new_state,
        }

    def compute_episode_reward(self, final_state: dict) -> float:
        from arcengine import GameState
        gs = GameState(final_state.get("game_state", "NOT_FINISHED"))
        turn = final_state.get("turn", 1)
        if gs == GameState.WIN:
            return 1.0 / max(1, turn)
        levels = final_state.get("levels_completed", 0)
        if levels > 0:
            return 0.1 * levels
        return 0.0

    def _status_lines(self, state: dict) -> str:
        from src.envs.arc_agi_env import _CELL_CHARS
        from arcengine import GameAction

        avail = state.get("available_actions", [])
        action_names = [GameAction.from_id(a).name for a in avail]

        return (
            f"State: {state['game_state']}\n"
            f"Levels: {state['levels_completed']}\n"
            f"Available: {', '.join(action_names)}\n"
            f"Legend: {_CELL_CHARS}\n"
            f"Output: ACTION<N> (or ACTION6 X Y for click)"
        )
