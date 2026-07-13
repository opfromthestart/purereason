import copy
import json
import random
import re

from datasets import Dataset

from src.task_env import TaskEnv, TaskRegistry
from src.envs.cryptid_env import (
    Board, Clue, Terrain, Territory, StructureType, StructureColor,
    _generate_all_clues, _count_joint_matches,
)


@TaskRegistry.register("cryptid")
class CryptidTask(TaskEnv):
    is_interactive = True

    def __init__(self, num_games: int = 500, width: int = 5, height: int = 5,
                 num_players: int = 3, max_turns: int = 10):
        self.num_games = num_games
        self.width = width
        self.height = height
        self.num_players = num_players
        self.max_turns = max_turns

    def load_dataset(self) -> Dataset:
        return Dataset.from_list([])

    def get_prompt(self, example: dict) -> str:
        return example.get("prompt", "")

    def compute_reward(self, prompt: str, completion: str) -> float:
        return 0.0

    def get_initial_state(self, idx: int) -> dict:
        seed = 500 + idx
        board = Board(width=self.width, height=self.height, seed=seed)

        possible_habitats = []
        for x, y in board.all_spaces():
            possible_habitats.append((x, y))
        if not possible_habitats:
            possible_habitats = [(0, 0)]
        habitat = random.Random(seed + 1000).choice(possible_habitats)

        clues = _generate_all_clues(board, habitat, self.num_players)

        my_idx = 0
        return {
            "board": board,
            "clues": clues,
            "my_idx": my_idx,
            "habitat": habitat,
            "cubes": set(),
            "discs": set(),
            "turn": 0,
            "game_over": False,
            "seed": seed,
        }

    def get_initial_prompt(self, state: dict) -> str:
        board = state["board"]
        my_idx = state["my_idx"]
        clue = state["clues"][my_idx]
        return self._render_game_state(state, "Game start")

    def process_action(self, state: dict, action_text: str) -> dict:
        action = self._parse_action(action_text)
        if action is None:
            return {
                "observation": "Invalid action. Use: QUESTION (x,y) player N  or  SEARCH (x,y)",
                "reward": 0.0,
                "done": False,
                "state": state,
            }

        board = state["board"]
        my_idx = state["my_idx"]
        my_clue = state["clues"][my_idx]
        cubes = set(state["cubes"])
        discs = set(state["discs"])
        turn = state["turn"]

        if action["type"] == "question":
            qx, qy = action["x"], action["y"]
            target_player = action["player"]

            if target_player < 0 or target_player >= self.num_players or target_player == my_idx:
                return {
                    "observation": f"Invalid player: {target_player}",
                    "reward": 0.0, "done": False,
                    "state": state,
                }

            if not (0 <= qx < self.width and 0 <= qy < self.height):
                return {
                    "observation": f"Space ({qx},{qy}) out of bounds.",
                    "reward": 0.0, "done": False,
                    "state": state,
                }

            target_clue = state["clues"][target_player]
            matches = target_clue.matches(qx, qy, board)
            if matches:
                discs.add((target_player, qx, qy))
            else:
                cubes.add((target_player, qx, qy))

            my_cube_placed = None
            for cx, cy in board.all_spaces():
                if (my_idx, cx, cy) in cubes:
                    continue
                if any((pi, cx, cy) in cubes for pi in range(self.num_players)):
                    continue
                if not my_clue.matches(cx, cy, board):
                    cubes.add((my_idx, cx, cy))
                    my_cube_placed = (cx, cy)
                    break

            new_state = {
                **state,
                "cubes": cubes,
                "discs": discs,
                "turn": turn + 1,
            }

            result_word = "YES (disc)" if matches else "NO (cube)"
            obs = (f"Asked Player {target_player} about ({qx},{qy}): {result_word}\n"
                   f"Your cube placed at ({my_cube_placed[0]},{my_cube_placed[1]})\n\n")
            obs += self._render_game_state(new_state, f"Turn {turn + 1}")

            done = turn + 1 >= self.max_turns

            return {
                "observation": obs,
                "reward": 0.0,
                "done": done,
                "state": new_state,
            }

        elif action["type"] == "search":
            sx, sy = action["x"], action["y"]

            if not (0 <= sx < self.width and 0 <= sy < self.height):
                return {
                    "observation": f"Space ({sx},{sy}) out of bounds.",
                    "reward": 0.0, "done": False,
                    "state": state,
                }

            if not my_clue.matches(sx, sy, board):
                return {
                    "observation": f"Space ({sx},{sy}) does not match your own clue.",
                    "reward": 0.0, "done": False,
                    "state": state,
                }

            discs.add((my_idx, sx, sy))

            all_match = True
            for pi in range(self.num_players):
                if pi == my_idx:
                    continue
                clue = state["clues"][pi]
                if clue.matches(sx, sy, board):
                    discs.add((pi, sx, sy))
                else:
                    cubes.add((pi, sx, sy))
                    all_match = False
                    break

            new_state = {
                **state,
                "cubes": cubes,
                "discs": discs,
                "turn": turn + 1,
            }

            if all_match and (sx, sy) == state["habitat"]:
                new_state["game_over"] = True
                obs = f"SEARCH ({sx},{sy}) - ALL YES! You found the habitat!\n\n"
                obs += self._render_game_state(new_state, "GAME WON")
                return {
                    "observation": obs,
                    "reward": 1.0,
                    "done": True,
                    "state": new_state,
                }
            else:
                my_cube_placed = None
                for cx, cy in board.all_spaces():
                    if (my_idx, cx, cy) in cubes:
                        continue
                    if any((pi, cx, cy) in cubes for pi in range(self.num_players)):
                        continue
                    if not my_clue.matches(cx, cy, board):
                        cubes.add((my_idx, cx, cy))
                        my_cube_placed = (cx, cy)
                        break
                new_state["cubes"] = cubes
                new_state["discs"] = discs
                done = turn + 1 >= self.max_turns
                obs = f"SEARCH ({sx},{sy}) - Someone said NO. Not the habitat.\n"
                if my_cube_placed:
                    obs += f"Your cube placed at ({my_cube_placed[0]},{my_cube_placed[1]})\n"
                obs += "\n" + self._render_game_state(new_state, f"Turn {turn + 1}")
                return {
                    "observation": obs,
                    "reward": 0.0,
                    "done": done,
                    "state": new_state,
                }

        return {
            "observation": "Unknown action type.",
            "reward": 0.0,
            "done": False,
            "state": state,
        }

    def compute_episode_reward(self, final_state: dict) -> float:
        return 1.0 if final_state.get("game_over", False) else 0.0

    def _render_game_state(self, state: dict, header: str) -> str:
        board = state["board"]
        my_idx = state["my_idx"]
        cubes = state["cubes"]
        discs = state["discs"]
        my_clue = state["clues"][my_idx]

        lines = [f"=== {header} ==="]
        lines.append(f"Turn: {state['turn']}/{self.max_turns}")
        lines.append(f"Your clue: {my_clue.describe()}")
        lines.append("")

        lines.append("Board:")
        col_header = "    " + "  ".join(str(x) for x in range(board.width))
        lines.append(col_header)
        for y in range(board.height):
            row_parts = [f" {y} "]
            for x in range(board.width):
                cell = board.get_cell(x, y)
                terrain = cell.terrain.value

                markers = ""
                if cell.bear_territory:
                    markers += "b"
                if cell.cougar_territory:
                    markers += "c"
                if cell.structure_type:
                    st = "S" if cell.structure_type == StructureType.STANDING_STONE else "A"
                    sc = cell.structure_color.value[0] if cell.structure_color else "?"
                    markers += st.lower() + sc

                pieces = ""
                for pi in range(self.num_players):
                    if (pi, x, y) in cubes:
                        pieces += f"X{pi}"
                    if (pi, x, y) in discs:
                        pieces += f"O{pi}"

                cell_str = terrain
                if markers:
                    cell_str += f":{markers}"
                if pieces:
                    cell_str += f"|{pieces}"

                row_parts.append(f"{cell_str:<10}")
            lines.append("".join(row_parts))

        lines.append("")
        lines.append("Legend: D=desert F=forest W=water M=mountain S=swamp")
        lines.append("  b=bear c=cougar s=stone a=shack (w/g/b/k=color)")
        lines.append("  XN=cube from player N, ON=disc from player N")
        lines.append("")
        lines.append("Actions: QUESTION (x,y) player N   or   SEARCH (x,y)")

        return "\n".join(lines)

    def _parse_action(self, text: str) -> dict | None:
        text = text.strip()

        search_match = re.search(r'SEARCH\s*[\(\[]?\s*(\d+)\s*[,;]\s*(\d+)\s*[\)\]]?', text, re.IGNORECASE)
        if search_match:
            return {
                "type": "search",
                "x": int(search_match.group(1)),
                "y": int(search_match.group(2)),
            }

        question_match = re.search(
            r'QUESTION\s*[\(\[]?\s*(\d+)\s*[,;]\s*(\d+)\s*[\)\]]?\s*player\s*(\d+)',
            text, re.IGNORECASE,
        )
        if question_match:
            return {
                "type": "question",
                "x": int(question_match.group(1)),
                "y": int(question_match.group(2)),
                "player": int(question_match.group(3)),
            }

        json_actions = self._parse_json(text)
        if json_actions:
            return json_actions[0]

        return None

    def _parse_json(self, text: str) -> list[dict]:
        decoder = json.JSONDecoder()
        actions = []
        pos = 0
        while pos < len(text):
            while pos < len(text) and text[pos] in (" ", "\n", "\r", "\t", ","):
                pos += 1
            if pos >= len(text):
                break
            if text[pos] == "{":
                try:
                    obj, end = decoder.raw_decode(text, pos)
                    if isinstance(obj, dict) and "action" in obj:
                        actions.append(obj)
                    pos = end
                except json.JSONDecodeError:
                    pos += 1
            else:
                pos += 1
        return actions
