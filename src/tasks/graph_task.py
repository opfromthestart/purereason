import random
import re
from collections import deque

from datasets import Dataset

from src.task_env import TaskEnv, TaskRegistry

_graph_refs: dict[str, dict] = {}


def _generate_graph(num_nodes: int, edge_prob: float, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    nodes = [chr(ord("A") + i) for i in range(num_nodes)]
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if rng.random() < edge_prob:
                edges.append((nodes[i], nodes[j]))
    return edges


def _build_adjacency(edges: list[tuple[str, str]]) -> dict[str, set[str]]:
    adj = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def _find_shortest_path(edges: list[tuple[str, str]], start: str, end: str) -> list[str] | None:
    adj = _build_adjacency(edges)
    if start not in adj or end not in adj:
        return None
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        for neighbor in adj.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None


def _format_edges(edges: list[tuple[str, str]]) -> str:
    return ", ".join(f"{u}-{v}" for u, v in sorted(edges))


@TaskRegistry.register("graph_walk")
class GraphWalkTask(TaskEnv):
    PROMPT_TEMPLATE = (
        "You are given an undirected graph as a list of edges.\n"
        "Find a path from the start node to the end node.\n"
        "If no path exists, answer 'NOT CONNECTED'.\n"
        "Output your answer as a sequence of nodes separated by arrows.\n\n"
        "Edges: {edges}\n"
        "Start: {start}\n"
        "End: {end}\n\n"
        "Path:"
    )

    def __init__(self, num_puzzles: int = 500, num_nodes: int = 7,
                 edge_prob: float = 0.35):
        self.num_puzzles = num_puzzles
        self.num_nodes = num_nodes
        self.edge_prob = edge_prob

    def load_dataset(self) -> Dataset:
        data = []
        for i in range(self.num_puzzles):
            seed = 700 + i
            edges = _generate_graph(self.num_nodes, self.edge_prob, seed)
            nodes = [chr(ord("A") + j) for j in range(self.num_nodes)]
            rng = random.Random(seed + 1000)
            start = rng.choice(nodes)
            end = rng.choice([n for n in nodes if n != start])
            if rng.random() < 0.3:
                while True:
                    end = rng.choice(nodes)
                    if end != start:
                        break

            path = _find_shortest_path(edges, start, end)

            prompt = self.get_prompt({
                "edges": _format_edges(edges),
                "start": start,
                "end": end,
            })
            _graph_refs[prompt] = {
                "edges": edges,
                "start": start,
                "end": end,
                "path": path,
            }
            data.append({"prompt": prompt, "task_name": "graph_walk"})
        return Dataset.from_list(data)

    def get_prompt(self, example: dict) -> str:
        return self.PROMPT_TEMPLATE.format(
            edges=example["edges"],
            start=example["start"],
            end=example["end"],
        )

    def compute_reward(self, prompt: str, completion: str) -> float:
        ref = _graph_refs.get(prompt)
        if ref is None:
            return 0.0

        pred = self._extract_path(completion)

        if pred is None:
            not_connected = re.search(r'NOT\s+CONNECTED', completion, re.IGNORECASE)
            if not_connected and ref["path"] is None:
                return 1.0
            return 0.0

        if ref["path"] is None:
            return 0.0

        if len(pred) < 2:
            return 0.0

        if pred[0] != ref["start"] or pred[-1] != ref["end"]:
            return 0.0

        adj = _build_adjacency(ref["edges"])
        for i in range(len(pred) - 1):
            u, v = pred[i], pred[i + 1]
            if v not in adj.get(u, set()):
                return 0.0

        return 1.0

    def _extract_path(self, text: str) -> list[str] | None:
        clean = re.sub(r'Path:\s*', '', text, flags=re.IGNORECASE).strip()
        for line in reversed(clean.split("\n")):
            line = line.strip()
            nodes = re.findall(r'[A-Z]', line)
            if len(nodes) >= 2:
                return nodes

        match = re.search(r'([A-Z](?:\s*->\s*[A-Z])+)', text)
        if match:
            nodes = re.findall(r'[A-Z]', match.group(1))
            if len(nodes) >= 2:
                return nodes
        return None
