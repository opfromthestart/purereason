#!/usr/bin/env python3
"""Clone community ARC-AGI-3 environment repos and register their games."""

import os
import subprocess
import sys

COMMUNITY_REPOS = {
    "witness": {
        "url": "https://github.com/Guanghan/arc-witness-envs.git",
        "env_dir": "environment_files",
        "task_name": "arc_agi_witness",
    },
    "arc_interactive": {
        "url": "https://github.com/theredbluepill/arc-interactive.git",
        "env_dir": "environment_files",
        "task_name": "arc_agi_community",
    },
}


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    community_dir = os.path.join(base_dir, "community_envs")
    os.makedirs(community_dir, exist_ok=True)

    for name, info in COMMUNITY_REPOS.items():
        repo_dir = os.path.join(community_dir, name)
        if not os.path.exists(repo_dir):
            print(f"Cloning {name} from {info['url']}...")
            subprocess.run(
                ["git", "clone", "--depth=1", info["url"], repo_dir],
                check=True,
            )
        else:
            print(f"{name} already cloned at {repo_dir}")

        env_path = os.path.join(repo_dir, info["env_dir"])
        if os.path.isdir(env_path):
            count = len([d for d in os.listdir(env_path)
                        if os.path.isdir(os.path.join(env_path, d))])
            print(f"  {info['task_name']}: {count} games in {env_path}")
        else:
            print(f"  WARNING: {env_path} not found")

    print("\nDone. Add these tasks to your config:")
    print('  "arc_agi_witness": 0.05,')
    print('  "arc_agi_community": 0.05,')


if __name__ == "__main__":
    main()
