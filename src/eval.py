import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

from src.config import Config
from src.model import load_model_and_tokenizer
from src.task_env import TaskRegistry


def evaluate(model, tokenizer, task_name: str, max_samples: int = 100):
    task = TaskRegistry.get(task_name)
    ds = task.load_dataset()
    ds = ds.select(range(min(len(ds), max_samples)))

    total_reward = 0.0
    results = []

    model.eval()
    for example in tqdm(ds, desc=f"Evaluating {task_name}"):
        prompt = task.get_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        reward = task.compute_reward(prompt, completion)
        total_reward += reward
        results.append({"prompt": prompt[:200], "completion": completion[:200], "reward": reward})

    avg_reward = total_reward / len(ds) if len(ds) > 0 else 0.0
    print(f"{task_name}: avg_reward={avg_reward:.4f} over {len(ds)} samples")
    return avg_reward, results


def main():
    config = Config()

    import src.tasks.math_task        # noqa
    import src.tasks.code_task        # noqa
    import src.tasks.logic_task       # noqa
    import src.tasks.sokoban_task     # noqa
    import src.tasks.spreadsheet_task # noqa
    import src.tasks.lean_task                # noqa
    import src.tasks.knights_knaves_task  # noqa
    import src.tasks.block_world_task         # noqa
    import src.tasks.cryptid_task         # noqa
    import src.tasks.sat_task            # noqa
    import src.tasks.graph_task         # noqa
    import src.tasks.blue_prince_task   # noqa

    model, tokenizer = load_model_and_tokenizer(config.model)

    all_results = {}
    for task_name in TaskRegistry.list_tasks():
        try:
            avg, results = evaluate(model, tokenizer, task_name)
            all_results[task_name] = {"avg_reward": avg, "samples": len(results)}
        except Exception as e:
            print(f"[WARN] Evaluation failed for {task_name}: {e}")

    os.makedirs(config.training.output_dir, exist_ok=True)
    with open(os.path.join(config.training.output_dir, "eval_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nSummary:")
    for name, r in all_results.items():
        print(f"  {name}: {r['avg_reward']:.4f}")


if __name__ == "__main__":
    main()
