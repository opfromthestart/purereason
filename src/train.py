import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trl import GRPOConfig, GRPOTrainer  # type: ignore[reportPrivateImportUsage]

from src.config import Config
from src.model import load_model_and_tokenizer, save_model
from src.dataset import build_mixed_dataset
from src.task_env import TaskRegistry

_reward_tracker: dict[str, list[float]] = defaultdict(list)


def create_reward_fn():
    def reward_fn(prompts, completions, task_name=None, **kwargs):
        if task_name is None:
            return [0.0] * len(completions)
        rewards = []
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            try:
                task = TaskRegistry.get(task_name[i])
                r = task.compute_reward(prompt, completion)
                rewards.append(r)
                _reward_tracker[task_name[i]].append(r)
            except Exception as e:
                print(f"[WARN] Reward computation failed: {e}")
                rewards.append(0.0)
        return rewards
    return reward_fn


def log_rewards():
    if not _reward_tracker:
        return
    parts = []
    for task_name in sorted(_reward_tracker):
        vals = _reward_tracker[task_name]
        if vals:
            avg = sum(vals) / len(vals)
            parts.append(f"{task_name}: {avg:.2f} (n={len(vals)})")
    print(f"[REWARD] {' | '.join(parts)}")


def main():
    import argparse
    from src.config import colab_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["local", "colab"], default="local")
    args = parser.parse_args()

    config = colab_config() if args.preset == "colab" else Config()

    import src.tasks.math_task        # noqa
    import src.tasks.code_task        # noqa
    import src.tasks.logic_task       # noqa
    import src.tasks.sokoban_task     # noqa
    import src.tasks.spreadsheet_task # noqa
    import src.tasks.lean_task        # noqa

    os.makedirs(config.training.output_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(config.model)

    dataset = build_mixed_dataset(config.task_sampling)

    checkpoint_dir = config.training.output_dir
    resume = any(
        (Path(checkpoint_dir) / d / "trainer_state.json").exists()
        for d in os.listdir(checkpoint_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))
    )
    if resume:
        print(f"[INFO] Resuming from checkpoint in {checkpoint_dir}")

    grpo_config = GRPOConfig(
        output_dir=config.training.output_dir,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_generations=config.training.num_generations_per_prompt,
        max_prompt_length=config.training.max_prompt_length,
        max_completion_length=config.training.max_completion_length,
        temperature=config.training.temperature,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.max_steps,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_strategy="steps",
        save_total_limit=3,
        bf16=config.training.bf16,
        beta=config.training.beta,
        seed=config.training.seed,
        report_to="wandb",
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=create_reward_fn(),
        args=grpo_config,
        train_dataset=dataset,
    )

    trainer.train(resume_from_checkpoint=resume or None)

    log_rewards()
    save_model(model, tokenizer, os.path.join(config.training.output_dir, "final"))


if __name__ == "__main__":
    main()
