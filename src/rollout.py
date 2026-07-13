import copy

from datasets import Dataset

from src.task_env import TaskRegistry

_interactive_states: dict[str, dict] = {}


def rollout_episodes(model, tokenizer, task_names: list[str],
                     num_episodes_per_task: int = 32, max_turns: int = 20,
                     temperature: float = 0.8) -> Dataset:
    import torch

    global _interactive_states
    _interactive_states = {}

    examples: list[dict] = []
    model_device = next(model.parameters()).device

    for task_name in task_names:
        task = TaskRegistry.get(task_name)
        if not task.is_interactive:
            continue

        for ep_idx in range(num_episodes_per_task):
            state = task.get_initial_state(ep_idx)
            prompt = task.get_initial_prompt(state)
            conversation_text = prompt

            _interactive_states[conversation_text] = copy.deepcopy(state)
            examples.append({
                "prompt": conversation_text,
                "task_name": task_name,
            })

            for _ in range(max_turns):
                inputs = tokenizer(conversation_text, return_tensors="pt").to(model_device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                action_text = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()

                conversation_text += "\n" + action_text

                result = task.process_action(state, action_text)
                done = result.get("done", False)
                state = result.get("state", state)

                if result.get("observation"):
                    conversation_text += "\n" + result["observation"]

                if done:
                    break

                _interactive_states[conversation_text] = copy.deepcopy(state)
                examples.append({
                    "prompt": conversation_text,
                    "task_name": task_name,
                })

    return Dataset.from_list(examples)


def get_interactive_state(prompt: str) -> dict | None:
    return _interactive_states.get(prompt)
