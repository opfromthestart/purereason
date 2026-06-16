# PureReason: RL-Based LLM Reasoning Training (No KL Divergence)

## Overview

Train a small LLM (~1.5B params) using reinforcement learning for multi-domain reasoning tasks without any KL divergence penalty. The model is allowed to diverge arbitrarily from its base checkpoint — the only constraint is reward signal. This explores whether pure RL without distribution constraints can produce useful reasoning behavior or collapses into gibberish.

Target hardware: GTX 1650 Ti (4 GB VRAM).

## Model & Quantization

- **Base model**: `Qwen/Qwen2.5-1.5B-Instruct` (fallback: `Qwen/Qwen2.5-0.5B-Instruct`)
- **Quantization**: 4-bit NF4 via `bitsandbytes` (~500 MB for 1.5B weights)
- **Adapters**: LoRA on all linear layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj). Rank 16, alpha 32. (~25 MB trainable params)
- **Optimizer**: 8-bit AdamW on LoRA weights only (~200 MB)
- **Estimated total VRAM**: ~3.1 GB, leaving ~0.9 GB headroom (assuming batch=4, 2048-token generations)

## RL Algorithm

**GRPO** (Group Relative Policy Optimization) via HuggingFace TRL.

- No reference model needed (saves memory)
- No value/critic network needed (saves memory)
- Per prompt: sample N=4 completions, compute rewards, normalize within group as advantages
- Loss: `L = -E[min(r * A, clip(r, 1-ε, 1+ε) * A)]` where `r = π_new / π_old`
- KL coefficient set to `0.0` — the model can freely diverge

### Training Loop (per step)
1. Sample batch of prompts from mixed task dataset
2. Generate N=4 completions (temperature=0.8, max_new_tokens=2048)
3. Compute reward per completion via task-specific reward functions
4. Group-normalize rewards → advantages
5. Compute GRPO loss, backward, step optimizer
6. Log per-task rewards

## Benchmarks & Reward Functions

Four task domains, all with deterministic verifiable rewards (no LLM-as-judge):

### Math — GSM8K + MATH
- **Dataset**: `openai/gsm8k` (7.5k train) + `hendrycks/competition_math` (7.5k train)
- **Prompt**: "Solve this math problem step by step: {question}"
- **Reward**: Extract answer after `\boxed{}` or `####`, compare to ground truth with `sympy.simplify()`. 1.0 if match, 0.0 otherwise.

### Code — HumanEval + MBPP
- **Dataset**: `openai/openai_humaneval` (eval only, 164 problems) + `google-research-datasets/mbpp` (374 train)
- **Prompt**: "Complete this Python function:\n{prompt}"
- **Reward**: Extract code block, execute with test cases in `subprocess` sandbox. `pass_rate = passed / total_tests`.

### Logic — PRONTOQA
- **Dataset**: `ashen/PRONTOQA` (deductive reasoning with proof chains)
- **Prompt**: "Given these facts: {context}\nAnswer: {question}"
- **Reward**: 1.0 if final answer matches ground truth, 0.0 otherwise.

### Lean Proofs — miniF2F
- **Dataset**: `lean-dojo/LeanDojo` with `miniF2F` benchmark (488 problems, 244 train / 244 test)
- **Prompt**: "Prove this theorem in Lean 4: {theorem_statement}"
- **Reward**: Binary — proof compiles via LeanDojo environment = 1.0, otherwise 0.0. Partial credit via number of valid tactics before first error (optional).
- **Environment**: Lean 4 toolchain via `elan` + `lake`. Installed via `scripts/setup_lean.sh`.

### Sokoban (Tool-Use) — Procedurally Generated Puzzles
- **Environment**: Custom text-based Sokoban simulator. Generated puzzles with varying grid sizes (6x6 to 10x10), walls, boxes, and targets. Model sees the grid as ASCII art and outputs move sequences.
- **Prompt**: "You control an agent (@) on a grid. Push boxes ($) onto targets (.) to score. Moves: U D L R. Here is the board:\n{board}\n\nOutput your moves as a sequence on one line:"
- **Reward**: `boxes_on_target / total_targets` after executing the move sequence. Walls (#), agent cannot walk through boxes/walls. Moves past the puzzle limit are ignored.
- **Multi-turn variant** (future): Model sees updated board state after each move and can replan.

### Spreadsheet (Tool-Use) — Synthesized Grid Tasks
- **Environment**: Custom spreadsheet simulator (10x10 grid). Tasks include formula application (SUM, AVG, COUNT, IF), cell lookups, and data transformation.
- **Prompt**: "You have a spreadsheet. Cells are referenced as A1, B2, etc. Write formulas to compute the requested result.\n\nGrid:\n{grid}\n\nTask: {task_description}\n\nOutput your answer as: FORMULA: <cell>=<formula> or VALUE: <result>"
- **Reward**: 1.0 if the final cell value matches the expected result, 0.0 otherwise. Grid formulas support `SUM(range)`, `AVG(range)`, `COUNT(range)`, `IF(cond, true_val, false_val)`.
- **Dataset**: Procedurally generated — random grids with numeric data and templated tasks.

## Task Registry (Plugin Architecture)

All tasks implement a common `TaskEnv` interface. Adding a new benchmark requires zero changes to the training loop.

```python
class TaskEnv(ABC):
    name: str

    @abstractmethod
    def load_dataset(self) -> Dataset:
        """Load and return a HuggingFace Dataset of prompts."""
        ...

    @abstractmethod
    def get_prompt(self, example: dict) -> str:
        """Format a single prompt from a dataset row."""
        ...

    @abstractmethod
    def compute_reward(self, prompt: str, completion: str) -> float:
        """Compute reward 0.0--1.0 for a completion."""
        ...
```

Tasks are registered via decorator:

```python
@TaskRegistry.register("gsm8k")
class GSM8KTask(TaskEnv):
    ...
```

To add a new task: create `src/tasks/<name>.py` with a `TaskEnv` subclass and decorate it. The training loop discovers registered tasks automatically.

## Project Structure

```
purereason/
├── src/
│   ├── train.py              # Main training entry point
│   ├── config.py             # All hyperparameters, paths, registry
│   ├── model.py              # QLoRA model loading, LoRA config
│   ├── task_env.py           # TaskEnv base class + TaskRegistry
│   ├── dataset.py            # Mixed dataset loader + sampling
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── math_task.py      # GSM8K + MATH
│   │   ├── code_task.py      # HumanEval + MBPP
│   │   ├── logic_task.py     # PRONTOQA
│   │   ├── lean_task.py      # miniF2F via LeanDojo
│   │   ├── sokoban_task.py   # Sokoban puzzle solver
│   │   └── spreadsheet_task.py  # Spreadsheet formula tasks
│   ├── envs/                  # RL environment simulators (pure Python)
│   │   ├── __init__.py
│   │   ├── sokoban_env.py    # Text-based Sokoban simulator
│   │   └── spreadsheet_env.py # Grid formula evaluator
│   └── eval.py               # Evaluation on held-out benchmarks
├── scripts/
│   └── setup_lean.sh         # Install Lean 4 toolchain
├── requirements.txt
└── pyproject.toml
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Tensor compute |
| `transformers` | Model loading, tokenizer |
| `trl` | GRPO trainer |
| `peft` | LoRA adapters |
| `bitsandbytes` | 4-bit quantization |
| `datasets` | Dataset loading/preprocessing |
| `sympy` | Math answer verification |
| `lean-dojo` | Lean 4 proof environment |
| `wandb` / `tensorboard` | Training logging |

## Error Handling

- **Code execution**: Subprocess with 5-second timeout. Syntax errors, runtime errors, and timeouts all → reward 0.0. No imports beyond stdlib allowed unless whitelisted.
- **Lean verification**: Timeout after 30 seconds per proof attempt. Compiler errors → reward 0.0.
- **Math verification**: If answer extraction fails (no `\boxed{}` or `####` found), reward 0.0.
- **GPU OOM**: Graceful fallback from 1.5B → 0.5B model with config flag. Batch size auto-reduction on first OOM.
- **Missing environment**: Skip Lean tasks if Lean toolchain not installed, log warning. Other tasks continue unaffected.

## Open Questions

- Will the model collapse into gibberish without KL? Monitor entropy + perplexity alongside reward.
- GRPO uses an implicit KL via importance ratio clipping (ε=0.2). To remove even that, fallback to pure REINFORCE: `loss = -log_prob * advantage` without clipping.
- Partial credit for Lean proofs: count valid tactics before first error, or binary only? Start binary, iterate if signal is too sparse.
