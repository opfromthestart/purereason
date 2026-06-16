from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    fallback_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class TrainingConfig:
    output_dir: str = "./output"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_generations_per_prompt: int = 4
    max_prompt_length: int = 512
    max_completion_length: int = 2048
    temperature: float = 0.8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 50
    max_steps: int = 1000
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    beta: float = 0.0  # KL coefficient — disabled
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True


@dataclass
class TaskSamplingConfig:
    tasks: dict[str, float] = field(default_factory=lambda: {
        "gsm8k": 0.25,
        "math": 0.15,
        "mbpp": 0.15,
        "prontoqa": 0.15,
        "lean_minif2f": 0.10,
        "sokoban": 0.10,
        "spreadsheet": 0.10,
    })
    max_samples_per_task: int = 2000


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    task_sampling: TaskSamplingConfig = field(default_factory=TaskSamplingConfig)
