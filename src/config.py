from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    base_model: str = "LiquidAI/LFM2.5-350M"
    fallback_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_seq_length: int = 4096
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    gradient_checkpointing: bool = True
    seed: int = 42


@dataclass
class TrainingConfig:
    output_dir: str = "./output"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_generations_per_prompt: int = 4
    max_prompt_length: int = 1024
    max_completion_length: int = 512
    temperature: float = 0.8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 50
    max_steps: int = 1000
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    beta: float = 0.0
    seed: int = 42
    bf16: bool = False


@dataclass
class TaskSamplingConfig:
    tasks: dict[str, float] = field(default_factory=lambda: {
        "gsm8k": 0.25,
        "math": 0.15,
        "mbpp": 0.15,
        "prontoqa": 0.15,
        "lean_minif2f": 0.00,
        "sokoban": 0.10,
        "spreadsheet": 0.10,
        "knights_knaves": 0.10,
        "block_world": 0.10,
        "block_world_interactive": 0.05,
        "cryptid": 0.05,
        "sokoban_interactive": 0.05,
        "sat_counterfactual": 0.05,
        "graph_walk": 0.10,
        "blue_prince": 0.10,
        "arc_agi": 0.05,
        "arc_agi_one": 0.10,
        "arc_agi_two": 0.10,
        "arc_agi_witness": 0.05,
        "arc_agi_community": 0.05,
    })
    max_samples_per_task: int = 2000


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    task_sampling: TaskSamplingConfig = field(default_factory=TaskSamplingConfig)


def colab_config() -> Config:
    """Configuration for Google Colab T4/A100 (16GB+ VRAM)."""
    return Config(
        model=ModelConfig(
            base_model="LiquidAI/LFM2.5-350M",
            fallback_model="Qwen/Qwen2.5-0.5B-Instruct",
            max_seq_length=8192,
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            gradient_checkpointing=True,
            seed=42,
        ),
        training=TrainingConfig(
            output_dir="./output",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations_per_prompt=4,
            max_prompt_length=512,
            max_completion_length=2048,
            temperature=0.8,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_steps=50,
            max_steps=5000,
            logging_steps=10,
            save_steps=200,
            bf16=True,
            beta=0.0,
            seed=42,
        ),
        task_sampling=TaskSamplingConfig(
            tasks={
                "gsm8k": 0.25,
                "math": 0.20,
                "mbpp": 0.15,
                "prontoqa": 0.15,
                "lean_minif2f": 0.00,
                "sokoban": 0.10,
                "spreadsheet": 0.10,
                "knights_knaves": 0.10,
                "block_world": 0.10,
                "block_world_interactive": 0.05,
                "cryptid": 0.05,
                "sokoban_interactive": 0.05,
                "sat_counterfactual": 0.05,
                "graph_walk": 0.10,
                "blue_prince": 0.10,
                "arc_agi": 0.05,
                "arc_agi_one": 0.10,
                "arc_agi_two": 0.10,
                "arc_agi_witness": 0.05,
                "arc_agi_community": 0.05,
            },
            max_samples_per_task=2000,
        ),
    )
