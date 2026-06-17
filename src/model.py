import unsloth  # noqa: must be first
from unsloth import FastLanguageModel
import torch
from src.config import ModelConfig


def load_model_and_tokenizer(config: ModelConfig) -> tuple:
    kwargs = {
        "model_name": config.base_model,
        "max_seq_length": config.max_seq_length,
        "load_in_4bit": config.load_in_4bit,
        "use_gradient_checkpointing": "unsloth" if config.gradient_checkpointing else False,
    }
    if not config.load_in_4bit:
        kwargs["dtype"] = getattr(torch, config.bnb_4bit_compute_dtype)

    model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout if not config.load_in_4bit else 0.0,
        target_modules=config.lora_target_modules,
        use_gradient_checkpointing="unsloth" if config.gradient_checkpointing else False,
        random_state=config.seed,
    )

    return model, tokenizer


def save_model(model, tokenizer, path: str):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
