from dataclasses import dataclass
from typing import Dict

from datasets import Dataset
from transformers import TokenizersBackend
from .results import Result


@dataclass
class Params:
    """Options for direct feature extraction through TDAEvaluation."""

    layer_norm: bool = True
    layer: int = -1
    optimizer_state: Dict = None  # Per-parameter optimizer state, without parameter groups.


def tokenize(sample: Dict, tokenizer: TokenizersBackend, max_length=512):
    if tokenizer.chat_template:
        message = [
            {"role": "user", "content": sample["prompts"]},
            {"role": "assistant", "content": sample["response"]},
        ]
        full_text = tokenizer.apply_chat_template(message, tokenize=False)
        full_tokenized = tokenizer(
            full_text, truncation=True, max_length=max_length, padding=False
        )
        prompt_only_text = tokenizer.apply_chat_template(
            [message[0]], tokenize=False, add_generation_prompt=True
        )
        prompt_length = len(
            tokenizer(
                prompt_only_text,
                truncation=True,
                max_length=max_length,
                padding=False,
            )["input_ids"]
        )
    else:
        full_text = (
            "[INST] "
            + sample["prompts"]
            + " [/INST]"
            + sample["response"]
            + tokenizer.eos_token
        )
        full_tokenized = tokenizer(
            full_text, truncation=True, max_length=max_length, padding=False
        )
        prompt_length = len(
            tokenizer(
                "[INST] " + sample["prompts"] + " [/INST]",
                truncation=True,
                max_length=max_length,
                padding=False,
            )["input_ids"]
        )

    input_ids = full_tokenized["input_ids"]
    labels = list(input_ids)
    labels[:prompt_length] = [-100] * prompt_length
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": full_tokenized["attention_mask"],
    }

def get_tokenized_dataset(
    tokenizer: TokenizersBackend,
    dataset: Dataset,
    max_length=512,
):
    return dataset.map(
        tokenize,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": max_length,
        },
        remove_columns=list(dataset.features),
    )
