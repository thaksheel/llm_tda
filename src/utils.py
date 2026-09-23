from dataclasses import dataclass
from typing import Literal, Optional, Dict, List 
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TokenizersBackend
from transformers import TokenizersBackend
from peft import PeftModel


@dataclass
class Result:
    top_k: int
    auPRC: float
    P_at_k: float
    R_at_k: float
    MRR: float
    method: str

    def __repr__(self):
        return (
            f"Result(top_k={self.top_k}, "
            f"method='{self.method}', "
            f"auPRC={self.auPRC:.4f}, "
            f"P@k={self.P_at_k:.4f}, "
            f"R@k={self.R_at_k:.4f}, "
            f"MRR={self.MRR:.4f})"
        )

@dataclass
class Params:
    """Parameters for TDA in TDAEvaluation for each of use across self.traces evaluations for all methods."""
    layer_norm: bool = True
    layer: int = -1
    optimizer_state = None # load optimizer.pt then query for ['state']

def tokenize(tokenizer: TokenizersBackend, sample: Dict, max_length=512):
    if tokenizer.chat_template:
        message = [
            {"role": "user", "content": sample["prompts"]},
            {"role": "assistant", "content": sample["response"]},
        ]
        full_text = tokenizer.apply_chat_template(message, tokenize=False)
        full_tokenized = tokenizer(
            full_text, truncation=True, max_length=512, padding=False
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
            full_text, truncation=True, max_length=512, padding=False
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
    self,
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
