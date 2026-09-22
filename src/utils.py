import torch 
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TokenizersBackend

def get_tokenized_dataset(tokenizer: TokenizersBackend, dataset: Dataset, max_length=512):
    def tokenize(sample):
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

    return dataset.map(tokenize, remove_columns=list(dataset.features))


def get_tokenized_text(tokenizer: TokenizersBackend, sample, device, max_length=512):
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
                prompt_only_text, truncation=True, max_length=max_length, padding=False
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
        "input_ids": torch.tensor([input_ids]).to(device),
        "labels": torch.tensor([labels]).to(device),
        "attention_mask": torch.tensor([full_tokenized["attention_mask"]]).to(device),
    }
