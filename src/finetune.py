import os
import torch
from datasets import load_from_disk, Dataset
from typing import List, Tuple, Dict, Any, Optional, Literal
from peft import LoraConfig, get_peft_model
from transformers import (
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForCausalLM,
    TokenizersBackend,
)
from peft import PeftModel

from .utils import get_tokenized_dataset


class Tuner:
    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        training_args: TrainingArguments,
        lora_config: LoraConfig,
        load_in_4bit: bool = True,
    ):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.training_args = training_args
        self.lora_config = lora_config
        self.load_in_4bit = load_in_4bit

        # null fields
        self.model: PeftModel = None
        self.tokenizer: TokenizersBackend = None
        self.dataset: Dataset = None

    def initialize(
        self,
        lora_path: str = None,
    ) -> Tuple[Dataset, PeftModel, TokenizersBackend]:
        tokenizer: TokenizersBackend = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            if self.load_in_4bit
            else None
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
        )
        model.config.use_cache = False
        model = (
            PeftModel.from_pretrained(model, lora_path, is_trainable=True)
            if lora_path is not None
            else model
        )
        dataset = load_from_disk(self.dataset_name)
        dataset = get_tokenized_dataset(tokenizer, dataset)
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        return dataset, model, tokenizer

    def train(
        self,
        save_model: bool = True,
        checkpoint_path: str = None,
    ):
        # TODO: add lora_path
        dataset, model, tokenizer = self.initialize(lora_path=None)
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest"
        )
        model = get_peft_model(model, self.lora_config)
        trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        if checkpoint_path is not None:
            trainer.train(resume_from_checkpoint=checkpoint_path)
        else:
            trainer.train()
        # TODO: improve the save outputs where it saves by dataset
        if save_model:
            lora_save_path = "./lora_adapter/" + self.model_name
            if not os.path.exists(lora_save_path):
                os.mkdir(lora_save_path)
            trainer.save_model(f"{lora_save_path}/{self.dataset_name}")
            torch.save(
                trainer.optimizer.state_dict(),
                f"{lora_save_path}/{self.dataset_name}/optimizer.pt",
            )
        return trainer

    def evaluate(
        self,
    ):
        pass
