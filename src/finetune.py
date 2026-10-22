import json
import math
import shutil
from pathlib import Path
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
    TrainerCallback,
)
from peft import PeftModel

from .utils import get_tokenized_dataset


class EpochArtifactsCallback(TrainerCallback):
    """Save quarter-epoch recovery points and protect completed epochs."""

    def __init__(self, tuner):
        self.tuner = tuner
        self.trainer = None
        self.last_quarter = 0
        self.last_progress = 0

    def on_train_begin(self, args, state, control, **kwargs):
        # Resume from the current epoch rather than replaying previous quarters.
        self.last_quarter = math.floor(float(state.epoch or 0) * 4 + 1e-8)
        total = getattr(state, "max_steps", 0)
        self.last_progress = (20 * getattr(state, "global_step", 0) // total) if total > 0 else 0
        return control

    def on_step_end(self, args, state, control, **kwargs):
        total = getattr(state, "max_steps", 0)
        if total > 0:
            progress = 20 * state.global_step // total
            if progress > self.last_progress:
                self.last_progress = progress
                control.should_log = True
        epoch = float(state.epoch or 0)
        quarter = math.floor(epoch * 4 + 1e-8)
        if quarter > self.last_quarter:
            self.last_quarter = quarter
            # The epoch-end event saves the fourth quarter separately.
            if not math.isclose(epoch, round(epoch), abs_tol=1e-6, rel_tol=0):
                control.should_save = True
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        # The fourth save is the durable completed-epoch checkpoint.
        control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if not getattr(args, "should_save", True):
            return control
        epoch = state.epoch
        if epoch is not None and epoch >= 1 and abs(epoch - round(epoch)) < 1e-6:
            self.tuner.save_training_artifacts(
                self.trainer,
                save_dir=Path(args.output_dir) / f"checkpoint-{state.global_step}",
            )
        self.prune_intermediate_checkpoints(args.output_dir)
        return control

    def prune_intermediate_checkpoints(self, output_dir, keep=1):
        root = Path(output_dir).resolve()
        if not root.is_dir():
            return
        intermediate = []
        for path in root.iterdir():
            prefix, separator, step = path.name.partition("-")
            if prefix != "checkpoint" or not separator or not step.isdigit():
                continue
            # Do not follow links or delete outside this exact output directory.
            if path.is_symlink() or not path.is_dir() or path.resolve().parent != root:
                continue
            try:
                state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
                epoch = float(state["epoch"])
                if state["global_step"] != int(step) or not math.isfinite(epoch) or epoch < 0:
                    continue
            except (OSError, ValueError, KeyError, TypeError):
                # Unknown/partial saves are retained rather than deleted.
                continue
            if epoch >= 1 and math.isclose(epoch, round(epoch), abs_tol=1e-6, rel_tol=0):
                continue
            intermediate.append((int(step), path))

        ordered = sorted(intermediate, key=lambda item: item[0])
        expired = ordered if keep == 0 else ordered[:-keep]
        for _, path in expired:
            if path.is_symlink() or path.resolve().parent != root:
                raise ValueError(f"Refusing to delete checkpoint outside {root}: {path}")
            print(f"Removing old intermediate checkpoint: {path}", flush=True)
            shutil.rmtree(path)


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
        self.max_length = 512
        self.source_fingerprint = None

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
        self.source_fingerprint = dataset._fingerprint
        dataset = get_tokenized_dataset(
            tokenizer, dataset, max_length=self.max_length
        )
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        return dataset, model, tokenizer

    def train(
        self,
        save_model: bool = True,
        checkpoint_path: str = None,
    ):
        dataset, model, tokenizer = self.initialize(lora_path=None)
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest"
        )
        model = get_peft_model(model, self.lora_config)
        self.model = model
        epoch_artifacts = EpochArtifactsCallback(self)
        trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            callbacks=[epoch_artifacts],
        )
        epoch_artifacts.trainer = trainer
        if checkpoint_path is not None:
            trainer.train(resume_from_checkpoint=checkpoint_path)
        else:
            trainer.train()
        if save_model:
            self.save_training_artifacts(trainer)
        # Only clean recovery saves after training and final artifact saving
        # succeed. Exceptions/interruptions leave the last recovery point intact.
        epoch_artifacts.prune_intermediate_checkpoints(self.training_args.output_dir, keep=0)
        return trainer

    def save_training_artifacts(self, trainer, save_dir=None):
        # Preserve the existing adapter location used by src/run_tda.py.
        save_dir = Path(save_dir) if save_dir is not None else Path(
            f"./lora_adapter/{self.model_name}/{self.dataset_name}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        complete_path = save_dir / "complete.json"
        complete_path.unlink(missing_ok=True)

        optimizer = trainer.optimizer
        if optimizer is None:
            raise ValueError("Cannot save training artifacts without optimizer state")

        optimizer_state = optimizer.state_dict()
        names_by_id = {
            id(parameter): name
            for name, parameter in trainer.model.named_parameters()
        }
        parameter_groups = []
        for live_group, saved_group in zip(
            optimizer.param_groups, optimizer_state["param_groups"], strict=True
        ):
            parameters = []
            for parameter, saved_id in zip(
                live_group["params"], saved_group["params"], strict=True
            ):
                parameters.append({
                    "optimizer_id": saved_id,
                    "name": names_by_id[id(parameter)],
                    "shape": list(parameter.shape),
                })
            parameter_groups.append(parameters)

        trainer.save_model(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))
        # Keep the full state dictionary, including optimizer parameter groups.
        torch.save(optimizer_state, save_dir / "optimizer.pt")
        (save_dir / "optimizer_parameters.json").write_text(
            json.dumps(parameter_groups, indent=2), encoding="utf-8"
        )
        (save_dir / "training_args.json").write_text(
            trainer.args.to_json_string(), encoding="utf-8"
        )
        trainer.state.save_to_json(str(save_dir / "trainer_state.json"))

        metadata = {
            "schema_version": 1,
            "model_name": self.model_name,
            "model_revision": getattr(trainer.model.config, "_commit_hash", None),
            "dataset_path": self.dataset_name,
            "source_fingerprint": self.source_fingerprint,
            "tokenized_fingerprint": self.dataset._fingerprint,
            "training_examples": len(self.dataset),
            "load_in_4bit": self.load_in_4bit,
            "max_length": self.max_length,
            "padding_side": self.tokenizer.padding_side,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "global_step": trainer.state.global_step,
            "trained_epochs": trainer.state.epoch,
            "recovery_saves_per_epoch": 4,
        }
        (save_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        # Publish completion only after every artifact saves successfully.
        pending_path = save_dir / "complete.json.tmp"
        pending_path.write_text(
            json.dumps({"global_step": trainer.state.global_step}), encoding="utf-8"
        )
        pending_path.replace(complete_path)
        return save_dir

