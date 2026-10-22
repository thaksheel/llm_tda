import json
import math
import shutil
import uuid
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
from .run_ft import checkpoint_identity


class EpochArtifactsCallback(TrainerCallback):
    """Save quarter-epoch recovery points and protect completed epochs."""

    def __init__(self, tuner):
        self.tuner = tuner
        self.trainer = None
        self.last_quarter = 0
        self.last_progress = 0
        self.lr_interval = None
        self.pending_lr = None
        self.ancestors = []

    def on_train_begin(self, args, state, control, **kwargs):
        # Resume from the current epoch rather than replaying previous quarters.
        self.last_quarter = math.floor(float(state.epoch or 0) * 4 + 1e-8)
        total = getattr(state, "max_steps", 0)
        self.last_progress = (20 * getattr(state, "global_step", 0) // total) if total > 0 else 0
        checkpoint = getattr(self.tuner, "resume_checkpoint", None)
        if isinstance(checkpoint, (str, Path)):
            metadata_path = Path(checkpoint) / "metadata.json"
            if metadata_path.is_file():
                parent = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.ancestors = list(parent.get("training_ancestors", []))
                identity = checkpoint_identity(checkpoint)
                if identity not in self.ancestors:
                    self.ancestors.append(identity)
                if parent.get("training_run_id"):
                    self.tuner.training_run_id = parent["training_run_id"]
            rate_path = Path(checkpoint) / "tracin_lr.json"
            if rate_path.is_file():
                saved = json.loads(rate_path.read_text(encoding="utf-8"))
                if saved["global_step"] != state.global_step:
                    raise ValueError("Saved TracIn learning-rate state does not match resume checkpoint")
                self.lr_interval = saved["interval"]
                self.tuner.training_run_id = saved["training_run_id"]
                for ancestor in saved.get("training_ancestors", []):
                    if ancestor not in self.ancestors:
                        self.ancestors.append(ancestor)
            elif state.epoch and not math.isclose(float(state.epoch), round(float(state.epoch)), abs_tol=1e-6):
                print("Historical epoch learning rates are unavailable for this legacy recovery checkpoint.", flush=True)
        self.tuner.tracin_interval = self.lr_interval
        self.tuner.training_ancestors = list(self.ancestors)
        return control

    def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
        if optimizer is None:
            return control
        epoch = math.floor(float(state.epoch or 0) + 1e-8) + 1
        if self.lr_interval is None or self.lr_interval["epoch"] != epoch:
            self.lr_interval = {"epoch": epoch, "start_step": state.global_step,
                                "end_step": state.global_step, "count": 0, "sum": 0.0,
                                "complete_start": math.isclose(float(state.epoch or 0), epoch - 1, abs_tol=1e-6)}
        rates = [float(group["lr"]) for group in optimizer.param_groups if group["params"]]
        if not rates or any(not math.isclose(rate, rates[0]) for rate in rates):
            raise ValueError("TracIn rate tracking requires a shared learning rate for trained parameter groups")
        self.pending_lr = rates[0]
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.pending_lr is not None:
            skipped = getattr(kwargs.get("optimizer"), "step_was_skipped", False)
            self.lr_interval["sum"] += 0.0 if skipped else self.pending_lr
            self.lr_interval["count"] += 1
            self.lr_interval["end_step"] = state.global_step
            self.tuner.tracin_interval = dict(self.lr_interval)
            self.pending_lr = None
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
        if self.lr_interval is not None:
            path = Path(args.output_dir) / f"checkpoint-{state.global_step}" / "tracin_lr.json"
            pending = path.with_suffix(".json.tmp")
            pending.write_text(json.dumps({"global_step": state.global_step,
                "training_run_id": self.tuner.training_run_id, "interval": self.lr_interval,
                "training_ancestors": self.ancestors}), encoding="utf-8")
            pending.replace(path)
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
        self.training_run_id = uuid.uuid4().hex
        self.resume_checkpoint = None
        self.tracin_interval = None
        self.training_ancestors = []

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
        self.resume_checkpoint = checkpoint_path
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
            "training_run_id": self.training_run_id,
            "training_ancestors": self.training_ancestors,
            "tracin_learning_rate": self.tracin_interval,
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

