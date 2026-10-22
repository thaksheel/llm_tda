import json
import math
from pathlib import Path


def get_lora_options(r,alpha,dropout):
    return dict(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj"
        ],
    )


def get_lora_config(r,alpha,dropout):
    from peft import LoraConfig

    return LoraConfig(**get_lora_options(r, alpha, dropout))


def get_training_options(output_dir,
                      per_device_train_batch_size,
                      gradient_accumulation_steps,
                      num_train_epochs,
                      warmup_steps,
                      max_grad_norm):
    return dict(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_strategy="epoch",
        fp16=True,
        gradient_checkpointing=True,
        num_train_epochs=num_train_epochs,
        # The callback logs at 5% boundaries of total training progress.
        logging_strategy="no",
        # The callback manages quarter-epoch recovery saves and cleanup.
        save_total_limit=None,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        max_grad_norm=max_grad_norm,
        learning_rate=5e-5,
        seed=42,
        data_seed=None,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_steps=-1,
        bf16=False,
    )


def get_training_args(**options):
    from transformers import TrainingArguments

    return TrainingArguments(**get_training_options(**options))


def get_tuner(data_set,model,training_args,lora_config):
    from .finetune import Tuner

    tuner = Tuner(
        dataset_name=data_set,
        model_name=model,
        training_args=training_args,
        lora_config=lora_config,
        load_in_4bit=True,
    )
    return tuner


def train_tuner(tuner, checkpoint_path=None):
    return tuner.train(save_model=True, checkpoint_path=checkpoint_path)


def get_resume_checkpoint(output_dir):
    """Select the newest usable checkpoint, skipping interrupted partial saves."""
    output_path = Path(output_dir)
    checkpoints = []
    if output_path.is_dir():
        for path in output_path.iterdir():
            prefix, separator, step = path.name.partition("-")
            if path.is_dir() and prefix == "checkpoint" and separator and step.isdigit():
                checkpoints.append((int(step), path))
    if not checkpoints:
        raise ValueError(f"No checkpoint-N directory found in output directory: {output_path}")

    last_error = ""
    for step, checkpoint in sorted(checkpoints, key=lambda item: item[0], reverse=True):
        required_files = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_config.json"]
        missing = [name for name in required_files if not (checkpoint / name).is_file()]
        if not any((checkpoint / name).is_file() for name in
                   ("adapter_model.safetensors", "adapter_model.bin")):
            missing.append("adapter weights")
        if missing:
            last_error = f"Incomplete checkpoint {checkpoint}; missing: {', '.join(missing)}"
            print(f"Skipping {last_error}", flush=True)
            continue
        try:
            state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
            if state["global_step"] != step:
                raise ValueError("global_step does not match checkpoint directory")
        except (OSError, ValueError, KeyError, TypeError) as error:
            last_error = f"Invalid Trainer state in checkpoint {checkpoint}: {error}"
            print(f"Skipping {last_error}", flush=True)
            continue
        return str(checkpoint)
    raise ValueError(f"No usable checkpoint found in {output_path}. {last_error}")


def check_training_artifacts(model_name, dataset_name, lora_options, training_options,
                             artifact_dir=None, target_epoch=None):
    """Return the artifact directory and whether its completed run is reusable."""
    artifact_dir = Path(artifact_dir) if artifact_dir is not None else Path(
        f"./lora_adapter/{model_name}/{dataset_name}"
    )
    required_files = [
        "adapter_config.json", "optimizer.pt", "optimizer_parameters.json",
        "training_args.json", "trainer_state.json", "metadata.json",
        "complete.json", "tokenizer_config.json",
    ]
    missing = [name for name in required_files
               if not (artifact_dir / name).is_file()]
    if missing:
        return artifact_dir, False, "Missing artifacts: " + ", ".join(missing)

    if not any((artifact_dir / name).is_file() for name in
               ("adapter_model.safetensors", "adapter_model.bin")):
        return artifact_dir, False, "Missing adapter weights"
    if not any((artifact_dir / name).is_file() for name in
               ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json", "vocab.txt")):
        return artifact_dir, False, "Missing tokenizer vocabulary"

    try:
        metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        complete = json.loads((artifact_dir / "complete.json").read_text(encoding="utf-8"))
        progress = json.loads((artifact_dir / "trainer_state.json").read_text(encoding="utf-8"))
        saved_lora = json.loads((artifact_dir / "adapter_config.json").read_text(encoding="utf-8"))
        saved_training = json.loads((artifact_dir / "training_args.json").read_text(encoding="utf-8"))
        parameter_groups = json.loads((artifact_dir / "optimizer_parameters.json").read_text(encoding="utf-8"))
        json.loads((artifact_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
        if metadata["schema_version"] != 1:
            return artifact_dir, False, "Unsupported artifact metadata version"
        if not metadata["source_fingerprint"] or metadata["training_examples"] <= 0:
            return artifact_dir, False, "Missing training dataset identity"
        step = metadata["global_step"]
        if not isinstance(step, int) or step <= 0 or not (
            step == complete["global_step"] == progress["global_step"]
        ):
            return artifact_dir, False, "Inconsistent training completion state"
        expected_epoch = training_options["num_train_epochs"] if target_epoch is None else target_epoch
        completed_epoch = float(progress["epoch"])
        if not math.isfinite(completed_epoch) or not math.isclose(
            completed_epoch, float(expected_epoch), abs_tol=1e-6, rel_tol=0
        ):
            return artifact_dir, False, "Artifacts are not at the requested completed epoch"
        if not math.isclose(float(metadata["trained_epochs"]), completed_epoch,
                            abs_tol=1e-6, rel_tol=0):
            return artifact_dir, False, "Inconsistent completed epoch metadata"
        # AdamW may have an empty no-weight-decay group for LoRA-only training.
        # Preserve group positions; only the mapping as a whole must be nonempty.
        if not isinstance(parameter_groups, list) or not parameter_groups or not all(
            isinstance(group, list) and all(
                isinstance(parameter, dict) and
                {"optimizer_id", "name", "shape"} <= parameter.keys()
                for parameter in group
            ) for group in parameter_groups
        ) or not any(parameter_groups):
            return artifact_dir, False, "Invalid optimizer parameter mapping"

        expected_metadata = {
            "model_name": model_name,
            "load_in_4bit": True,
            "max_length": 512,
        }
        for key, value in expected_metadata.items():
            if metadata[key] != value:
                return artifact_dir, False, f"Changed setting: {key}"

        expected_lora = dict(bias="none", use_rslora=False, **lora_options)
        for key in ("r", "lora_alpha", "lora_dropout", "task_type", "bias", "use_rslora"):
            if saved_lora.get(key) != expected_lora.get(key):
                return artifact_dir, False, f"Changed LoRA setting: {key}"
        if set(saved_lora["target_modules"]) != set(expected_lora["target_modules"]):
            return artifact_dir, False, "Changed LoRA target modules"

        expected_training = training_options
        # Output, logging, and checkpoint-retention settings do not change weights.
        for key in (
            "per_device_train_batch_size", "gradient_accumulation_steps",
            "num_train_epochs", "max_steps", "learning_rate", "optim",
            "lr_scheduler_type", "warmup_steps", "max_grad_norm",
            "fp16", "bf16", "gradient_checkpointing", "seed", "data_seed",
            "weight_decay", "adam_beta1", "adam_beta2", "adam_epsilon",
        ):
            if key == "num_train_epochs" and target_epoch is not None:
                # Select a completed epoch from a longer training run.
                continue
            if saved_training.get(key) != expected_training.get(key):
                return artifact_dir, False, f"Changed training setting: {key}"
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return artifact_dir, False, "Unreadable or invalid artifact metadata"

    print("Loading dataset library and checking the training dataset fingerprint...", flush=True)
    from datasets import load_from_disk

    dataset = load_from_disk(dataset_name)
    if metadata["source_fingerprint"] != dataset._fingerprint or metadata["training_examples"] != len(dataset):
        return artifact_dir, False, "Changed training dataset"

    return artifact_dir, True, "Completed artifacts match the requested training run"


def find_epoch_artifacts(output_dir, model_name, dataset_name, lora_options, training_options):
    """Find a verified completed epoch without loading a model or starting training."""
    output_path = Path(output_dir)
    if not output_path.is_dir():
        return None
    candidates = []
    for path in output_path.iterdir():
        prefix, separator, step = path.name.partition("-")
        if path.is_dir() and prefix == "checkpoint" and separator and step.isdigit():
            candidates.append((int(step), path))
    for _, path in sorted(candidates, reverse=True):
        try:
            progress = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
            if not math.isclose(float(progress["epoch"]), float(training_options["num_train_epochs"]),
                                abs_tol=1e-6, rel_tol=0):
                continue
        except (OSError, ValueError, KeyError, TypeError):
            continue
        directory, reusable, _ = check_training_artifacts(
            model_name, dataset_name, lora_options, training_options,
            artifact_dir=path, target_epoch=training_options["num_train_epochs"],
        )
        if reusable:
            return directory
    return None
