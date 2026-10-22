import os
import gc
import math
from pathlib import Path
from argparse import ArgumentParser
from src.run_tda import run_attribution
from src.run_ft import (
    get_lora_config, get_training_args, get_tuner, train_tuner,
    check_training_artifacts, get_lora_options, get_training_options,
    get_resume_checkpoint,
    find_epoch_artifacts,
)

def main():
    print("Starting TDA pipeline; reading command-line arguments...", flush=True)
    # Keep RUN_PIPELINE.md updated when arguments, defaults, or behavior change.
    # Record significant correctness/recovery changes in Changes.md.
    parser = ArgumentParser(description="Command line flags for TDA pipeline.")


    #============================#
    #   lora_config(src/run_ft.py)   #
    #============================#
    parser.add_argument("--r",
                        type=int,
                        help="LoRA adapter rank; higher values add trainable parameters and use more memory",
                        default=8)
    
    parser.add_argument("--alpha",
                        type=int,
                        help="LoRA scaling factor; adapter updates are scaled by alpha / r",
                        default=16)
    
    parser.add_argument("--dropout",
                        type=float,
                        help="Dropout probability for LoRA layers, from 0 to less than 1",
                        default=0.1)


    #============================#
    #  training_args(src/run_ft.py)  #
    #============================#
    parser.add_argument("output_dir",
                        type=str,
                        help="Directory for Trainer checkpoints and training outputs (required positional argument)")
    
    parser.add_argument("--batch",
                        type=int,
                        help="Training examples processed per device in each forward/backward pass",
                        default=1)

    parser.add_argument("--accum",
                        type=int,
                        help="Number of batches accumulated before each optimizer update",
                        default=16)

    parser.add_argument("--epochs",
                        type=int,
                        help="Training epoch target, or completed epoch to select for attribution",
                        default=3)

    parser.add_argument("--warmup",
                        type=float,
                        help="Learning-rate warmup: a fraction below 1 (0.1 means 10%%), or a whole step count such as 100",
                        default=0.1)

    parser.add_argument("--clip",
                        type=float,
                        help="Maximum gradient norm for clipping; 0 disables gradient clipping",
                        default=1)


    #============================#
    #      tuner(src/run_ft.py)      #
    #============================#
    parser.add_argument("dataset",
                        type=str,
                        help="Path to formatted dataset for TDA")

    parser.add_argument("model",
                        type=str,
                        help="Model for TDA")


    #============================#
    #       Pipeline / TDA       #
    #============================#
    parser.add_argument("--stage",
                        choices=["all", "train", "tda"],
                        help="Run both stages, training only, or attribution only with verified existing artifacts",
                        default="train")

    parser.add_argument("--method",
                        choices=["rept"],
                        help="Attribution method to run after training or artifact reuse",
                        default="rept")

    parser.add_argument("--device",
                        help="Attribution device, such as cuda:0 or cpu; training still uses automatic placement",
                        default="cuda")

    parser.add_argument("--topk",
                        nargs="+",
                        type=int,
                        help="Retrieval cutoffs for attribution metrics",
                        default=[1, 5, 10, 30, 50, 100, 250, 500, 1000])

    parser.add_argument("--layer",
                        type=int,
                        help="RepT hidden-state layer; -1 selects the last layer",
                        default=-1)

    parser.add_argument("--sources",
                        type=int,
                        help="Total attribution source limit, balanced across labels when possible",
                        default=None)

    parser.add_argument("--queries",
                        type=int,
                        help="Total evaluation query limit, balanced across labels when possible",
                        default=None)


    #============================#
    #       Cache controls       #
    #============================#
    parser.add_argument("--force-train",
                        action="store_true",
                        help="Train again even when matching completed artifacts exist; replaces final artifacts",
                        default=False)

    parser.add_argument("--resume",
                        action="store_true",
                        help="Resume the latest usable checkpoint in output_dir, skipping partial saves; stop if none is usable",
                        default=False)

    
    args = parser.parse_args()

    if args.resume and args.force_train:
        parser.error("--resume and --force-train cannot be used together")
    if args.stage == "tda" and (args.resume or args.force_train):
        parser.error("--stage tda cannot be combined with --resume or --force-train")
    for name in ("r", "alpha", "batch", "accum", "epochs"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be between 0 (inclusive) and 1 (exclusive)")
    if not math.isfinite(args.warmup) or args.warmup < 0 or (
        args.warmup >= 1 and not args.warmup.is_integer()
    ):
        parser.error("--warmup must be a nonnegative fraction below 1 or a whole step count")
    if not math.isfinite(args.clip) or args.clip < 0:
        parser.error("--clip must be a finite nonnegative number")
    if args.layer != -1 and args.layer < 1:
        parser.error("--layer must be -1 or a positive hidden-state layer index")

    for name in ("sources", "queries"):
        limit = getattr(args, name)
        if limit is not None and limit <= 0:
            parser.error(f"--{name} must be positive")
    if any(k <= 0 for k in args.topk):
        parser.error("--topk values must be positive")
    if args.sources is not None and max(args.topk) > args.sources:
        parser.error("--topk cannot exceed --sources")

    query_path = Path(str(Path(args.dataset)) + "_test.csv")
    if args.stage in ("all", "tda") and not query_path.is_file():
        parser.error(f"Evaluation CSV not found: {query_path}; use --stage train for training only")

    checkpoint_path = None
    if args.resume:
        print(f"Looking for the latest training checkpoint in {args.output_dir}...", flush=True)
        try:
            checkpoint_path = get_resume_checkpoint(args.output_dir)
        except ValueError as error:
            parser.error(str(error))

    lora_options = get_lora_options(
        r=args.r,
        alpha=args.alpha,
        dropout=args.dropout,
    )

    training_inputs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup,
        max_grad_norm=args.clip,
    )

    reusable = False
    artifact_dir = Path(f"./lora_adapter/{args.model}/{args.dataset}")
    if args.resume:
        print(f"Resuming training from checkpoint: {checkpoint_path}", flush=True)
    elif args.force_train:
        print("Training requested with --force-train; existing final artifacts may be replaced.", flush=True)
    else:
        print("Checking saved training artifacts and configuration...", flush=True)
        artifact_dir, reusable, reason = check_training_artifacts(
            args.model, args.dataset, lora_options,
            get_training_options(**training_inputs),
        )
        if not reusable:
            print(f"Checking retained checkpoints for completed epoch {args.epochs}...", flush=True)
            epoch_artifacts = find_epoch_artifacts(
                args.output_dir, args.model, args.dataset, lora_options,
                get_training_options(**training_inputs),
            )
            if epoch_artifacts is not None:
                artifact_dir = epoch_artifacts
                reusable = True
        if reusable:
            print(f"Skipping fine-tuning; using existing artifacts: {artifact_dir}", flush=True)
        elif args.stage == "tda":
            parser.error(f"Cannot run TDA with the existing training artifacts: {reason}. "
                         "Use matching training settings or run --stage train/all first.")
        else:
            print(f"Fine-tuning needed: {reason}", flush=True)

    if not reusable:
        token = os.getenv("HUGGING_FACE_TOKEN")
        if token:
            print("Loading Hugging Face authentication and signing in...", flush=True)
            from huggingface_hub import login

            login(token=token)

        print("Loading training libraries and creating LoRA configuration...", flush=True)
        lora_config = get_lora_config(args.r, args.alpha, args.dropout)
        print("Creating training arguments and initializing the compute device...", flush=True)
        training_args = get_training_args(**training_inputs)
        print("Loading training pipeline...", flush=True)
        tuner = get_tuner(args.dataset,args.model,training_args,lora_config)
        print("Loading model/tokenizer, preparing data, and starting fine-tuning...", flush=True)
        trainer = train_tuner(tuner, checkpoint_path=checkpoint_path)
        print("Fine-tuning and artifact saving complete.", flush=True)
        # Release training references before attribution loads the saved model.
        del trainer, tuner
        gc.collect()

    if args.stage in ("all", "tda"):
        metrics = run_attribution(
            artifact_dir, args.dataset, query_path, method=args.method,
            output_dir=args.output_dir,
            device=args.device, topk=args.topk, layer=args.layer,
            source_limit=args.sources, query_limit=args.queries,
        )
        print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()
