# Running the pipeline

This guide describes the current `run_pipeline.py`, with configuration helpers
in `src/run_ft.py` and training/artifact saving in `src/finetune.py`.
Keep this guide in sync when flags, defaults, paths, or execution behavior change.
Significant corrections and compatibility changes are recorded in [Changes.md](Changes.md).

## Repository structure

```text
run_pipeline.py             # Command-line entry point
src/
  __init__.py               # Public exports loaded on demand
  run_ft.py               # Training configuration, reuse, and resume checks
  finetune.py               # Tuner, checkpoint callbacks, and artifact saving
  run_tda.py            # Attribution execution and feature/metric exports
  tda.py                    # Attribution feature calculations
  tda_evaluation.py         # Feature extraction loops and ranking metrics
  utils.py                  # Shared types and training tokenization
  load_data.py              # Raw dataset downloader
tests/                      # CPU regression tests
modules/                    # Bundled method implementations
```

Run commands from the repository root. The controller imports the training and
attribution helpers from `src`; these modules do not define separate command-line
parsers. Public package exports load their dependencies only when accessed.
Download raw datasets separately with `python -m src.load_data`; the pipeline
expects prepared training/query data.

## Current behavior

The controller parses arguments, creates the LoRA and training configurations,
loads the model and formatted training dataset, trains, and saves artifacts.

By default it runs **fine-tuning only** (`--stage train`). With `--stage all`, it
runs RepT after training or after reusing completed training artifacts.
`--stage tda` requires matching completed artifacts and runs attribution only;
it stops with an error if the artifacts are missing, incomplete, or incompatible.
Before loading model weights, the
controller checks whether completed training artifacts match the model,
prepared dataset fingerprint, LoRA settings, and training settings. Matching
artifacts skip training. Missing/incomplete artifacts or changed inputs trigger
training, with a printed reason. Changing only the output directory, logging,
or checkpoint-retention settings does not require retraining.

Startup messages are flushed immediately before artifact checks, dataset
fingerprint validation, training-library loading, device setup, and training.
Configuration checks use standard-library code; PEFT, Transformers training
objects, `src`, and Hugging Face login are deferred until training is needed.
Attribution imports its dependencies only when that stage runs.
Validating an otherwise complete cached run still imports `datasets` and reads
the dataset fingerprint; that library may import additional dependencies.
`--help` does not import the training libraries or authenticate.

The pipeline runs RepT attribution and saves features and metrics for each
selected epoch. Checkpoint resume is explicit through `--resume`. Final training
artifacts use the same model/dataset directory; retraining may overwrite them.

## Usage in WSL

Run from the repository root with your Python environment activated:

```bash
cd /mnt/d/GitHub/llm_tda
# Activate whichever environment you created, for example:
source .venv/bin/activate

python run_pipeline.py \
  output \
  ./data/datasets/harmful-tuning \
  TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

There are three required positional arguments, in this order:

| Argument | Meaning | Example |
| --- | --- | --- |
| `output_dir` | Trainer checkpoint directory and root for method-specific TDA outputs. Does not set the final adapter directory. | `output` |
| `dataset` | Path to a formatted Hugging Face dataset saved with `save_to_disk()`. Use a repository-relative path to preserve the existing artifact layout. | `./data/datasets/harmful-tuning` |
| `model` | Hugging Face base-model identifier or compatible local model directory. | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |

The input must be a single training `Dataset`, rather than a `DatasetDict`,
with string columns `prompts` and `response`. The attribution evaluator later
also needs `label`. Raw FTRACE/T-REx downloads require preparation first.
The training stage does not load an evaluation CSV. With `--stage all` or `--stage tda`, the
controller checks that `<dataset>_test.csv` exists before training.

Display all arguments without starting training:

```bash
python run_pipeline.py --help
```

## LoRA flags

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--r` | Integer | `8` | Adapter rank. Higher ranks add trainable parameters and increase memory use. |
| `--alpha` | Integer | `16` | LoRA scaling factor; standard adapter scaling is `alpha / r`. |
| `--dropout` | Float | `0.1` | LoRA dropout probability, from zero to less than one. |

## Training flags

Flags use short names, with hyphens for multiword names. The longer training-library parameter names are still used internally.

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--batch` | Integer | `1` | Examples processed on each device per forward/backward pass. Larger batches use more GPU memory. |
| `--accum` | Integer | `16` | Batches accumulated before an optimizer update. On one GPU, effective batch size is batch size multiplied by this value. |
| `--epochs` | Integer | `3` | Training target, or the completed epoch to select from saved checkpoints for attribution. |
| `--warmup` | Float | `0.1` | Warmup fraction below one (`0.1` means 10%), or a whole optimizer-step count such as `100`, with a Transformers version supporting this API. |
| `--clip` | Float | `1` | Gradient clipping threshold. Zero disables clipping. |

For the defaults and one GPU, effective batch size is `1 * 16 = 16` examples.
Training metrics are logged automatically at each 5% boundary of the full
training run (5%, 10%, ..., 100%), rounded to the next optimizer update after
gradient accumulation. Resumed runs continue at the next progress boundary.
Runs shorter than 20 optimizer updates log once per update because finer
intervals are not possible. Attribution has its own progress bars.

Checkpoint timing and retention are automatic. Recovery saves occur around
25%, 50%, and 75% of each epoch, with
a permanent completed-epoch checkpoint at 100% (four save points total). Saves
are rounded to optimizer update boundaries, so very short epochs can have fewer
than four distinct save points.

During training, the latest intermediate recovery checkpoint is retained and
older intermediates are removed only after a newer save succeeds. All completed
epoch checkpoints are protected. After training and final artifact export both
succeed, remaining intermediate checkpoints are removed, leaving completed epochs.
Exceptions and interruptions skip this final cleanup so recovery remains possible.
Built-in Trainer rotation is disabled. Unrelated directories and unknown/malformed
checkpoint states are retained rather than blindly deleted.

Example with one training epoch and automatic 5% reporting:

```bash
python run_pipeline.py \
  output/save-test \
  ./data/datasets/harmful-tuning \
  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --epochs 1
```

One epoch processes the full training dataset. `--sources` and `--queries`
limit attribution only.

## Cache controls

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--force-train` | Boolean switch | `False` | Bypass the reuse check and train again. Replaces final artifacts at the existing model/dataset location. |
| `--resume` | Boolean switch | `False` | Resume the highest numbered usable `checkpoint-N` inside `output_dir`. Skips partial/invalid newer saves and stops before training if none are usable. Bypasses completed-artifact reuse. Cannot be combined with `--force-train`. |

Neither `--resume` nor `--force-train` can be combined with `--stage tda`, which
never starts training. TDA-only still checks the model, dataset fingerprint,
LoRA configuration, and training settings. Supply the same `--r` and other
training settings. `--epochs` may select an earlier completed epoch from the
same run, even when the original total training target was larger.

Repeat a command to reuse matching final artifacts. If those do not match the
requested epoch, the controller searches `output_dir/checkpoint-N` for that
completed epoch, verifies its artifacts/configuration/dataset, and uses it without
training. This applies to all stages. With `--stage tda`, an unavailable epoch
causes an error; `train`/`all` can start training when no compatible artifacts
exist. `--force-train` and `--resume` explicitly bypass this automatic selection.

To resume instead of starting from the base model:

```bash
python run_pipeline.py \
  output/save-test \
  ./data/datasets/harmful-tuning \
  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --epochs 3 \
  --resume
```

Checkpoints are tried in descending numeric step order, not modification time.
A usable checkpoint must include adapter configuration/weights, Trainer state
with a matching step, optimizer state, and scheduler state. Partial or invalid
newer saves are reported and skipped, allowing recovery if a crash interrupted
a save. If no usable checkpoint remains, the command stops instead of starting
fresh. Tensor compatibility/corruption is checked later by Trainer during loading.
After new epoch saves, checkpoints also contain attribution metadata, tokenizer,
parameter mapping, and a completion marker, so they can be used directly by TDA.

Use the same model, dataset, LoRA setup, and optimizer as the checkpoint.
`--epochs` is the **total target**, not additional epochs: resuming one completed
epoch with `--epochs 3` targets three epochs overall. Resuming an already finished
target may run no further updates and save final artifacts again. Increasing the
target of a completed cosine-scheduled run changes the schedule and is not
equivalent to choosing that larger target from the start.

To deliberately train again:

```bash
python run_pipeline.py \
  out-save-test \
  ./data/datasets/harmful-tuning \
  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --epochs 1 \
  --force-train
```

Reuse requires adapter weights, tokenizer configuration and vocabulary,
optimizer state/mapping, configuration/metadata files, and consistent completion
step counts. The check validates JSON metadata and dataset identity, but does
allow empty optimizer parameter groups (normal for some LoRA runs), provided
at least one group contains parameters. It does
not deserialize optimizer/weight tensors to check for corruption or contact the
Hub to detect changes to an unpinned model revision. Preserve the same base-model
revision for reproducible comparisons. Dataset fingerprints are Hugging Face
dataset identities, not a byte-by-byte integrity check of every data file.

## Selecting a saved epoch

After training to epoch 3, use the existing `--epochs` argument to analyze epoch 1:

```bash
python run_pipeline.py \
  output/save-test \
  ./data/datasets/harmful-tuning \
  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --epochs 1 --stage tda --sources 20 --queries 4 --topk 1 5 10
```

The selected checkpoint must include per-epoch metadata and a completion
marker. The final adapter folder contains the latest completed training export.

A checkpoint after epoch 1 of a three-epoch cosine-scheduled run is not the same
as a separate one-epoch run. The original total training target remains recorded
in its training configuration; epoch selection deliberately allows that target
to differ while validating the other settings and actual completed epoch.

## Settings fixed in code

| Setting | Current value | Purpose |
| --- | --- | --- |
| LoRA task | `CAUSAL_LM` | Prompt/response next-token training. |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` | Attention and feed-forward projections receiving adapters. Other model architectures must expose these module names or need a code change. |
| Optimizer | `adamw_torch` | Standard AdamW optimization with saved optimizer state. |
| Learning-rate scheduler | `cosine` | Decreases learning rate after warmup. |
| FP16 | Enabled | Mixed-precision training. Requires compatible hardware. |
| Gradient checkpointing | Enabled | Reduces memory use by recomputing activations. |
| Base-model loading | 4-bit, `device_map="auto"` | Quantized training-model loading and automatic placement. `--device` selects the attribution device only. |
| Maximum tokenized length | `512` | Truncation limit for training examples. |
| Padding | Left; pad token equals EOS token | Tokenizer configuration used during training. |

## Authentication

If `HUGGING_FACE_TOKEN` is set, the controller calls Hugging Face login with that
token only when training is needed, after argument parsing and artifact checks.
Otherwise it uses existing Hugging Face
credentials where available. TinyLlama is public; gated models require approved
access and appropriate credentials. Never put tokens in committed configuration.

## Saved artifacts

Trainer checkpoints are saved under the positional `output_dir`.
Final artifacts are saved separately under:

```text
lora_adapter/<model>/<dataset>/
```

For the example command, this resolves to:

```text
lora_adapter/TinyLlama/TinyLlama-1.1B-Chat-v1.0/data/datasets/harmful-tuning/
```

| Artifact | Purpose |
| --- | --- |
| Adapter weights and `adapter_config.json` | Reload the trained LoRA adapter on its base model. |
| Tokenizer files | Reload the tokenizer and saved special-token configuration. |
| `optimizer.pt` | Full optimizer state, including moments, step counters, and parameter groups. |
| `optimizer_parameters.json` | Saved optimizer IDs mapped to ordered model parameter names and shapes, grouped as in the optimizer. |
| `training_args.json` | Training configuration. |
| `trainer_state.json` | Trainer progress and logged state. |
| `metadata.json` | Model revision when available, dataset fingerprints, tokenization settings, and final step count. |
| `complete.json` | Written last to indicate the save routine completed. Not a configuration/cache compatibility check by itself. |

Training checkpoints include the saved state used to resume interrupted
training. The adapter, tokenizer, and metadata identify the model used for attribution.

## Attribution functions and pipeline flags

`src/run_tda.py` is a function module with no argument parser or command-line entry
point. `run_attribution(...)` dispatches to `run_rept(...)`. Importing the module
loads no ML libraries, model, or datasets. All command-line options belong to
`run_pipeline.py`.

Run training/reuse followed by a small RepT test:

```bash
python run_pipeline.py \
  output/save-test \
  ./data/datasets/harmful-tuning \
  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --epochs 1 --stage all --method rept --sources 20 --queries 4 --topk 1 5 10
```

It loads the saved tokenizer, base model/revision from `metadata.json`, and
adapter. RepT does not load `optimizer.pt`. Limited subsets take examples
round-robin across labels; limits are total counts, not counts per class.
Each selected query label must have a selected supporting source label.
Required values must be present, and the largest `topk` cannot exceed source count.

| Controller flag | Default | Meaning |
| --- | --- | --- |
| `--stage` | `train` | `train` runs/reuses training; `all` continues to attribution; `tda` requires verified existing artifacts and runs attribution without training. |
| `--method` | `rept` | Runs representation-gradient attribution using RepT. |
| `--device` | `cuda` | Attribution device such as `cuda:0` or `cpu`; training retains automatic placement. |
| `--topk` | `1 5 10 30 50 100 250 500 1000` | One or more retrieval cutoffs. For a small subset, pass smaller cutoffs explicitly. |
| `--layer` | `-1` | RepT hidden-state layer; last layer by default. |
| `--sources` | All | Maximum total sources; balanced across labels when limited. |
| `--queries` | All | Maximum total queries; balanced across labels when limited. |

TDA files are written to `<output_dir>/<method>/epoch-<N>/` using the actual
selected epoch, so testing an earlier checkpoint does not overwrite a later
epoch's outputs. For `output/save-test` with RepT at epoch 2, the layout is:

```text
output/save-test/rept/epoch-2/
  source_features.npz
  query_features.npz
  source_rows.json
  query_rows.json
  feature_metadata.json
  features_complete.json
  metrics.xlsx
```

The compressed NumPy files contain a `features` matrix (one row per selected
example) and `row_indices` pointing to the original source dataset or query CSV.
Load them with `numpy.load(path, allow_pickle=False)`. Row JSON files preserve
the matching original row indices and labels in feature order, including after
balanced subset selection.

Feature metadata identifies the method/version, layer, base model/revision,
training step, adapter/tokenizer file hashes, source fingerprint, query-file
hash, selection limits, matrix shapes/dtypes, and extraction settings. Files
are replaced atomically, and `features_complete.json` is written last after all
feature files and metadata are saved. Features save before metric calculation,
so a later metrics/export failure does not discard the computed vectors.
The previous metrics export is removed when replacement features begin saving,
so a failed run cannot leave old metrics alongside the new features.

Metrics are also returned as a DataFrame. Each attribution run computes
features and metrics and replaces the files for the selected method and epoch.

For attribution only, use the same command with `--stage tda`:

```bash
python run_pipeline.py \
  output/save-test \
  ./data/datasets/harmful-tuning \
  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --epochs 1 --stage tda --method rept --sources 20 --queries 4 --topk 1 5 10
```

`output_dir` remains required in TDA-only mode and controls where attribution
features/metrics are saved, and its retained epoch checkpoints can be selected
automatically for attribution. TDA-only never resumes optimizer training.

RepT's sample signature concatenates the last prompt-token hidden vector with
the full gradient vector for the first response token (paper section 3.2:
https://arxiv.org/html/2510.02334v1#S3.SS2). The gradient extractor preserves its
token-by-hidden-dimension shape, yielding a fixed signature length of twice the
hidden dimension. It uses evaluation mode with autograd enabled to disable
dropout during feature extraction. `--layer` selects the representation layer
used for sample-level attribution.

## Maintaining this guide

Run the CPU regression suite from the repository root:

```bash
python -B -m unittest discover -s tests
```

The suite covers controller behavior, checkpoint selection and retention,
feature exports, and RepT gradients using a small CPU model. Runner tests use
mocked model loading; they do not replace a CUDA smoke test in the training
environment.

When changing `run_pipeline.py`, `src/run_ft.py`, or artifact saving, update the
corresponding flag rows, fixed settings, examples, and artifact descriptions here.
Describe current behavior only. Record significant mathematical corrections,
result-compatibility changes, and recovery/data-loss fixes in `Changes.md`.
Routine refactors and flag wording changes do not need changelog entries.
