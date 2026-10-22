# Running the pipeline

Run from the repository root with your WSL environment activated:

```bash
cd /mnt/d/GitHub/llm_tda
source .venv/bin/activate
python run_pipeline.py --help
```

## Inputs

```bash
python run_pipeline.py OUTPUT_DIR DATASET MODEL [options]
```

| Argument | Example | Meaning |
| --- | --- | --- |
| `OUTPUT_DIR` | `output/save-test` | Training checkpoints and attribution outputs. |
| `DATASET` | `./data/datasets/harmful-tuning` | Prepared Hugging Face dataset saved with `save_to_disk()`. |
| `MODEL` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Base-model identifier or compatible local directory. |

Training data needs `prompts` and `response`; attribution also needs `label`.
Queries must be in `<dataset>_test.csv`, with `prompts`, `expected_response`, and
`label`. For this example: `data/datasets/harmful-tuning_test.csv`.
Raw downloaded datasets must be prepared first.

## Common commands

Train/reuse one epoch, then run a small RepT test:

```bash
python run_pipeline.py output/save-test \
  ./data/datasets/harmful-tuning \
  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --epochs 1 --stage all --method rept \
  --sources 20 --queries 4 --topk 1 5 10
```

Use `--method tracin` for the same small TracIn test.
For training only, use `--stage train` (the default).

Run full TracIn attribution using two existing completed epochs:

```bash
python run_pipeline.py output/save-test \
  ./data/datasets/harmful-tuning \
  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --epochs 2 --stage tda --method tracin
```

Resume training toward three total epochs:

```bash
python run_pipeline.py output/save-test \
  ./data/datasets/harmful-tuning \
  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --epochs 3 --resume
```

## Flags

| Flag | Default | Use |
| --- | --- | --- |
| `--stage` | `train` | `train`: training only; `tda`: attribution only; `all`: both. |
| `--method` | `rept` | Attribution method: `rept` or `tracin`. |
| `--device` | `cuda` | Attribution device, e.g. `cuda:0` or `cpu`. TracIn scoring uses it too. Training uses automatic placement; 4-bit gradient extraction requires CUDA. |
| `--epochs` | `3` | Total training target or completed epoch to analyze. RepT uses that epoch; TracIn combines epochs 1 through this value. |
| `--sources` | All | Total attribution source limit, balanced across labels when possible. |
| `--queries` | All | Total attribution query limit, balanced across labels when possible. |
| `--topk` | `1 5 10 30 50 100 250 500 1000` | Retrieval cutoffs; none may exceed the selected source count. |
| `--layer` | `-1` | RepT hidden-state layer; `-1` means last. Does not affect TracIn. |
| `--r` | `8` | LoRA rank. |
| `--alpha` | `16` | LoRA scaling factor. |
| `--dropout` | `0.1` | LoRA dropout probability. |
| `--batch` | `1` | Training examples per device per batch. |
| `--accum` | `16` | Batches accumulated per optimizer update. |
| `--warmup` | `0.1` | Warmup fraction below 1, or a whole optimizer-step count. |
| `--clip` | `1` | Gradient clipping threshold; `0` disables clipping. |
| `--resume` | Off | Resume training from the latest usable checkpoint in `OUTPUT_DIR`; fail if none exists. |
| `--force-train` | Off | Train again even when matching final artifacts exist. |

`--sources` and `--queries` limit attribution, not training. For small tests,
set smaller `--topk` values explicitly. Selected query labels must have matching
selected source labels.

## Reuse and resume

- Repeat a command to reuse matching training artifacts. Keep the model, dataset,
  LoRA options, and training settings consistent.
- `--stage tda` never trains. It stops if the requested compatible artifacts or
  epochs are unavailable. TracIn needs the earlier completed epoch checkpoints
  under `OUTPUT_DIR` as well as the selected final epoch.
- `--resume` is for training and uses a total epoch target, not additional epochs.
  It cannot be combined with `--force-train`; neither flag works with `--stage tda`.
- Completed epoch checkpoints are retained. Recovery saves occur around each
  quarter epoch; intermediate saves are removed after successful training/export.
- To resume TracIn, stop the old process and rerun the same command. Compatible
  gradients and completed checkpoint scores are reused automatically. Partial
  extraction resumes from saved rows; unfinished checkpoint scoring restarts.
  Older partial gradient files without saved row progress must be recomputed.
- Gradients and checkpoint scores are shared across epoch selections. After a
  two-epoch run, `--epochs 1` reuses epoch 1 without extracting or scoring again.
- RepT features are recomputed on each attribution run. Do not run two writers
  against the same output directory at once.

## Saved files and moving drives

```text
OUTPUT_DIR/
  checkpoint-N/                   # Saved training epochs/recovery points
  gradients/CACHE_ID/              # Raw gradients, row mappings, metadata, resume files
  rept/epoch-N/                   # RepT features, row mappings, metadata, metrics.xlsx
  tracin/checkpoint_scores/SCORE_ID/ # Completed weighted checkpoint scores
  tracin/epoch-N/
    scores.npz                    # Combined query-by-source scores
    metrics.xlsx
```

Final training artifacts are stored separately at `lora_adapter/<model>/<dataset>/`.
For the example: `lora_adapter/TinyLlama/TinyLlama-1.1B-Chat-v1.0/data/datasets/harmful-tuning/`.

Full TracIn gradients for the example dataset/model need about **110 GiB per
checkpoint**. Ensure the output drive has room. To move outputs, wait until the
run stops, move the entire directory containing checkpoints and method results,
then use its new location as `OUTPUT_DIR`. For example, moving `output` to
`Y:\output` makes the argument `/mnt/y/output/save-test`. Continue launching
from the repo so relative dataset/adapter paths resolve.

Optional lossless-compression sample test (original gradients are unchanged):

```bash
python -m src.check_compression \
  output/save-test/gradients/CACHE_ID/query_features.npy
```

Replace `CACHE_ID` with the saved cache directory name. This saves eight sampled
rows to a separate gzip file and verifies exact recovery.
Use `--rows`, `--level`, or `--output` to customize the test. Gzip samples are not
loaded by the pipeline.

Method corrections and numerical choices: [Changes.md](Changes.md).
