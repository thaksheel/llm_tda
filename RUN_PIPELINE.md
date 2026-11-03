# Running the pipeline

Run from the repository root with your WSL environment activated:

```bash
cd /mnt/d/GitHub/TDA
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py --help
```

## Inputs

```bash
python run_pipeline.py OUTPUT_DIR DATASET MODEL [options]
```

| Argument | Meaning |
| --- | --- |
| `OUTPUT_DIR` | Training checkpoints, attribution caches, and results; for example `/mnt/y/output/save-test`. |
| `DATASET` | Prepared Hugging Face dataset saved with `save_to_disk()`; for example `./data/datasets/harmful-tuning`. |
| `MODEL` | Hugging Face base-model identifier or local model directory; for example `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. |

Training data needs `prompts` and `response`; attribution also needs `label`.
Queries must be in `<dataset>_test.csv`, with `prompts`, `expected_response`, and
`label`. For this example, use `data/datasets/harmful-tuning_test.csv`.
Selected query labels must have matching selected source labels. Prepare raw
downloaded datasets before running the pipeline.

For the downloaded FTRACE data, see [DATA_PREPARATION.md](DATA_PREPARATION.md).
Sources can use JSON arrays of fact IDs in `label`; queries match any shared fact.
Scalar labels continue to match by equality.

## Commands

Train or reuse one epoch, then run a small RepT test:

```bash
python run_pipeline.py /mnt/y/output/save-test \
  ./data/datasets/harmful-tuning \
  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --epochs 1 --stage all --method rept \
  --sources 20 --queries 4 --topk 1 5 10
```

Run full attribution using an existing completed epoch:

```bash
python run_pipeline.py /mnt/y/output/save-test \
  ./data/datasets/harmful-tuning \
  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --epochs 2 --stage tda --method trackstar
```

Replace `--method` to select another method. Omit `--sources` and `--queries` to
use all rows. For small tests, set `--topk` values within the selected source count.

Resume training toward three total epochs:

```bash
python run_pipeline.py /mnt/y/output/save-test \
  ./data/datasets/harmful-tuning \
  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --epochs 3 --resume
```

## Flags

| Flag | Default | Use |
| --- | --- | --- |
| `--stage` | `train` | `train`: training only; `tda`: attribution only; `all`: both. |
| `--method` | `rept` | `rept`, `tracin`, `less`, `rapid_in`, `lorif`, `trak`, `logra`, or `trackstar`. |
| `--device` | `cuda` | Attribution device, such as `cuda:0` or `cpu`. Training uses automatic placement; 4-bit gradient extraction requires CUDA. |
| `--epochs` | `3` | Total training target or completed epoch to analyze. See the method table below for checkpoint selection. |
| `--sources` | All | Attribution source limit, balanced across labels when possible. |
| `--queries` | All | Attribution query limit, balanced across labels when possible. |
| `--topk` | `1 5 10 30 50 100 250 500 1000` | Retrieval cutoffs; none may exceed the selected source count. |
| `--layer` | `-1` | RepT hidden-state layer; `-1` means last. Applies only to RepT. |
| `--r` | `8` | LoRA rank. |
| `--alpha` | `16` | LoRA scaling factor. |
| `--dropout` | `0.1` | LoRA dropout probability. |
| `--batch` | `1` | Training examples per device per batch. |
| `--accum` | `16` | Batches accumulated per optimizer update. |
| `--warmup` | `0.1` | Warmup fraction below 1, or a whole optimizer-step count. |
| `--clip` | `1` | Gradient clipping threshold; `0` disables clipping. |
| `--resume` | Off | Resume the latest usable training checkpoint in `OUTPUT_DIR`; fail if none exists. |
| `--force-train` | Off | Train again even when matching final artifacts exist. |

`--sources` and `--queries` limit attribution, not training.

## Outputs and cache reuse

Final results are under `OUTPUT_DIR/METHOD/epoch-N/`, including `metrics.xlsx` and
saved features or scores. Shared raw adapter gradients are under
`OUTPUT_DIR/gradients/`. Method-specific caches are under `OUTPUT_DIR/METHOD/`.

| Method | Checkpoints | Cached work and interruption recovery |
| --- | --- | --- |
| RepT | Selected epoch | Saves representation/gradient features; recomputes them on each run. |
| TracIn | Epochs 1 through selected epoch | Shared raw gradients resume by row; completed checkpoint scores are reused. Unfinished checkpoint scoring restarts. |
| LESS | Epochs 1 through selected epoch | Reuses shared raw gradients; `features/` projection resumes by batch. Completed `checkpoint_scores/` are reused. Requires saved AdamW moments. |
| RapidIn | Selected epoch | Reuses shared raw gradients; `features/` compression resumes by batch. Completed `checkpoint_scores/` are reused. |
| LoRIF | Selected epoch | Separate `factors/` extraction resumes after each 128-row shard; `scores/` resumes after each layer. |
| TRAK | Selected epoch | Separate `features/` extraction resumes by batch; completed `kernels/` and `scores/` are reused. An unfinished kernel solve restarts. |
| LoGRA | Selected epoch | Separate `features/` extraction resumes after each 128-row shard; `scores/` resumes after each layer. |
| TrackStar | Selected epoch | Reuses shared raw gradients; `features/` projection resumes by batch and whitening by block. Completed `checkpoint_scores/` are reused. Requires saved AdamW moments. |

Repeat the same command to reuse compatible caches. TrackStar whitening depends
on the selected queries, so changing queries creates a separate feature cache.
LoRIF, TRAK, and LoGRA cannot substitute the shared adapter gradients for their
own features. Do not run two writers against the same output directory at once.

## Training reuse and resume

- Matching completed training artifacts are reused automatically. Keep the model,
  dataset, LoRA options, and training settings consistent.
- `--stage tda` never trains. It stops if compatible artifacts are unavailable.
  TracIn and LESS also need the earlier completed epoch checkpoints in `OUTPUT_DIR`.
- `--resume` uses a total epoch target, not additional epochs. It cannot be combined
  with `--force-train`; neither flag works with `--stage tda`.
- Completed epoch checkpoints are retained. Recovery saves occur around each
  quarter epoch; intermediate saves are removed after successful training/export.

See [Changes.md](Changes.md) for method formulas and adaptations.
