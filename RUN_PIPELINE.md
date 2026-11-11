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

For the downloaded FTRACE and T-REx data, see [DATA_PREPARATION.md](DATA_PREPARATION.md).
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

Select several methods with a space-separated list:

```bash
python run_pipeline.py /mnt/y/output/ftrace-small-compact \
  ./data/datasets/ftrace-small \
  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --epochs 1 --stage tda --method rept lorif trak logra trackstar
```

When `--sources` or `--queries` is supplied, the pipeline first saves a seeded
selection under `OUTPUT_DIR/data/KEY/`. The selected sources are used for both
training and attribution. `selection.json` records the exact original rows and
support links. Repeat the same corpus, tokenizer, seed, and limits to reproduce
it elsewhere. Use a new output directory for a different subset. Existing runs
can use their original prepared dataset with both limits omitted.

Methods run sequentially in the listed order; duplicates run once. Use
`--method all` by itself to run all eight attribution methods and BM25. Each method uses its own
output folder and compatible caches. RepT features are recomputed. GPU cache
memory is released between methods. A failure stops the run and preserves
completed outputs.

`--stage all` trains or reuses training once before running the selected methods.
`--stage train` remains training-only. Omit `--sources` and `--queries` to use all
rows. For small tests, keep `--topk` within the selected source count. Add `--seed 42`
to make the selection explicit.

The example above avoids creating shared raw-gradient files. TracIn, LESS, and
RapidIn still create those files when no compatible raw cache exists.

For later download and reuse, keep the whole output directory, the prepared
source dataset and query CSV, and the matching training artifacts. Final training
artifacts may be under `lora_adapter/`, outside the output directory; retain that
matching adapter directory or compatible completed epoch checkpoints too. A new
`--stage tda` output directory does not copy training artifacts into it. Selected
source/query data and its manifest are included under `data/`. Keep the
same model, data, and training settings when restoring the run.

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
| `--method` | `rept` | `rept`, `tracin`, `less`, `rapid_in`, `lorif`, `trak`, `logra`, `trackstar`, `bm25`; list one or more in execution order, or use `all` alone. |
| `--device` | `cuda` | Attribution device, such as `cuda:0` or `cpu`. Training uses automatic placement; 4-bit gradient extraction requires CUDA. |
| `--epochs` | `3` | Total training target or completed epoch to analyze. See the method table below for checkpoint selection. |
| `--sources` | All | Seeded source count for both training and attribution, preserving query support. |
| `--queries` | All supported during selection | Seeded supported query count, balanced across relations or labels. |
| `--seed` | `42` | Data selection seed; used when `--sources` or `--queries` is set. |
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
| TrackStar | Selected epoch | Projects missing gradients directly without saving raw gradients; reuses existing compatible raw rows; `features/` projection resumes by batch and whitening by block. Completed `checkpoint_scores/` are reused. Requires saved AdamW moments. |

TrackStar stores projected, whitened, and normalized features under `trackstar/features/`.
It does not create or expand the shared raw-gradient cache. Existing raw caches are
left unchanged. Other gradient methods still create raw caches when needed.

Repeat the same command to reuse compatible caches. TrackStar whitening depends
on the selected queries, so changing queries creates a separate feature cache.
LoRIF, TRAK, and LoGRA cannot substitute the shared adapter gradients for their
own features. Do not run two writers against the same output directory at once.

## Training reuse and resume

- Matching completed training artifacts are reused automatically. Keep the selected data, seed, model,
  dataset, LoRA options, and training settings consistent.
- `--stage tda` never trains. It stops if compatible artifacts are unavailable.
  TracIn and LESS also need the earlier completed epoch checkpoints in `OUTPUT_DIR`.
- `--resume` uses a total epoch target, not additional epochs. It cannot be combined
  with `--force-train`; neither flag works with `--stage tda`.
- Completed epoch checkpoints are retained. Recovery saves occur around each
  quarter epoch; intermediate saves are removed after successful training/export.

See [Changes.md](Changes.md) for method formulas and adaptations.

## BM25 lexical baseline

Run BM25 on the exact prepared sources and queries used by an attribution run:

```bash
python run_pipeline.py /mnt/y/output/ftrace-small-compact \
  ./data/datasets/ftrace-small \
  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --stage tda --method bm25 --topk 1 5 10
```

BM25 alone with `--stage tda` or `--stage all` skips training and checkpoint
validation and runs on CPU. The positional model argument is still required;
it is used only if `--sources` or `--queries` requests tokenizer-aware data
selection. For a previous selected run, point DATASET at its saved
`OUTPUT_DIR/data/KEY/sources` and omit the selection flags to reuse those exact rows.

Each training prompt plus response is a document. Each query prompt plus
expected response is the search text, matching the expected-response access
of the attribution methods. Labels never enter scoring. This is lexical
retrieval, not a measurement of causal training influence or answer accuracy.

The implementation uses positive IDF `log(1 + (N - df + 0.5)/(df + 0.5))`,
`k1=1.2`, and `b=0.75`. Unicode word tokens are case-folded, with no stemming
or stopword removal; distinct query tokens contribute once. Full prepared
texts are searched without model-token truncation. Higher scores rank first;
ties follow the existing evaluator's descending NumPy argsort.

Outputs are under `OUTPUT_DIR/bm25/`: `metrics.xlsx` (the shared auPRC,
precision@k, recall@k, and MRR evaluation), `scores.npz` (full query-by-source
scores and row indices), `rankings.csv` (up to the largest requested k per
query, with scores and label relevance), `source_rows.json`, `query_rows.json`
(including text), and metadata and completion files. BM25 is independent of
model epochs. Raw BM25 scores are not calibrated against attribution scores;
compare relevance metrics and rankings instead.
