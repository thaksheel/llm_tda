# Preparing datasets

## Convert each corpus once

Run from the repository root with the environment activated. The downloaded raw
Hugging Face datasets must already exist under `data/raw/`.

```bash
python -m src.prepare_ftrace --full --output data/datasets/ftrace-full
python -m src.prepare_trex --full --output data/datasets/trex-full
```

FTRACE reads `abstracts/` and `queries/`. T-REx reads `queries/` and all 20
`sentences/shard_00` through `shard_19` directories; its separate `facts/` download
is not required because query records already contain the fact definitions.

Conversion streams sources to disk, preserving all rows, labels, and provenance.
It does not load the entire corpus into pandas, use a model, or train. Token
length checks happen later during pipeline subset selection. Full T-REx
conversion processes about 19.6 million sentences and can take substantial time.
Existing output paths are never overwritten.

If T-REx sources already exist but the companion query CSV is missing, recover
only the queries without converting the sources again:

```bash
python -m src.prepare_trex --queries-only --output data/datasets/trex-full
```

Each conversion writes:

- `DATASET/`: prepared sources saved with `save_to_disk()`.
- `DATASET_test.csv`: all query prompts and expected answers.
- `DATASET_preparation.json`: row counts and content checksums.

Content-derived source fingerprints and normalized CSV line endings make these
identities independent of the conversion machine or output location.

## Select a seeded subset in the pipeline

```bash
python run_pipeline.py /mnt/y/output/trex-1000-seed42 \
  ./data/datasets/trex-full \
  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --epochs 1 --stage all --method rept trackstar \
  --sources 1000 --queries 100 --seed 42 --topk 1 5 10
```

Use `data/datasets/ftrace-full` in the same way. Selection happens before training:
`--sources` controls both training and attribution, and `--queries` controls
evaluation. Queries are sampled across relations or scalar labels. Each selected
query has at least one selected source with a matching complete fact or label.
Remaining source slots are seeded distractors. Sentences supporting several facts
remain single source examples.

With a source limit, selected source examples must fit 512 tokens including chat
formatting and the answer. Selected queries must also fit. Invalid or unsupported
query candidates are skipped; if the requested count cannot be supplied, selection
stops with an error. With no query limit, all supported candidate queries that fit
the source budget are kept. With no source limit, all source rows are used and
retain the training tokenizer's normal truncation behavior. Omit both limits to
use an existing prepared dataset unchanged, including its full query CSV.

The output contains `data/KEY/sources/`, `sources_test.csv`, and `selection.json`.
The manifest records the seed, original row indices, source IDs when available,
supporting row indices, corpus/query identity, and tokenizer revision. Repeating
the same command reuses this selection. Another machine can regenerate it from
the same prepared corpus, tokenizer version, seed, and limits. This controls data
selection, not bitwise reproducibility of model training across hardware.

Use a new output directory when changing the selection. Resume requires the
original selection; incompatible or unverifiable checkpoint selections are
rejected. Existing prepared small datasets remain usable without conversion.
The preparation scripts still support their earlier small-subset mode when
`--full` is omitted, but pipeline selection is the recommended workflow.

## Formats and relevance

FTRACE prompts ask the model to fill `[blank]`; `<extra_id_0>` is removed from the
answer. Facts use `subject|predicate|object` IDs. Raw `facts` fields store
predicate/object/subject order; only entity pairs linked to the masked example
by `example_uris` are retained. Merged links from several pages are preserved.

T-REx sources use `Write a factual sentence.` with the complete sentence as the
response. All annotated `fact_triples` become labels. Queries use the supplied
`prompt0` and first `entity1` name as the expected response; aliases remain
metadata. Unannotated sentences can be distractors with an empty label list.

Sources store fact IDs as a JSON array in `label`; queries normally store one fact
ID. Relevance means nonempty fact intersection. Existing scalar labels such as
`clean` and `harmful` retain equality-based matching. Matching labels indicate
annotated support, rather than measured causal influence. These formats adapt
the datasets to this chat fine-tuning pipeline; they do not reproduce the original
pretraining experiments.
