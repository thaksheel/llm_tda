# Preparing FTRACE

Run from the repository root with the environment activated. The downloaded data
must already exist under `data/raw/ftrace/abstracts/` and `queries/` as Hugging Face
datasets saved with `save_to_disk()`.

```bash
python -m src.prepare_ftrace
```

Defaults: 1,000 sources, 100 distinct query facts, seed 42, and the
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` tokenizer. The script scans the source corpus
to find supporting examples. It does not download the dataset or train a model.

Outputs:

- `data/datasets/ftrace-small/`: prepared training sources.
- `data/datasets/ftrace-small_test.csv`: query prompts and expected answers.
- `data/datasets/ftrace-small_preparation.json`: selection settings, raw indices,
  fingerprints, and the number of selected supporting sources per query.

Existing output paths are not overwritten. To prepare a different subset:

```bash
python -m src.prepare_ftrace \
  --raw data/raw/ftrace \
  --output data/datasets/ftrace-2000 \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --sources 2000 --queries 200 --seed 42
```

Use `--sources >= --queries > 0`. The preparation tokenizer must match the pipeline
model. Examples that would exceed 512 tokens, including chat formatting and the
answer, are excluded rather than truncated.

## Format and relevance

Each prompt asks the model to fill `[blank]` and return only the missing text.
FTRACE's `<extra_id_0>` marker is removed from the target answer. Source records
retain `source_id`, `page_uri`, `masked_uri`, and `raw_row_index` for inspection.

Fact IDs use `subject|predicate|object`, such as `Q668|P47|Q902`. The raw
[`ekinakyurek/ftrace` dataset](https://huggingface.co/datasets/ekinakyurek/ftrace)
stores `facts` in predicate/object/subject order. Preparation keeps only fact pairs
linked to the masked example by `example_uris`; both subject- and object-masked
examples can support the same fact.
Some rows combine links from several pages/entities for identical masked text;
all those links are retained even when they differ from the representative
`masked_uri` or `page_uri` fields.

Sources store all matching fact IDs as a JSON array in `label`; each query stores
one fact ID in `label`. A source is relevant when its fact set contains the query
fact. Sources are not duplicated once per fact. Existing scalar labels such as
`clean` and `harmful` retain their equality-based matching.

Queries are sampled across relations, with one query per distinct fact. Every
selected query has at least one selected supporting source. The remaining source
slots come from a seeded reservoir sample, so they may include additional
supporting examples as well as distractors. This is a supported test subset, not
a reproduction of the original FTRACE training or evaluation protocol.

## Test the pipeline

```bash
python run_pipeline.py /mnt/y/output/ftrace-small \
  ./data/datasets/ftrace-small \
  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --epochs 1 --stage all --method rept --topk 1 5 10
```

Use a separate output directory for this training run. Further `--sources` or
`--queries` limits can remove supporting examples; use the prepared subset as-is
for the initial test. See [RUN_PIPELINE.md](RUN_PIPELINE.md) for other methods.
