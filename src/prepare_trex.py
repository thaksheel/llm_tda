"""Prepare T-REx sentences and fact queries for the TDA pipeline."""

import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import random

from .prepare_ftrace import fact_id, within_context


SOURCE_PROMPT = "Write a factual sentence."


def canonical_fact(row):
    return fact_id(*(row[key].rsplit("/", 1)[-1] for key in
                     ("entity0_uri", "predicate_uri", "entity1_uri")))


def source_facts(row):
    return sorted({canonical_fact(triple) for triple in row["fact_triples"]})


def query_pair(row):
    prompt = row["prompt0"].strip()
    answers = row["entity1"]
    if not prompt or not answers or not answers[0].strip():
        raise ValueError("T-REx query requires prompt0 and an object name")
    return prompt, answers[0].strip()


def full_query_record(row, index):
    prompt, answer = query_pair(row)
    return {"prompts": prompt, "expected_response": answer, "label": canonical_fact(row),
            "query_id": row["fact_id"], "raw_row_index": index,
            "subject_id": row["entity0_uri"], "predicate_id": row["predicate_uri"],
            "object_id": row["entity1_uri"], "answer_aliases": json.dumps(row["entity1_alias"])}


def export_full_queries(raw, output):
    """Recover a missing query CSV without reading or rewriting source shards."""
    import csv
    import tempfile
    from datasets import load_from_disk

    raw, output = Path(raw), Path(output)
    if not (output / "state.json").is_file():
        raise FileNotFoundError(f"Prepared source dataset not found: {output}")
    query_path = output.with_name(output.name + "_test.csv")
    if query_path.exists():
        raise FileExistsError(f"Query CSV already exists: {query_path}")
    count = 0
    with tempfile.TemporaryDirectory(prefix="queries-", dir=output.parent) as temporary:
        pending = Path(temporary) / "queries.csv"
        with pending.open("w", encoding="utf-8", newline="") as stream:
            writer = None
            for index, row in enumerate(load_from_disk(str(raw / "queries"))):
                record = full_query_record(row, index)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(record), lineterminator="\n")
                    writer.writeheader()
                writer.writerow(record)
                count += 1
        if not count:
            raise ValueError("Raw T-REx query dataset is empty")
        pending.replace(query_path)
    print(f"Prepared {count} full T-REx queries: {query_path}", flush=True)
    return query_path


def balanced_queries(rows, seed):
    rng, groups, seen = random.Random(seed), defaultdict(list), set()
    for number, row in enumerate(rows):
        key = canonical_fact(row)
        if key not in seen:
            groups[row["predicate_uri"]].append((number, key))
            seen.add(key)
    for group in groups.values():
        rng.shuffle(group)
    relations = sorted(groups)
    rng.shuffle(relations)
    while any(groups.values()):
        for relation in relations:
            if groups[relation]:
                yield groups[relation].pop()


def sentence_locations(shards, wanted):
    """Scan only sentence IDs; read text and triples only for matching rows."""
    import pyarrow as pa
    import pyarrow.compute as pc
    from tqdm import tqdm

    values = pa.array(sorted(wanted), type=pa.string())
    for shard_index, shard in enumerate(tqdm(shards, desc="Finding T-REx supporting sentences")):
        if not len(shard):
            continue
        matches = pc.is_in(shard.data.column("sentence_id"), value_set=values)
        for row_index in pc.indices_nonzero(matches).to_pylist():
            yield shard_index, row_index


def prepare(raw_dir, output, model, sources=1000, queries=100, seed=42):
    import pandas as pd
    from datasets import Dataset, load_from_disk
    from transformers import AutoTokenizer

    if sources < queries or queries <= 0:
        raise ValueError("Use sources >= queries > 0 for a supported test subset")
    raw_dir, output = Path(raw_dir), Path(output)
    query_path = output.with_name(output.name + "_test.csv")
    manifest_path = output.with_name(output.name + "_preparation.json")
    if any(path.exists() for path in (output, query_path, manifest_path)):
        raise FileExistsError("Prepared output already exists; choose a new --output")
    shard_paths = sorted((raw_dir / "sentences").glob("shard_*"))
    if len(shard_paths) != 20:
        raise ValueError("Expected all 20 downloaded T-REx sentence shards")
    shards = [load_from_disk(str(path)) for path in shard_paths]
    query_data = load_from_disk(str(raw_dir / "queries"))
    tokenizer = AutoTokenizer.from_pretrained(model)
    candidates = []
    for number, key in balanced_queries(query_data, seed):
        prompt, answer = query_pair(query_data[number])
        if within_context(tokenizer, prompt, answer):
            candidates.append((number, key))
        if len(candidates) == queries * 10:
            break
    if len(candidates) < queries:
        raise ValueError("Too few distinct queries fit the model context")
    wanted = {key for _, key in candidates}
    sentence_ids = {link["sentence_id"] for number, _ in candidates
                    for link in query_data[number]["trex_sentences"]}
    supporting, valid = {}, {}

    def valid_source(location):
        if location not in valid:
            row = shards[location[0]][location[1]]
            text = row["text"].strip()
            valid[location] = bool(text) and within_context(tokenizer, SOURCE_PROMPT, text)
        return valid[location]

    for location in sentence_locations(shards, sentence_ids):
        row = shards[location[0]][location[1]]
        missing = wanted.intersection(source_facts(row)) - supporting.keys()
        if missing and valid_source(location):
            for key in missing:
                supporting[key] = location
    selected_queries = [(number, key) for number, key in candidates if key in supporting][:queries]
    if len(selected_queries) != queries:
        raise ValueError(f"Only {len(selected_queries)} candidate queries have usable supporting sentences")
    selected = list(dict.fromkeys(supporting[key] for _, key in selected_queries))
    selected_set = set(selected)
    rng = random.Random(seed)
    total = sum(len(shard) for shard in shards)
    ends, count = [], 0
    for shard in shards:
        count += len(shard)
        ends.append(count)
    for index in rng.sample(range(total), min(total, sources * 3)):
        if len(selected) == sources:
            break
        shard_index = bisect_right(ends, index)
        location = (shard_index, index - (ends[shard_index - 1] if shard_index else 0))
        if location not in selected_set and valid_source(location):
            selected.append(location)
            selected_set.add(location)
    if len(selected) != sources:
        raise ValueError("Too few sampled sentences fit the context; request a smaller subset")
    rng.shuffle(selected)
    source_rows, query_rows = [], []
    for shard_index, row_index in selected:
        row = shards[shard_index][row_index]
        source_rows.append({"prompts": SOURCE_PROMPT, "response": row["text"].strip(),
            "label": json.dumps(source_facts(row), separators=(",", ":")),
            "source_id": row["sentence_id"], "abstract_uri": row["abstract_uri"],
            "raw_shard": shard_paths[shard_index].name, "raw_row_index": row_index})
    supports = []
    for number, key in selected_queries:
        row = query_data[number]
        prompt, answer = query_pair(row)
        query_rows.append({"prompts": prompt, "expected_response": answer, "label": key,
            "query_id": row["fact_id"], "raw_row_index": number,
            "subject_id": row["entity0_uri"], "predicate_id": row["predicate_uri"],
            "object_id": row["entity1_uri"], "answer_aliases": json.dumps(row["entity1_alias"])})
        count = sum(key in json.loads(source["label"]) for source in source_rows)
        if not count:
            raise ValueError(f"Selected query lost its supporting sentence: {key}")
        supports.append(count)
    output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(source_rows).save_to_disk(str(output))
    pd.DataFrame(query_rows).to_csv(query_path, index=False)
    manifest = {"version": "trex-sentences-v1", "model": model, "seed": seed,
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"), "max_length": 512,
        "source_count": sources, "query_count": queries, "raw_source_count": total,
        "raw_query_fingerprint": query_data._fingerprint,
        "raw_shards": {path.name: shard._fingerprint for path, shard in zip(shard_paths, shards, strict=True)},
        "selection": "relation-round-robin queries; one guaranteed support each; uniform sampled distractors",
        "source_format": "generic instruction with complete sentence as response",
        "query_format": "prompt0 with first entity1 name as expected response",
        "relevance": "complete subject|predicate|object from sentence fact_triples",
        "support_counts": supports, "source_locations": selected,
        "query_indices": [number for number, _ in selected_queries]}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {sources} sources and {queries} supported queries: {output}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, help="Downloaded T-REx directory", default=Path("data/raw/trex"))
    parser.add_argument("--output", type=Path, help="New prepared dataset path", default=Path("data/datasets/trex-small"))
    parser.add_argument("--model", help="Tokenizer used for context-length validation", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--sources", type=int, help="Prepared source count", default=1000)
    parser.add_argument("--queries", type=int, help="Distinct supported query facts", default=100)
    parser.add_argument("--seed", type=int, help="Deterministic subset seed", default=42)
    parser.add_argument("--full", action="store_true", help="Convert the whole corpus; select subsets in run_pipeline.py", default=False)
    parser.add_argument("--queries-only", action="store_true",
                        help="Write the missing full query CSV for an existing prepared source dataset")
    args = parser.parse_args()
    if args.queries_only:
        try:
            export_full_queries(args.raw, args.output)
        except (ValueError, FileExistsError, FileNotFoundError) as error:
            parser.error(str(error))
        return
    mode = "corpus" if args.full else "subset"
    print(f"Preparing T-REx {mode}; loading dataset libraries...", flush=True)
    try:
        if args.full:
            from .prepare_full import prepare_full
            prepare_full(args.raw, args.output, "trex")
        else:
            prepare(args.raw, args.output, args.model, args.sources, args.queries, args.seed)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
