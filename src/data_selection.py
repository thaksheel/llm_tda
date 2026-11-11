"""Deterministic, support-preserving selection before training and attribution."""

from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import tempfile

from .relevance import label_set
from .prepare_ftrace import within_context


VERSION = "supported-selection-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def query_order(frame, seed):
    groups = defaultdict(list)
    for index, row in frame.iterrows():
        labels = sorted(label_set(row["label"]))
        fact = labels[0].split("|") if len(labels) == 1 else []
        relation = fact[1] if len(fact) == 3 else json.dumps(labels)
        groups[relation].append(int(index))
    keys = sorted(groups, key=lambda key: digest([seed, "relation", key]))
    queues = {key: deque(sorted(groups[key], key=lambda i: digest([seed, "query", i]))) for key in keys}
    while any(queues.values()):
        for key in keys:
            if queues[key]:
                yield queues[key].popleft()


def random_indices(size, seed, count):
    """Hash-based sampling avoids platform-dependent RNG or set iteration order."""
    swaps, remaining = {}, size
    for counter in range(min(size, count)):
        slot = int(digest([seed, "source", counter]), 16) % remaining
        index = swaps.get(slot, slot)
        remaining -= 1
        swaps[slot] = swaps.get(remaining, remaining)
        swaps.pop(remaining, None)
        yield index


def select_data(source_path, query_path, output_dir, model, *, sources, queries, seed=42):
    import pandas as pd
    from datasets import load_from_disk
    from tqdm import tqdm

    source = load_from_disk(str(source_path))
    required = {"prompts", "response", "label"}
    if not required <= set(source.column_names) or not len(source):
        raise ValueError("Selection requires nonempty prepared sources with prompts, response, and label")
    query_frame = pd.read_csv(query_path)
    if not {"prompts", "expected_response", "label"} <= set(query_frame.columns) or query_frame.empty:
        raise ValueError("Selection requires a nonempty prepared query CSV")
    if query_frame[["prompts", "expected_response", "label"]].isna().any().any():
        raise ValueError("Queries contain missing prompts, responses, or labels")
    if sources is not None and not 0 < sources <= len(source):
        raise ValueError(f"--sources must be between 1 and {len(source)}")
    if queries is not None and not 0 < queries <= len(query_frame):
        raise ValueError(f"--queries must be between 1 and {len(query_frame)}")
    config = {"version": VERSION, "source_fingerprint": source._fingerprint,
              "query_sha256": file_digest(query_path), "sources": sources, "queries": queries,
              "seed": seed, "model": model, "max_length": 512}
    key = digest(config)
    parent = Path(output_dir) / "data"
    destination = parent / key
    selected_source, selected_query = destination / "sources", destination / "sources_test.csv"
    if destination.exists():
        saved = json.loads((destination / "selection.json").read_text(encoding="utf-8"))
        cached = load_from_disk(str(selected_source))
        checks = {"configuration": saved["config"] == config,
                  "fingerprint": cached._fingerprint == saved["selected_fingerprint"],
                  "source count": len(cached) == saved["source_count"],
                  "query checksum": file_digest(selected_query) == saved["selected_query_sha256"]}
        if not all(checks.values()):
            raise ValueError("Saved selection is inconsistent: " + ", ".join(k for k, ok in checks.items() if not ok))
        print(f"Reusing seeded data selection: {saved['source_count']} sources, {saved['query_count']} queries", flush=True)
        return str(selected_source), selected_query
    print(f"Selecting data with seed {seed}; loading tokenizer...", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    candidates = []
    for index in query_order(query_frame, seed):
        row = query_frame.iloc[index]
        if within_context(tokenizer, row["prompts"], row["expected_response"]):
            candidates.append(index)
    wanted = set().union(*(label_set(query_frame.iloc[i]["label"]) for i in candidates))
    supporting, validity = {}, {}

    def valid(index):
        if index not in validity:
            row = source[index]
            validity[index] = bool(row["prompts"]) and bool(row["response"]) and within_context(
                tokenizer, row["prompts"], row["response"])
        return validity[index]

    offset = 0
    with tqdm(total=len(source), desc="Checking source fact support") as progress:
        for batch in source.select_columns(["label"]).iter(batch_size=8192):
            for value in batch["label"]:
                missing = (label_set(value) & wanted) - supporting.keys()
                if missing and valid(offset):
                    for label in sorted(missing):
                        supporting[label] = offset
                offset += 1
            progress.update(len(batch["label"]))
    selected, selected_set, selected_queries = [], set(), []
    for index in candidates:
        supports = sorted({supporting[label] for label in label_set(query_frame.iloc[index]["label"])
                           if label in supporting})
        if not supports:
            continue
        existing = next((i for i in supports if i in selected_set), None)
        support = supports[0] if existing is None else existing
        if support not in selected_set:
            if sources is not None and len(selected) == sources:
                continue
            selected.append(support)
            selected_set.add(support)
        selected_queries.append(index)
        if queries is not None and len(selected_queries) == queries:
            break
    if not selected_queries or (queries is not None and len(selected_queries) != queries):
        raise ValueError("Too few supported queries fit the selection; increase --sources or reduce --queries")
    if sources is None:
        source_indices = range(len(source))
    else:
        for index in random_indices(len(source), seed, len(source)):
            if len(selected) == sources:
                break
            if index not in selected_set and valid(index):
                selected.append(index)
                selected_set.add(index)
        if len(selected) != sources:
            raise ValueError("Too few sources fit the tokenizer context")
        source_indices = sorted(selected, key=lambda index: digest([seed, "source-order", index]))
    fingerprint = digest([VERSION, config, list(source_indices) if sources is not None else "all",
                          selected_queries])
    subset = source.select(source_indices, new_fingerprint=fingerprint)
    selected_frame = query_frame.iloc[selected_queries].copy()
    manifest = {"config": config, "selected_fingerprint": fingerprint,
        "source_count": len(subset), "query_count": len(selected_frame),
        "source_indices": list(source_indices) if sources is not None else "all",
        "query_indices": selected_queries,
        "source_ids": list(subset["source_id"]) if sources is not None and "source_id" in subset.column_names else None,
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "support_source_indices": [min(supporting[label] for label in label_set(query_frame.iloc[i]["label"])
            if label in supporting and (sources is None or supporting[label] in selected_set)) for i in selected_queries]}
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="selection-", dir=parent) as temporary:
        temporary = Path(temporary)
        subset.save_to_disk(str(temporary / "sources"))
        manifest["selected_fingerprint"] = load_from_disk(str(temporary / "sources"))._fingerprint
        selected_frame.to_csv(temporary / "sources_test.csv", index=False, lineterminator="\n")
        manifest["selected_query_sha256"] = file_digest(temporary / "sources_test.csv")
        (temporary / "selection.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(destination)
    print(f"Saved seeded selection: {len(subset)} sources, {len(selected_frame)} queries", flush=True)
    return str(selected_source), selected_query


def check_resume_selection(source_path, checkpoint):
    from datasets import load_from_disk
    path = Path(checkpoint) / "metadata.json"
    if not path.is_file():
        path = Path(checkpoint) / "tracin_lr.json"
    if not path.is_file() or json.loads(path.read_text(encoding="utf-8")).get("source_fingerprint") != load_from_disk(str(source_path))._fingerprint:
        raise ValueError("Checkpoint data selection differs or cannot be verified; use the original selection settings")
