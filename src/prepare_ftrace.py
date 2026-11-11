"""Prepare FTRACE for causal masked-entity fine-tuning."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re


MASK = "<extra_id_0>"
INSTRUCTION = "Fill in [blank] in the passage below. Reply with only the missing text.\n\n"


def fact_id(subject, predicate, obj):
    if not (re.fullmatch(r"Q\d+", subject) and re.fullmatch(r"P\d+", predicate)
            and re.fullmatch(r"Q\d+", obj)):
        raise ValueError(f"Invalid Wikidata fact: {subject}, {predicate}, {obj}")
    return f"{subject}|{predicate}|{obj}"


def source_facts(row):
    """FTRACE stores facts as predicate,object,subject; URI suffix 0 masks object."""
    pairs = set()
    for uri in row["example_uris"].split(";"):
        if not uri:
            continue
        parts = uri.rsplit("-", 4)
        if len(parts) != 5 or parts[-1] not in ("0", "1"):
            raise ValueError(f"Invalid FTRACE example URI: {uri}")
        _, _, obj, subject, role = parts
        # Identical masked texts can carry links from several pages/entities;
        # masked_uri is only a representative, not an exhaustive link filter.
        pairs.add((subject, obj))
    result = set()
    for value in row["facts"].split(";"):
        if not value:
            continue
        parts = value.split(",")
        if len(parts) != 3:
            raise ValueError(f"Invalid FTRACE fact: {value}")
        predicate, obj, subject = parts
        canonical = fact_id(subject, predicate, obj)
        if (subject, obj) in pairs:
            result.add(canonical)
    return sorted(result)


def text_pair(row):
    source, target = row["inputs_pretokenized"].strip(), row["targets_pretokenized"].strip()
    if source.count(MASK) != 1 or not target.startswith(MASK):
        raise ValueError("FTRACE examples must have one input mask and a mask-prefixed answer")
    answer = target[len(MASK):].strip()
    if not answer or re.search(r"<extra_id_\d+>", answer):
        raise ValueError("FTRACE answer is empty or contains another mask")
    return INSTRUCTION + source.replace(MASK, "[blank]"), answer


def within_context(tokenizer, prompt, answer, maximum=512):
    from .utils import tokenize
    sample = {"prompts": prompt, "response": answer}
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template([
            {"role": "user", "content": prompt}, {"role": "assistant", "content": answer},
        ], tokenize=False)
    else:
        text = "[INST] " + prompt + " [/INST]" + answer + tokenizer.eos_token
    if len(tokenizer(text, truncation=False, padding=False)["input_ids"]) > maximum:
        return False
    labels = tokenize(sample, tokenizer, maximum)["labels"]
    return any(value != -100 for value in labels[1:])


def balanced_queries(rows, seed):
    """Cycle through shuffled relations, retaining only one query per complete fact."""
    rng, groups, seen = random.Random(seed), defaultdict(list), set()
    for number, row in enumerate(rows):
        key = fact_id(row["sub_uri"], row["predicate_id"], row["obj_uri"])
        if key not in seen:
            seen.add(key)
            groups[row["predicate_id"]].append((number, key))
    for values in groups.values():
        rng.shuffle(values)
    keys = sorted(groups)
    rng.shuffle(keys)
    result = []
    while any(groups.values()):
        for key in keys:
            if groups[key]:
                result.append(groups[key].pop())
    return result


def prepare(raw_dir, output, model, sources=1000, queries=100, seed=42):
    import pandas as pd
    from datasets import Dataset, load_from_disk
    from tqdm import tqdm
    from transformers import AutoTokenizer

    if sources < queries or queries <= 0:
        raise ValueError("Use sources >= queries > 0 for a supported test subset")
    raw_dir, output = Path(raw_dir), Path(output)
    query_path = output.with_name(output.name + "_test.csv")
    manifest_path = output.with_name(output.name + "_preparation.json")
    if any(path.exists() for path in (output, query_path, manifest_path)):
        raise FileExistsError("Prepared output already exists; choose a new --output")
    abstracts = load_from_disk(str(raw_dir / "abstracts"))
    query_data = load_from_disk(str(raw_dir / "queries"))
    tokenizer = AutoTokenizer.from_pretrained(model)
    candidates = []
    for number, key in balanced_queries(query_data, seed):
        prompt, answer = text_pair(query_data[number])
        if within_context(tokenizer, prompt, answer):
            candidates.append((number, key))
        if len(candidates) == queries * 10:
            break
    if len(candidates) < queries:
        raise ValueError("Too few distinct queries fit the model context")
    wanted, supporting, validity = {key for _, key in candidates}, {}, {}
    rng, reservoir = random.Random(seed), []
    reservoir_size = sources * 3
    def valid_source(number):
        if number not in validity:
            prompt, answer = text_pair(abstracts[number])
            validity[number] = within_context(tokenizer, prompt, answer)
        return validity[number]
    columns = abstracts.select_columns(["facts", "example_uris", "masked_uri"])
    offset = 0
    with tqdm(total=len(abstracts), desc="Scanning FTRACE fact links") as progress:
        for batch in columns.iter(batch_size=2048):
            for row in zip(batch["facts"], batch["example_uris"], batch["masked_uri"], strict=True):
                if len(reservoir) < reservoir_size:
                    reservoir.append(offset)
                else:
                    slot = rng.randrange(offset + 1)
                    if slot < reservoir_size:
                        reservoir[slot] = offset
                matches = wanted.intersection(source_facts(dict(zip(("facts", "example_uris", "masked_uri"), row))))
                missing = matches - supporting.keys()
                if missing and valid_source(offset):
                    for key in missing:
                        supporting[key] = offset
                offset += 1
            progress.update(len(batch["facts"]))
    selected_queries = [(number, key) for number, key in candidates if key in supporting][:queries]
    if len(selected_queries) != queries:
        raise ValueError(f"Only {len(selected_queries)} candidate queries have usable supporting sources")
    selected = list(dict.fromkeys(supporting[key] for _, key in selected_queries))
    selected_set = set(selected)
    for number in reservoir:
        if len(selected) == sources:
            break
        if number not in selected_set and valid_source(number):
            selected.append(number)
            selected_set.add(number)
    if len(selected) != sources:
        raise ValueError("Too few sampled sources fit the context; request a smaller subset")
    rng.shuffle(selected)
    source_rows, query_rows = [], []
    for number in selected:
        row = abstracts[number]
        prompt, answer = text_pair(row)
        source_rows.append({"prompts": prompt, "response": answer,
            "label": json.dumps(source_facts(row), separators=(",", ":")),
            "source_id": row["id"], "raw_row_index": number,
            "page_uri": row["page_uri"], "masked_uri": row["masked_uri"]})
    supports = []
    for number, key in selected_queries:
        row = query_data[number]
        prompt, answer = text_pair(row)
        query_rows.append({"prompts": prompt, "expected_response": answer, "label": key,
            "query_id": row["uuid"], "raw_row_index": number,
            "subject_id": row["sub_uri"], "predicate_id": row["predicate_id"], "object_id": row["obj_uri"]})
        count = sum(key in json.loads(source["label"]) for source in source_rows)
        if count == 0:
            raise ValueError(f"Selected query lost its supporting source: {key}")
        supports.append(count)
    output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(source_rows).save_to_disk(str(output))
    pd.DataFrame(query_rows).to_csv(query_path, index=False)
    manifest = {"version": "ftrace-masked-facts-v1", "seed": seed, "model": model,
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"), "max_length": 512,
        "source_count": sources, "query_count": queries, "query_candidate_count": len(candidates),
        "raw_source_fingerprint": abstracts._fingerprint, "raw_query_fingerprint": query_data._fingerprint,
        "selection": "relation-round-robin queries; one guaranteed support each; reservoir distractors",
        "relevance": "complete subject|predicate|object from facts restricted to example_uris pairs",
        "support_counts": supports, "source_indices": selected,
        "query_indices": [number for number, _ in selected_queries]}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {sources} sources and {queries} supported queries: {output}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, help="Downloaded FTRACE directory", default=Path("data/raw/ftrace"))
    parser.add_argument("--output", type=Path, help="New prepared dataset path", default=Path("data/datasets/ftrace-small"))
    parser.add_argument("--model", help="Tokenizer used for context-length validation", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--sources", type=int, help="Prepared source count", default=1000)
    parser.add_argument("--queries", type=int, help="Distinct supported query facts", default=100)
    parser.add_argument("--seed", type=int, help="Deterministic subset seed", default=42)
    parser.add_argument("--full", action="store_true", help="Convert the whole corpus; select subsets in run_pipeline.py", default=False)
    args = parser.parse_args()
    try:
        if args.full:
            from .prepare_full import prepare_full
            prepare_full(args.raw, args.output, "ftrace")
        else:
            prepare(args.raw, args.output, args.model, args.sources, args.queries, args.seed)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
