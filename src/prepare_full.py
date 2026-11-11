"""Stream raw FTRACE or T-REx into a complete, model-independent prepared corpus."""

import hashlib
import json
from pathlib import Path
import tempfile

from .data_selection import file_digest


def _source_rows(raw, kind, stats):
    from datasets import load_from_disk
    from tqdm import tqdm
    from . import prepare_ftrace as ftrace, prepare_trex as trex
    raw = Path(raw)
    if kind == "ftrace":
        paths = [raw / "abstracts"]
    else:
        paths = [raw / "sentences" / f"shard_{i:02d}" for i in range(20)]
    checksum, count = hashlib.sha256(), 0
    for path in paths:
        dataset = load_from_disk(str(path))
        for index, row in enumerate(tqdm(dataset, desc=f"Converting {kind}: {path.name}")):
            if kind == "ftrace":
                prompt, answer = ftrace.text_pair(row)
                record = {"prompts": prompt, "response": answer,
                    "label": json.dumps(ftrace.source_facts(row), separators=(",", ":")),
                    "source_id": row["id"], "raw_row_index": index,
                    "page_uri": row["page_uri"], "masked_uri": row["masked_uri"]}
            else:
                record = {"prompts": trex.SOURCE_PROMPT, "response": row["text"].strip(),
                    "label": json.dumps(trex.source_facts(row), separators=(",", ":")),
                    "source_id": row["sentence_id"], "raw_row_index": index,
                    "abstract_uri": row["abstract_uri"], "raw_shard": path.name}
            checksum.update(json.dumps(record, sort_keys=True, ensure_ascii=False,
                                       separators=(",", ":")).encode() + b"\n")
            count += 1
            yield record
    Path(stats).write_text(json.dumps({"source_count": count, "source_sha256": checksum.hexdigest()}), encoding="utf-8")


def prepare_full(raw, output, kind):
    import pandas as pd
    from datasets import Dataset, load_from_disk
    from . import prepare_ftrace as ftrace, prepare_trex as trex
    raw, output = Path(raw), Path(output)
    query_path = output.with_name(output.name + "_test.csv")
    manifest_path = output.with_name(output.name + "_preparation.json")
    if any(path.exists() for path in (output, query_path, manifest_path)):
        raise FileExistsError("Prepared output already exists; choose a new --output")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="conversion-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        stats = temporary / "stats.json"
        data = Dataset.from_generator(_source_rows,
            gen_kwargs={"raw": str(raw), "kind": kind, "stats": str(stats)},
            cache_dir=str(temporary / "cache"))
        manifest = json.loads(stats.read_text(encoding="utf-8"))
        data = data.select(range(len(data)), new_fingerprint=manifest["source_sha256"])
        data.save_to_disk(str(temporary / "sources"))
        rows = []
        for index, row in enumerate(load_from_disk(str(raw / "queries"))):
            if kind == "ftrace":
                prompt, answer = ftrace.text_pair(row)
                key = ftrace.fact_id(row["sub_uri"], row["predicate_id"], row["obj_uri"])
                query = {"prompts": prompt, "expected_response": answer, "label": key,
                    "query_id": row["uuid"], "raw_row_index": index,
                    "subject_id": row["sub_uri"], "predicate_id": row["predicate_id"], "object_id": row["obj_uri"]}
            else:
                query = trex.full_query_record(row, index)
            rows.append(query)
        pd.DataFrame(rows).to_csv(temporary / "queries.csv", index=False, lineterminator="\n")
        manifest.update(version="full-corpus-v1", kind=kind, query_count=len(rows),
            query_sha256=file_digest(temporary / "queries.csv"), context_filter="pipeline selection")
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (temporary / "sources").replace(output)
        (temporary / "queries.csv").replace(query_path)
        (temporary / "manifest.json").replace(manifest_path)
    print(f"Converted {len(data)} sources and {len(rows)} queries: {output}", flush=True)
    return manifest
