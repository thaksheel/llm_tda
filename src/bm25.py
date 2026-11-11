"""CPU lexical baseline with positive-IDF BM25 and reproducible ranked exports.

Sources and queries both use prompt + response, matching attribution's access to
the expected query response. Labels are used only for evaluation, never scoring.
"""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re


def tokenize(text):
    """Unicode word tokens, case-folded; no stemming or stopword removal."""
    return re.findall(r"\w+", text.casefold())


class BM25:
    def __init__(self, documents, k1=1.2, b=0.75):
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 requires finite k1 > 0 and 0 <= b <= 1")
        self.k1, self.b = k1, b
        self.lengths, self.postings = [], defaultdict(list)
        for index, document in enumerate(documents):
            counts = Counter(tokenize(document))
            self.lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings[term].append((index, frequency))
        if not self.lengths:
            raise ValueError("BM25 requires at least one source document")
        self.average_length = sum(self.lengths) / len(self.lengths)

    def score(self, query):
        """One score per source; distinct query terms contribute once each."""
        scores = [0.0] * len(self.lengths)
        if not self.average_length:
            return scores
        for term in sorted(set(tokenize(query))):
            postings = self.postings.get(term, ())
            idf = math.log1p((len(self.lengths) - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, frequency in postings:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
                scores[index] += idf * frequency * (self.k1 + 1) / (frequency + norm)
        return scores


def run_bm25(source_path, query_path, *, output_dir="output", topk=(1, 5, 10),
             source_limit=None, query_limit=None):
    import numpy as np
    import pandas as pd
    from datasets import load_from_disk

    from .relevance import label_set, missing_query_labels
    from .run_tda import _limit_by_label, _sha256_file, _write_json
    from .tda_evaluation import TDAEvaluation

    dataset = load_from_disk(str(source_path))
    sources, queries = dataset.to_pandas(), pd.read_csv(query_path)
    for name, frame, response, limit in (
        ("Source", sources, "response", source_limit),
        ("Query", queries, "expected_response", query_limit),
    ):
        required = ["prompts", response, "label"]
        if not set(required) <= set(frame.columns) or frame.empty or frame[required].isna().any().any():
            raise ValueError(f"{name} data must be nonempty with prompts, {response}, and label and no missing values")
        if not all(isinstance(value, str) for column in ("prompts", response) for value in frame[column]):
            raise ValueError(f"{name} prompts and responses must be strings")
        if limit is not None and (not isinstance(limit, int) or limit <= 0):
            raise ValueError(f"{name} limit must be a positive integer")
    sources, queries = _limit_by_label(sources, source_limit), _limit_by_label(queries, query_limit)
    topk = sorted(set(topk))
    if not topk or any(not isinstance(k, int) or k <= 0 or k > len(sources) for k in topk):
        raise ValueError("Retrieval cutoffs must be positive integers within source count")
    if missing_query_labels(sources, queries):
        raise ValueError("Query labels have no matching selected sources")

    index = BM25(sources["prompts"] + "\n" + sources["response"])
    scores = np.array([index.score(text) for text in queries["prompts"] + "\n" + queries["expected_response"]])
    results = TDAEvaluation.compute_from_scores(sources, queries, scores, topk, "bm25")
    metrics = pd.DataFrame([result.__dict__ for result in results])
    directory = Path(output_dir) / "bm25"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "complete.json").unlink(missing_ok=True)
    pending = directory / "scores.npz.tmp"
    with pending.open("wb") as stream:
        np.savez_compressed(stream, scores=scores, source_indices=sources.index.to_numpy(dtype=np.int64),
                            query_indices=queries.index.to_numpy(dtype=np.int64))
    pending.replace(directory / "scores.npz")
    for kind, frame in (("source", sources), ("query", queries)):
        rows = frame.copy()
        rows.insert(0, "row_index", frame.index)
        _write_json(directory / f"{kind}_rows.json", json.loads(rows.to_json(orient="records")))
    ranked = []
    for query_position, row in enumerate(scores):
        query_labels = label_set(queries.iloc[query_position]["label"])
        # Match the existing evaluator's descending numpy argsort, including ties.
        for rank, source_position in enumerate(np.argsort(row)[::-1][:topk[-1]], 1):
            ranked.append(dict(query_index=int(queries.index[query_position]),
                               source_index=int(sources.index[source_position]), rank=rank,
                               score=float(row[source_position]),
                               relevant=bool(query_labels & label_set(sources.iloc[source_position]["label"]))))
    pd.DataFrame(ranked).to_csv(directory / "rankings.csv", index=False)
    metrics.to_excel(directory / "metrics.xlsx", index=False)
    version = "bm25-positive-idf-v1"
    _write_json(directory / "feature_metadata.json", dict(
        feature_version=version, method="bm25", k1=index.k1, b=index.b,
        tokenizer="unicode-word-casefold-v1", text_fields="prompt + newline + response",
        query_response="expected_response", query_terms="distinct", tie_order="numpy-argsort-descending",
        source_path=str(Path(source_path).resolve()), source_fingerprint=dataset._fingerprint,
        query_path=str(Path(query_path).resolve()), query_sha256=_sha256_file(query_path),
        source_limit=source_limit, query_limit=query_limit, score_shape=list(scores.shape), topk=topk,
    ))
    _write_json(directory / "complete.json", {"feature_version": version})
    print(f"Saved BM25 scores, rankings, rows, and metrics: {directory}", flush=True)
    return metrics
