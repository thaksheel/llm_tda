import json
import hashlib
from pathlib import Path

def run_attribution(
    artifact_dir, source_path, query_path, method="rept", *,
    output_dir="output", device="cuda", topk=(1, 5, 10), layer=-1,
    source_limit=None, query_limit=None,
):
    """Dispatch attribution calls from the pipeline."""
    if method != "rept":
        raise ValueError("This runner currently supports only 'rept'")
    return run_rept(
        artifact_dir, source_path, query_path, output_dir=output_dir, device=device, topk=topk,
        layer=layer, source_limit=source_limit, query_limit=query_limit,
    )


def run_rept(
    artifact_dir, source_path, query_path, *, output_dir="output", device="cuda",
    topk=(1, 5, 10), layer=-1, source_limit=None, query_limit=None,
):
    """Load saved training artifacts, run RepT, and return exported metrics."""
    method = "rept"
    for name, limit in (("source_limit", source_limit), ("query_limit", query_limit)):
        if limit is not None and (not isinstance(limit, int) or limit <= 0):
            raise ValueError(f"{name} must be a positive integer")
    topk = sorted(set(topk))
    if not topk or any(not isinstance(k, int) or k <= 0 for k in topk):
        raise ValueError("topk must contain positive integers")

    artifact_dir = Path(artifact_dir)
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    if not (artifact_dir / "complete.json").is_file():
        raise ValueError(f"Training artifacts are incomplete: {artifact_dir}")

    print("Loading attribution libraries and prepared datasets...", flush=True)
    import pandas as pd
    import torch
    from datasets import load_from_disk
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from src import TDAEvaluation, Params

    source_dataset = load_from_disk(str(source_path))
    sources = source_dataset.to_pandas()
    queries = pd.read_csv(query_path)
    for name, frame, columns in (
        ("Source", sources, {"prompts", "response", "label"}),
        ("Query", queries, {"prompts", "expected_response", "label"}),
    ):
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"{name} data is missing columns: {sorted(missing)}")
        if frame.empty or frame[list(columns)].isna().any().any():
            raise ValueError(f"{name} data must be nonempty with no missing required values")

    sources = _limit_by_label(sources, source_limit)
    queries = _limit_by_label(queries, query_limit)
    if topk[-1] > len(sources):
        raise ValueError(f"Largest topk ({topk[-1]}) exceeds source count ({len(sources)})")
    unsupported = set(queries["label"]) - set(sources["label"])
    if unsupported:
        raise ValueError(
            f"Query labels have no matching selected sources: {unsupported}. "
            "Increase source_limit or choose a subset with supporting labels."
        )

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA GPU")

    print("Loading saved tokenizer, base model, and LoRA adapter...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir))
    tokenizer.padding_side = metadata.get("padding_side", "left")
    model_options = {}
    if metadata.get("model_revision"):
        model_options["revision"] = metadata["model_revision"]
    model = AutoModelForCausalLM.from_pretrained(metadata["model_name"], **model_options)
    model = PeftModel.from_pretrained(model, str(artifact_dir))
    # Preserve RepT's representation gradients through the base model.
    for parameter in model.parameters():
        parameter.requires_grad = True
    model.config.use_cache = False

    print(f"Running {method}: {len(sources)} sources, {len(queries)} queries...", flush=True)
    evaluation = TDAEvaluation(
        model=model, tokenizer=tokenizer, device=str(target_device),
        params=Params(layer=layer, layer_norm=True, optimizer_state=None),
    )
    source_features, query_features = evaluation.extract_features(
        sources, queries, method
    )
    method_dir = Path(output_dir) / method
    epoch = metadata.get("trained_epochs")
    if epoch is None and (artifact_dir / "trainer_state.json").is_file():
        state = json.loads((artifact_dir / "trainer_state.json").read_text(encoding="utf-8"))
        epoch = state.get("epoch")
    if epoch is not None:
        method_dir = method_dir / f"epoch-{float(epoch):g}"
    identity_files = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin",
                      "tokenizer_config.json", "tokenizer.json", "tokenizer.model",
                      "spiece.model", "vocab.json", "vocab.txt", "merges.txt", "chat_template.jinja"]
    feature_metadata = {
        "feature_version": "rept-first-response-gradient-v2",
        "method": method,
        "layer": layer,
        "model_name": metadata["model_name"],
        "model_revision": metadata.get("model_revision"),
        "artifact_dir": str(artifact_dir.resolve()),
        "training_global_step": metadata.get("global_step"),
        "trained_epochs": epoch,
        "artifact_sha256": {name: _sha256_file(artifact_dir / name)
                            for name in identity_files if (artifact_dir / name).is_file()},
        "source_path": str(Path(source_path).resolve()),
        "source_fingerprint": source_dataset._fingerprint,
        "query_path": str(Path(query_path).resolve()),
        "query_sha256": _sha256_file(query_path),
        "source_limit": source_limit,
        "query_limit": query_limit,
        "selection": "round-robin-label-order-v1",
        "device": str(target_device),
        "padding_side": tokenizer.padding_side,
        "max_length": 512,
    }
    _save_features(method_dir, source_features, query_features, sources, queries, feature_metadata)
    print(f"Saved attribution features: {method_dir}", flush=True)

    results = evaluation.compute_attribution(
        sources, queries, source_features, query_features, topk, "cosine", method,
    )
    metrics = pd.DataFrame([result.__dict__ for result in results])
    metrics.to_excel(method_dir / "metrics.xlsx", index=False)
    print(f"Saved attribution metrics: {method_dir / 'metrics.xlsx'}", flush=True)
    return metrics


def _limit_by_label(frame, limit):
    if limit is None:
        return frame.copy()
    # Take examples round-robin across labels instead of only the first class.
    order = frame.groupby("label", sort=False).cumcount().sort_values(kind="stable").index
    return frame.loc[order[:limit]].copy()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, data):
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(data, indent=2), encoding="utf-8")
    pending.replace(path)


def _save_features(method_dir, source_features, query_features, sources, queries, metadata):
    import numpy as np

    matrices = [np.stack(source_features), np.stack(query_features)]
    if any(matrix.ndim != 2 or not np.isfinite(matrix).all() for matrix in matrices):
        raise ValueError("Attribution features must be finite two-dimensional matrices")
    if matrices[0].shape[1] != matrices[1].shape[1]:
        raise ValueError("Source and query feature dimensions do not match")
    if matrices[0].shape[0] != len(sources) or matrices[1].shape[0] != len(queries):
        raise ValueError("Feature counts do not match selected data rows")

    method_dir.mkdir(parents=True, exist_ok=True)
    complete = method_dir / "features_complete.json"
    complete.unlink(missing_ok=True)
    # Metrics belong to these exact features; discard the previous run's export.
    (method_dir / "metrics.xlsx").unlink(missing_ok=True)
    for name, matrix, frame in zip(("source", "query"), matrices, (sources, queries)):
        feature_path = method_dir / f"{name}_features.npz"
        pending = feature_path.with_name(feature_path.name + ".tmp")
        with pending.open("wb") as stream:
            np.savez_compressed(stream, features=matrix, row_indices=frame.index.to_numpy(dtype=np.int64))
        pending.replace(feature_path)
        # JSON preserves labels without requiring pickle when loading features.
        rows = frame[["label"]].copy()
        rows.insert(0, "row_index", frame.index)
        _write_json(method_dir / f"{name}_rows.json", json.loads(rows.to_json(orient="records")))

    metadata = dict(metadata, source_shape=list(matrices[0].shape),
                    query_shape=list(matrices[1].shape),
                    source_dtype=str(matrices[0].dtype), query_dtype=str(matrices[1].dtype))
    _write_json(method_dir / "feature_metadata.json", metadata)
    _write_json(complete, {"feature_version": metadata["feature_version"]})
