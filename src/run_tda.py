import json
import hashlib
import math
from pathlib import Path
from .gradient_cache import cache_key, gradient_directory, score_directory
from .less import feature_identity as less_identity, projected_features as less_features

def run_attribution(
    artifact_dir, source_path, query_path, method="rept", *,
    output_dir="output", device="cuda", topk=(1, 5, 10), layer=-1,
    source_limit=None, query_limit=None,
):
    """Dispatch attribution calls from the pipeline."""
    if method in ("tracin", "less"):
        runner = run_tracin if method == "tracin" else run_less
        return runner(
            artifact_dir, source_path, query_path, output_dir=output_dir,
            device=device, topk=topk, source_limit=source_limit, query_limit=query_limit,
        )
    if method != "rept":
        raise ValueError("Supported methods are 'rept', 'tracin', and 'less'")
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


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _tracin_weight(path):
    """Mean update learning rate in the epoch ending at this checkpoint."""
    metadata = _read_json(path / "metadata.json")
    state = _read_json(path / "trainer_state.json")
    epoch = int(round(float(state["epoch"])))
    recorded = metadata.get("tracin_learning_rate")
    if recorded is not None:
        if recorded["epoch"] != epoch or recorded["end_step"] != state["global_step"]:
            raise ValueError(f"Inconsistent recorded TracIn learning rate: {path}")
        if recorded["count"] <= 0 or not recorded.get("complete_start", True) or recorded["count"] != recorded["end_step"] - recorded["start_step"]:
            raise ValueError(f"Incomplete learning-rate interval: {path}")
        rate = float(recorded["sum"]) / recorded["count"]
        origin = "recorded-epoch-mean"
    else:
        # Older exports have no per-update rates. Reconstruct only a verifiable
        # uninterrupted standard cosine schedule, not a resumed/restarted run.
        options = _read_json(path / "training_args.json")
        history = state.get("log_history", [])
        total = int(state["max_steps"])
        step = int(state["global_step"])
        per_epoch = step / epoch
        if options["lr_scheduler_type"] != "cosine" or options.get("lr_scheduler_kwargs") or not math.isclose(
            per_epoch, round(per_epoch), abs_tol=1e-6
        ) or any("train_runtime" in log and log.get("step", step) < step for log in history):
            raise ValueError(f"Cannot reconstruct historical learning rates for {path}; use checkpoints with recorded rates")
        warmup = float(options["warmup_steps"])
        warmup = math.ceil(total * warmup) if warmup < 1 else int(warmup)
        base = float(options["learning_rate"])

        def rate_at(update):
            if update < warmup:
                return base * update / max(1, warmup)
            progress = (update - warmup) / max(1, total - warmup)
            return base * max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

        logs = [log for log in history if "learning_rate" in log and "step" in log]
        if not logs or not any(all(math.isclose(
            float(log["learning_rate"]), rate_at(int(log["step"]) - offset),
            rel_tol=1e-5, abs_tol=1e-12,
        ) for log in logs) for offset in (0, 1)):
            raise ValueError(f"Saved learning-rate logs do not verify a fresh cosine schedule: {path}")
        start = step - int(round(per_epoch))
        rate = sum(rate_at(update) for update in range(start, step)) / (step - start)
        origin = "reconstructed-fresh-cosine-epoch-mean"
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"TracIn checkpoint learning rate must be positive: {path}")
    return rate, origin


def _tracin_checkpoints(artifact_dir, output_dir):
    """Select one compatible completed checkpoint per epoch, ending at the target."""
    from .run_ft import check_training_artifacts, checkpoint_identity

    artifact_dir = Path(artifact_dir)
    reference = _read_json(artifact_dir / "metadata.json")
    training = _read_json(artifact_dir / "training_args.json")
    lora = _read_json(artifact_dir / "adapter_config.json")
    lora = {key: lora[key] for key in ("r", "lora_alpha", "lora_dropout", "task_type", "target_modules")}
    target = float(reference["trained_epochs"])
    if not math.isfinite(target) or target < 1 or not target.is_integer():
        raise ValueError("TracIn requires a completed integer epoch")
    candidates = [artifact_dir]
    if Path(output_dir).is_dir():
        candidates.extend(sorted(Path(output_dir).glob("checkpoint-*")))
    epochs = {}
    for path in candidates:
        try:
            metadata = _read_json(path / "metadata.json")
            epoch = float(metadata["trained_epochs"])
            if not epoch.is_integer() or not 1 <= epoch <= target:
                continue
            ancestors = reference.get("training_ancestors", [])
            try:
                verified_ancestor = bool(ancestors) and checkpoint_identity(path) in ancestors
            except (OSError, ValueError, KeyError):
                verified_ancestor = False
            if reference.get("training_run_id") != metadata.get("training_run_id") and not verified_ancestor:
                continue
            if any(reference.get(key) != metadata.get(key) for key in (
                "model_name", "model_revision", "source_fingerprint", "tokenized_fingerprint",
                "training_examples", "load_in_4bit", "max_length", "padding_side",
            )):
                continue
            _, reusable, _ = check_training_artifacts(
                reference["model_name"], reference["dataset_path"], lora, training,
                artifact_dir=path, target_epoch=epoch,
            )
            if not reusable:
                continue
            # Without a recorded run ID, differing schedules cannot be linked.
            if not verified_ancestor and reference.get("training_run_id") is None and _read_json(
                path / "training_args.json"
            )["num_train_epochs"] != training["num_train_epochs"]:
                continue
            key = int(epoch)
            if key in epochs:
                previous = epochs[key]
                def adapter_hash(folder):
                    name = "adapter_model.safetensors" if (folder / "adapter_model.safetensors").is_file() else "adapter_model.bin"
                    return _sha256_file(folder / name)
                if adapter_hash(previous) != adapter_hash(path):
                    raise ValueError(f"Conflicting adapters for epoch {key}; use a separate output directory per training run")
            else:
                epochs[key] = path
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
    missing = sorted(set(range(1, int(target) + 1)) - set(epochs))
    if missing:
        raise ValueError(f"TracIn needs compatible completed checkpoints for epochs {missing} in {output_dir}")
    return [(epochs[epoch], *_tracin_weight(epochs[epoch])) for epoch in sorted(epochs)]


def _tracin_parameters(model, checkpoint):
    """Use the exact saved optimizer parameter order and freeze all other weights."""
    parameters = dict(model.named_parameters())
    groups = _read_json(Path(checkpoint) / "optimizer_parameters.json")
    entries = [entry for group in groups for entry in group]
    names = [entry["name"] for entry in entries]
    if not names or len(names) != len(set(names)):
        raise ValueError("Saved optimizer parameter names must be nonempty and unique")
    for entry in entries:
        if entry["name"] not in parameters or list(parameters[entry["name"]].shape) != entry["shape"]:
            raise ValueError(f"Saved optimizer parameter does not match loaded model: {entry['name']}")
    selected = set(names)
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in selected)
    return names, sum(parameters[name].numel() for name in names)


def _tracin_scoring_blocks(dimensions, device="cpu", block=None):
    """Spend more memory on query rows to reduce complete source-matrix passes."""
    if dimensions <= 0 or (block is not None and (type(block) is not int or block <= 0)):
        raise ValueError("Gradient dimensions and scoring block sizes must be positive")
    if str(device) == "cpu":
        budget = 2 * 2**30
    elif str(device).split(":")[0] == "cuda":
        import torch
        target = torch.device(device)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA scoring requested but CUDA is unavailable")
        free, _ = torch.cuda.mem_get_info(target)
        budget = min(int(free * 0.80), 8 * 2**30)
    else:
        raise ValueError("TracIn scoring supports cpu or cuda devices")
    row_bytes = dimensions * 4
    if 2 * row_bytes > budget:
        raise MemoryError("Insufficient scoring memory for two complete gradient rows")
    query_rows = min(512, max(1, int(budget * 0.75) // row_bytes))
    source_rows = min(128, max(1, (budget - query_rows * row_bytes) // row_bytes))
    if block is not None:
        query_rows, source_rows = min(block, query_rows), min(block, source_rows)
    return query_rows, source_rows


def _cuda_gradient_block(matrix, start, stop, device, staging_bytes=256 * 2**20):
    """Overlap row reads and asynchronous transfers using two pinned host buffers."""
    import numpy as np
    import torch

    result = torch.empty((stop - start, matrix.shape[1]), dtype=torch.float32, device=device)
    transfer_rows = min(stop - start, max(1, staging_bytes // (matrix.shape[1] * 4)))
    # A row larger than the staging budget is still transferred intact.
    buffers = [torch.empty((transfer_rows, matrix.shape[1]), dtype=torch.float32, pin_memory=True)
               for _ in range(2)]
    events = [None, None]
    stream = torch.cuda.current_stream(device)
    try:
        with torch.cuda.stream(stream):
            for number, row in enumerate(range(start, stop, transfer_rows)):
                slot = number % 2
                if events[slot] is not None:
                    events[slot].synchronize()  # Do not overwrite a buffer still being transferred.
                end = min(row + transfer_rows, stop)
                host = buffers[slot][:end - row]
                np.copyto(host.numpy(), matrix[row:end], casting="same_kind")
                result[row - start:end - start].copy_(host, non_blocking=True)
                event = torch.cuda.Event()
                event.record(stream)
                events[slot] = event
    finally:
        # Keep pinned buffers alive until all queued copies complete, including
        # when interrupted. The returned tensor is ready on the caller's stream.
        stream.synchronize()
    return result


def _accumulate_tracin(scores, sources, queries, weight, block=None, progress=None, device="cpu"):
    """Float32 CPU/CUDA dot products; retain query blocks between source reads."""
    import numpy as np
    from contextlib import ExitStack

    if sources.ndim != 2 or queries.ndim != 2 or sources.shape[1] != queries.shape[1]:
        raise ValueError("TracIn gradient matrix dimensions must match")
    if scores.shape != (len(queries), len(sources)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("Invalid TracIn score shape or checkpoint weight")
    query_block, source_block = _tracin_scoring_blocks(sources.shape[1], device, block)
    cuda = str(device).split(":")[0] == "cuda"
    with ExitStack() as context:
        if cuda:
            import torch
            backend = torch.backends.cuda.matmul
            setting = "fp32_precision" if hasattr(backend, "fp32_precision") else "allow_tf32"
            previous = getattr(backend, setting)
            context.callback(setattr, backend, setting, previous)
            setattr(backend, setting, "ieee" if setting == "fp32_precision" else False)
            context.enter_context(torch.inference_mode())
            context.enter_context(torch.autocast(device_type="cuda", enabled=False))
        for q in range(0, len(queries), query_block):
            q_end = min(q + query_block, len(queries))
            query_values = (_cuda_gradient_block(queries, q, q_end, device) if cuda else
                            np.array(queries[q:q_end], dtype=np.float32, order="C", copy=True))
            for s in range(0, len(sources), source_block):
                s_end = min(s + source_block, len(sources))
                source_values = (_cuda_gradient_block(sources, s, s_end, device) if cuda else
                                 np.array(sources[s:s_end], dtype=np.float32, order="C", copy=True))
                product = query_values @ source_values.T
                contribution = product.cpu().numpy() if cuda else product
                if not np.isfinite(contribution).all():
                    raise ValueError("Nonfinite TracIn dot product")
                scores[q:q_end, s:s_end] += weight * contribution.astype(np.float64)
                if progress is not None:
                    progress(contribution.size)
                del source_values, product, contribution
            del query_values


def _tracin_rows(frame):
    rows = frame[["label"]].copy()
    rows.insert(0, "row_index", frame.index)
    return json.loads(rows.to_json(orient="records"))


def _gradient_identity(entry, manifest, frame, response):
    """Identify gradient inputs independently of paths, metric cutoffs, and weights."""
    rows = frame[["prompts", response, "label"]].to_json(orient="split", force_ascii=False)
    return {
        "feature_version": manifest["feature_version"],
        "source_fingerprint": manifest["source_fingerprint"],
        "query_sha256": manifest["query_sha256"],
        "rows_sha256": hashlib.sha256(rows.encode("utf-8")).hexdigest(),
        "shape": [len(frame), entry["dimensions"]], "dtype": "float32",
        "parameter_groups": entry["parameter_groups"],
        "artifact_sha256": entry["artifact_sha256"],
        "model_name": manifest["model_name"], "model_revision": entry["model_revision"],
        "load_in_4bit": entry["load_in_4bit"], "quantization": entry["quantization"],
        "loss": manifest["loss"], "max_length": manifest["max_length"],
        "padding_side": manifest["padding_side"], "device": manifest["device"],
    }


def _gradient_rows(directory, kind, frame, response, identity, extract):
    """Resume only flushed row prefixes; publish progress atomically every 8 rows."""
    import shutil
    import numpy as np
    from tqdm import tqdm

    path = directory / f"{kind}_features.npy"
    progress_path = directory / f"{kind}_progress.json"
    matrix, completed = None, 0
    try:
        if progress_path.is_file():
            progress = _read_json(progress_path)
            if progress["identity"] != identity:
                raise ValueError("gradient inputs changed")
            completed = progress["completed_rows"]
            if type(completed) is not int or not 0 <= completed <= len(frame):
                raise ValueError("invalid completed row count")
        else:
            raise ValueError("no compatible saved row progress")
        matrix = np.load(path, mmap_mode="r" if completed == len(frame) else "r+", allow_pickle=False)
        if list(matrix.shape) != identity["shape"] or matrix.dtype != np.float32 or (
            matrix.offset + matrix.nbytes != path.stat().st_size
        ):
            raise ValueError("gradient file shape, dtype, or length changed")
    except (OSError, ValueError, KeyError, TypeError) as error:
        if path.exists() or progress_path.exists():
            print(f"Saved TracIn {kind} gradients cannot be reused: {error}", flush=True)
        matrix = None
        completed = 0

    if matrix is None:
        required = math.prod(identity["shape"]) * 4
        # Existing files will be truncated rather than duplicated. Reserve only
        # the extra file length; actual write failures leave committed progress.
        existing = path.stat().st_size if path.is_file() else 0
        if max(0, required - existing) + 64 * 2**20 > shutil.disk_usage(directory).free:
            raise OSError("Insufficient free disk space for TracIn gradients; use smaller --sources/--queries")
        # Invalidate progress before replacing data, so a crash during truncate
        # cannot leave an old completed prefix pointing to an empty matrix.
        (directory / "features_complete.json").unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=tuple(identity["shape"]))
        _write_json(progress_path, {"identity": identity, "completed_rows": 0})
        print(f"Extracting TracIn {kind} gradients: {len(frame)} rows", flush=True)
    elif completed == len(frame):
        print(f"Reusing TracIn {kind} gradients: {completed}/{len(frame)} rows", flush=True)
    else:
        print(f"Resuming TracIn {kind} gradients at row {completed}/{len(frame)}", flush=True)

    if completed == len(frame):
        _write_json(progress_path, {"identity": identity, "completed_rows": completed})
        _write_json(directory / f"{kind}_rows.json", _tracin_rows(frame))
        return matrix

    def commit():
        matrix.flush()
        _write_json(progress_path, {"identity": identity, "completed_rows": completed})

    (directory / "features_complete.json").unlink(missing_ok=True)
    try:
        for row in tqdm(range(completed, len(frame)), initial=completed, total=len(frame), desc=f"TracIn {kind}"):
            sample = frame.iloc[row]
            vector = extract(sample["prompts"], sample[response])
            if vector.shape != (identity["shape"][1],) or not np.isfinite(vector).all():
                raise ValueError("Nonfinite or incorrectly shaped TracIn gradient")
            matrix[row] = vector
            completed = row + 1
            if completed % 8 == 0:
                commit()
    finally:
        # Includes Ctrl+C and extraction failures. Only fully written rows count.
        commit()
    _write_json(directory / f"{kind}_rows.json", _tracin_rows(frame))
    return matrix


def _checkpoint_scores(directory, matrices, identity, weight, progress_factory, device="cpu", *,
                       score_key=None, label="TracIn"):
    """Reuse completed checkpoint contributions without double-counting on restart."""
    import numpy as np
    from zipfile import BadZipFile

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "checkpoint_scores.npz"
    complete = directory / "scores_complete.json"
    key = _score_identity(identity, weight, device) if score_key is None else score_key
    shape = (len(matrices[1]), len(matrices[0]))
    try:
        if _read_json(complete) != key:
            raise ValueError("score inputs changed")
        with np.load(path, allow_pickle=False) as saved:
            scores = saved["scores"]
        if scores.shape != shape or scores.dtype != np.float64 or not np.isfinite(scores).all():
            raise ValueError("invalid checkpoint scores")
        print("Reusing completed checkpoint scores", flush=True)
        return scores
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
        pass
    complete.unlink(missing_ok=True)
    scores = np.zeros(shape, dtype=np.float64)
    with progress_factory(total=shape[0] * shape[1], unit="pair", desc=f"{label} scoring") as progress:
        _accumulate_tracin(scores, matrices[0], matrices[1], weight, progress=progress.update, device=device)
    pending = path.with_name(path.name + ".tmp")
    with pending.open("wb") as stream:
        np.savez_compressed(stream, scores=scores)
    pending.replace(path)
    _write_json(complete, key)
    return scores


def _score_identity(gradients, weight, device):
    return {"version": "tracin-checkpoint-dot-v2", "gradients": gradients, "weight": weight,
            "scoring_device": str(device), "precision": "float32-ieee"}


def run_tracin(
    artifact_dir, source_path, query_path, *, output_dir="output", device="cuda",
    topk=(1, 5, 10), source_limit=None, query_limit=None,
):
    """TracInCP: sum learning-rate-weighted adapter-gradient dot products."""
    return _run_gradient_attribution(
        artifact_dir, source_path, query_path, output_dir=output_dir, device=device,
        topk=topk, source_limit=source_limit, query_limit=query_limit, method="tracin",
    )


def run_less(
    artifact_dir, source_path, query_path, *, output_dir="output", device="cuda",
    topk=(1, 5, 10), source_limit=None, query_limit=None,
):
    """LESS: projected source Adam directions versus projected query gradients."""
    return _run_gradient_attribution(
        artifact_dir, source_path, query_path, output_dir=output_dir, device=device,
        topk=topk, source_limit=source_limit, query_limit=query_limit, method="less",
    )


def _run_gradient_attribution(
    artifact_dir, source_path, query_path, *, output_dir, device, topk,
    source_limit, query_limit, method,
):
    """Share checkpoint validation and raw gradient extraction across methods."""
    import gc
    import numpy as np
    import pandas as pd
    import torch
    from datasets import load_from_disk
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    from tqdm import tqdm
    from src import TDA, TDAEvaluation, Params

    topk = sorted(set(topk))
    if not topk or any(not isinstance(k, int) or k <= 0 for k in topk):
        raise ValueError("topk must contain positive integers")
    for limit in (source_limit, query_limit):
        if limit is not None and (not isinstance(limit, int) or limit <= 0):
            raise ValueError("Source/query limits must be positive integers")
    dataset = load_from_disk(str(source_path))
    sources, queries = dataset.to_pandas(), pd.read_csv(query_path)
    for frame, required in ((sources, ["prompts", "response", "label"]),
                            (queries, ["prompts", "expected_response", "label"])):
        if not set(required) <= set(frame.columns) or frame.empty or frame[required].isna().any().any():
            raise ValueError(f"Prepared data needs nonempty values for {required}")
    sources, queries = _limit_by_label(sources, source_limit), _limit_by_label(queries, query_limit)
    if topk[-1] > len(sources) or set(queries["label"]) - set(sources["label"]):
        raise ValueError("Retrieval cutoffs and query labels must have supporting selected sources")
    metadata = _read_json(Path(artifact_dir) / "metadata.json")
    if dataset._fingerprint != metadata["source_fingerprint"]:
        raise ValueError("Training source fingerprint does not match attribution sources")
    checkpoints = _tracin_checkpoints(artifact_dir, output_dir)
    target_device = torch.device(device)
    if metadata["load_in_4bit"] and target_device.type != "cuda":
        raise ValueError("These 4-bit training checkpoints require CUDA for matching gradient model loading")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    method_dir = Path(output_dir) / method / f"epoch-{float(metadata['trained_epochs']):g}"
    method_dir.mkdir(parents=True, exist_ok=True)
    (method_dir / "complete.json").unlink(missing_ok=True)
    (method_dir / "metrics.xlsx").unlink(missing_ok=True)
    scores = np.zeros((len(queries), len(sources)), dtype=np.float64)
    manifest = {"model_name": metadata["model_name"], "method": method, "feature_version": "tracincp-adapter-dot-v1",
                "scoring_device": str(target_device), "scoring_version": "tracin-checkpoint-dot-v2",
                "scoring_precision": "float32-ieee",
                "formula": "sum_c mean_epoch_learning_rate_c * query_gradient_c @ source_gradient_c.T",
                "checkpoint_selection": "completed-epochs-1-through-target",
                "learning_rate_convention": "mean of update learning rates in each preceding epoch; skipped updates contribute zero",
                "loss": "mean causal cross entropy over response tokens; prompt masked",
                "normalization": "none", "gradient_dtype": "float32", "score_dtype": "float64",
                "parameter_selection": "saved optimizer parameter names and order",
                "max_length": metadata["max_length"], "padding_side": metadata["padding_side"],
                "training_run_id": metadata.get("training_run_id"),
                "source_path": str(Path(source_path).resolve()), "query_path": str(Path(query_path).resolve()),
                "selection": "round-robin-label-order-v1",
                "source_fingerprint": dataset._fingerprint, "query_sha256": _sha256_file(query_path),
                "source_limit": source_limit, "query_limit": query_limit,
                "device": str(target_device), "checkpoints": []}
    if method == "less":
        total_weight = sum(weight for _, weight, _ in checkpoints)
        if not math.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("LESS needs a positive sum of checkpoint learning rates")
        manifest.update(
            scoring_version="less-reference-adam-cosine-v1",
            formula="sum_c normalized_mean_epoch_learning_rate_c * cosine(project(query_gradient), project(source_adam_direction))",
            normalization="row L2 after projection; checkpoint weights sum to one",
            source_transform="reference Adam moments without bias correction; epsilon inside sqrt",
            query_transform="untransformed raw gradient",
            evaluation="per-query attribution; no task averaging or subset retraining",
        )
        # Fail before gradient extraction if optimizer files are missing.
        for checkpoint, _, _ in checkpoints:
            if not (checkpoint / "optimizer.pt").is_file():
                raise ValueError(f"LESS needs saved optimizer moments: {checkpoint / 'optimizer.pt'}")
    for checkpoint, weight, weight_source in checkpoints:
        state = _read_json(checkpoint / "metadata.json")
        print(f"{'LESS' if method == 'less' else 'TracIn'} epoch {state['trained_epochs']}: weight={weight:.8g}; checking saved gradients...", flush=True)
        parameter_metadata = _read_json(checkpoint / "optimizer_parameters.json")
        dimensions = sum(math.prod(parameter["shape"]) for group in parameter_metadata for parameter in group)
        entry = {"artifact_dir": str(checkpoint.resolve()), "global_step": state["global_step"],
                 "epoch": state["trained_epochs"], "weight": weight, "weight_source": weight_source,
                 "parameter_groups": parameter_metadata, "dimensions": dimensions,
                 "source_shape": [len(sources), dimensions], "query_shape": [len(queries), dimensions],
                 "quantization": "4bit-default-fp4-double-quant-fp16-compute" if state["load_in_4bit"] else None,
                 "model_revision": state.get("model_revision"), "load_in_4bit": state["load_in_4bit"],
                 "artifact_sha256": {path.name: _sha256_file(path) for path in checkpoint.iterdir()
                     if path.name in ("adapter_model.safetensors", "adapter_model.bin", "adapter_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")}}
        identities = [
            _gradient_identity(entry, manifest, sources, "response"),
            _gradient_identity(entry, manifest, queries, "expected_response"),
        ]
        directory = gradient_directory(output_dir, identities)
        entry["gradient_cache"] = str(directory.relative_to(Path(output_dir)))
        if method == "less":
            processed_identity = less_identity(identities, _sha256_file(checkpoint / "optimizer.pt"), target_device)
            checkpoint_key = {"version": "less-cosine-v1", "features": processed_identity,
                              "scoring_device": str(target_device), "precision": "float32-ieee"}
            score_dir = Path(output_dir) / "less" / "checkpoint_scores" / cache_key(checkpoint_key)
            entry["normalized_weight"] = weight / total_weight
        else:
            score_dir = score_directory(output_dir, _score_identity(identities, weight, str(target_device)))
        entry["score_cache"] = str(score_dir.relative_to(Path(output_dir)))
        required_bytes = (len(sources) + len(queries)) * entry["dimensions"] * 4
        print(f"Raw gradient storage: approximately {required_bytes / 2**30:.2f} GiB per checkpoint (existing compatible files are reused)", flush=True)
        loaded = []

        def extract(prompt, response):
            if not loaded:
                print("Loading model for missing gradients...", flush=True)
                tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
                tokenizer.padding_side = state["padding_side"]
                options = {"revision": state["model_revision"]} if state.get("model_revision") else {}
                if state["load_in_4bit"]:
                    options.update(quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                    ), device_map={"": str(target_device)})
                base = AutoModelForCausalLM.from_pretrained(state["model_name"], **options)
                model = PeftModel.from_pretrained(base, str(checkpoint), is_trainable=True)
                if not state["load_in_4bit"]:
                    model.to(target_device)
                model.config.use_cache = False
                names, dimensions = _tracin_parameters(model, checkpoint)
                if dimensions != entry["dimensions"]:
                    raise ValueError("Loaded gradient dimensions differ from saved optimizer mapping")
                loaded.extend([TDA(model, tokenizer, Params(layer_norm=False), device=str(target_device)), names])
            return loaded[0].get_gradient_vector(prompt, response, loaded[1]).cpu().numpy()

        matrices = []
        try:
            for kind, frame, response, identity in (
                ("source", sources, "response", identities[0]),
                ("query", queries, "expected_response", identities[1]),
            ):
                matrices.append(_gradient_rows(directory, kind, frame, response, identity, extract))
        finally:
            loaded.clear()
            gc.collect()
            if target_device.type == "cuda":
                torch.cuda.empty_cache()
        _write_json(directory / "feature_metadata.json", entry)
        _write_json(directory / "features_complete.json", {"feature_version": manifest["feature_version"]})
        if method == "less":
            processed_dir, processed = less_features(
                output_dir, matrices, processed_identity, checkpoint, parameter_metadata, str(target_device),
            )
            entry["feature_cache"] = str(processed_dir.relative_to(Path(output_dir)))
            print("Accumulating LESS checkpoint cosine similarities...", flush=True)
            scores += entry["normalized_weight"] * _checkpoint_scores(
                score_dir, processed, identities, 1.0, tqdm, device=str(target_device),
                score_key=checkpoint_key, label="LESS",
            )
            del processed
        else:
            print("Accumulating checkpoint gradient dot products...", flush=True)
            scores += _checkpoint_scores(score_dir, matrices, identities, weight, tqdm, device=str(target_device))
        manifest["checkpoints"].append(entry)
        del matrices
        gc.collect()
    pending = method_dir / "scores.npz.tmp"
    with pending.open("wb") as stream:
        np.savez_compressed(stream, scores=scores, source_indices=sources.index.to_numpy(dtype=np.int64),
                            query_indices=queries.index.to_numpy(dtype=np.int64))
    pending.replace(method_dir / "scores.npz")
    _write_json(method_dir / "feature_metadata.json", manifest)
    results = TDAEvaluation.compute_from_scores(sources, queries, scores, topk, method)
    metrics = pd.DataFrame([result.__dict__ for result in results])
    metrics.to_excel(method_dir / "metrics.xlsx", index=False)
    _write_json(method_dir / "complete.json", {"feature_version": manifest["feature_version"]})
    print(f"Saved {method} gradients, scores, and metrics: {method_dir}", flush=True)
    return metrics


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
