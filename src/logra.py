"""LoGRA-random: Kronecker-projected weight gradients and damped influence."""

import hashlib
import json
import math
from pathlib import Path

from .gradient_cache import cache_key
from .relevance import missing_query_labels
from .lorif import ModelSession as WeightModelSession, _read, _write, _save_torch, select_modules


SETTINGS = {"projection_dim": 64, "projection_seed": 0,
            "projection_init": "logix-kaiming-uniform", "damping_multiplier": 0.1,
            "compute_dtype": "float32", "storage_dtype": "float32",
            "batch_size": 4, "shard_size": 128}
FEATURE_KEYS = ("projection_dim", "projection_seed", "projection_init", "compute_dtype",
                "storage_dtype", "batch_size", "shard_size")


class GradientCollector:
    """Compute Eq. 6 from projected activations, without forming full gradients."""

    def __init__(self, model, names, settings):
        import torch
        from modules.lorif.src.lorif.collector import LayerSpec
        self.specs, self.projections, self.activations, self.gradients = {}, {}, {}, {}
        self.handles = []
        modules = dict(model.named_modules())
        rank = settings["projection_dim"]
        if not names or len(names) != len(set(names)) or rank <= 0:
            raise ValueError("LoGRA needs unique linear modules and a positive projection dimension")
        for name in names:
            module = modules[name]
            if not isinstance(module, torch.nn.Linear):
                raise TypeError(f"LoGRA module is not linear: {name}")
            k = min(rank, module.in_features, module.out_features)
            self.specs[name] = LayerSpec(name, module.in_features, module.out_features, k, k)
            matrices = []
            for kind, shape, bound in (("input", (module.in_features, k), 1 / math.sqrt(module.in_features)),
                                       ("output", (module.out_features, k), 1 / math.sqrt(k))):
                seed = int(cache_key([settings["projection_seed"], name, kind])[:16], 16)
                generator = torch.Generator(device="cpu").manual_seed(seed)
                # Equivalent distributions to LogIX's A and C Kaiming-uniform initialization.
                matrices.append(torch.empty(shape, dtype=torch.float32, device="cpu")
                                .uniform_(-bound, bound, generator=generator)
                                .to(device=module.weight.device, dtype=torch.float32))
            self.projections[name] = matrices
            self.activations[name] = []
        try:
            for name in names:
                self.handles.append(modules[name].register_forward_hook(self._forward(name)))
                self.handles.append(modules[name].register_full_backward_hook(self._backward(name)))
        except Exception:
            self.close()
            raise

    def projection_sha256(self):
        digest = hashlib.sha256(b"logra-kaiming-projections-v1")
        for name, matrices in self.projections.items():
            digest.update(name.encode())
            for matrix in matrices:
                digest.update(str(tuple(matrix.shape)).encode())
                digest.update(matrix.cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _forward(self, name):
        def hook(module, inputs, output):
            import torch
            with torch.no_grad():
                values = inputs[0].detach()
                values = values.reshape(values.shape[0], -1, self.specs[name].input_dim)
                self.activations[name].append(values.float() @ self.projections[name][0])
        return hook

    def _backward(self, name):
        def hook(module, grad_input, grad_output):
            import torch
            if not self.activations[name] or not grad_output or grad_output[0] is None:
                raise RuntimeError(f"Incomplete LoGRA backward pass for {name}")
            with torch.no_grad():
                activation = self.activations[name].pop()
                delta = grad_output[0].detach()
                delta = delta.reshape(delta.shape[0], -1, self.specs[name].output_dim)
                if activation.shape[:2] != delta.shape[:2]:
                    raise ValueError(f"LoGRA activation/gradient positions differ for {name}")
                value = (activation.transpose(1, 2) @ (delta.float() @ self.projections[name][1])).cpu()
                if name in self.gradients:
                    self.gradients[name].add_(value)
                else:
                    self.gradients[name] = value
        return hook

    def clear(self):
        for values in self.activations.values():
            values.clear()
        self.gradients.clear()

    def pop(self):
        if any(self.activations.values()) or set(self.gradients) != set(self.specs):
            raise RuntimeError("LoGRA did not capture a complete backward pass for every layer")
        result, self.gradients = self.gradients, {}
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()


class ModelSession(WeightModelSession):
    method_name = "LoGRA"

    def make_collector(self):
        return GradientCollector(self.model, select_modules(self.model), self.settings)


class FeatureIndex:
    def __init__(self, directory, manifest):
        self.directory, self.manifest = Path(directory), manifest
        self.specs = {spec["name"]: spec for spec in manifest["signature"]["specs"]}

    def path(self, name, shard):
        return self.directory / cache_key(name) / f"shard-{shard:05d}.pt"

    def batches(self, name, device="cpu"):
        import torch
        spec = self.specs[name]
        dimension = spec["projected_input_dim"] * spec["projected_output_dim"]
        for number, shard in enumerate(self.manifest["shards"]):
            value = torch.load(self.path(name, number), map_location="cpu", weights_only=True)
            if (value.shape != (shard["end"] - shard["start"], dimension)
                    or value.dtype != torch.float32 or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid LoGRA gradient shard: {self.path(name, number)}")
            yield value.to(device=device)


def ensure_features(directory, frame, response, identity, session_factory, progress_factory):
    """Publish progress only after every layer of a row shard is written."""
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "index.json"
    try:
        manifest = _read(path)
        if manifest["identity"] != identity:
            raise ValueError("LoGRA feature identity mismatch")
        index = FeatureIndex(directory, manifest)
        completed, kept = 0, []
        for number, shard in enumerate(manifest["shards"]):
            if (shard["start"] != completed or not completed < shard["end"] <= len(frame)
                    or any(not index.path(name, number).is_file() for name in index.specs)):
                break
            completed = shard["end"]
            kept.append(shard)
        manifest["shards"] = kept
    except FileNotFoundError:
        manifest, completed = None, 0
    if completed == len(frame):
        print(f"Reusing LoGRA {identity['kind']} gradients: {completed}/{len(frame)} rows", flush=True)
        return FeatureIndex(directory, manifest)
    session = session_factory()
    if manifest is None:
        manifest = {"identity": identity, "signature": session.signature, "shards": []}
    elif manifest["signature"] != session.signature:
        raise ValueError("LoGRA projections or selected modules changed")
    index = FeatureIndex(directory, manifest)
    settings = identity["settings"]
    with progress_factory(total=len(frame), initial=completed, desc=f"LoGRA {identity['kind']} gradients") as progress:
        for start in range(completed, len(frame), settings["shard_size"]):
            stop = min(start + settings["shard_size"], len(frame))
            buffer = {name: [] for name in index.specs}
            for row in range(start, stop, settings["batch_size"]):
                end = min(row + settings["batch_size"], stop)
                values = session.extract(frame.iloc[row:end], response)
                if set(values) != set(buffer):
                    raise ValueError("LoGRA collector layer mismatch")
                for name, value in values.items():
                    spec = index.specs[name]
                    if (value.shape != (end - row, spec["projected_input_dim"], spec["projected_output_dim"])
                            or not torch.isfinite(value).all()):
                        raise ValueError(f"Invalid LoGRA projected gradients for {name}")
                    buffer[name].append(value.detach().flatten(1).float().cpu())
                progress.update(end - row)
            for name, batches in buffer.items():
                _save_torch(index.path(name, len(manifest["shards"])), torch.cat(batches))
            manifest["shards"].append({"start": start, "end": stop})
            _write(path, manifest)
    return index


def influence(train, query, damping_multiplier=0.1):
    """Exact layer-block solve with empirical Fisher H=G.T G / N and trace damping."""
    import torch
    if (train.ndim != 2 or query.ndim != 2 or train.shape[1] != query.shape[1]
            or min(train.shape) == 0 or not math.isfinite(damping_multiplier) or damping_multiplier <= 0
            or not torch.isfinite(train).all() or not torch.isfinite(query).all()):
        raise ValueError("LoGRA needs finite compatible matrices and positive damping")
    count, dimension = train.shape
    trace = train.square().sum() / count
    if trace == 0:
        return train.new_zeros((len(query), count))
    damping = damping_multiplier * trace / dimension
    if dimension <= count:
        covariance = train.T @ train / count
        covariance.diagonal().add_(damping)
        cholesky = torch.linalg.cholesky(covariance)
        result = torch.cholesky_solve(query.T.contiguous(), cholesky).T @ train.T
    else:
        # Algebraically identical dual solve avoids allocating dimension-squared curvature.
        covariance = train @ train.T / count
        covariance.diagonal().add_(damping)
        cholesky = torch.linalg.cholesky(covariance)
        result = torch.cholesky_solve((query @ train.T).T.contiguous(), cholesky).T
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite LoGRA influence scores")
    return result


def score_layer(train, query, name, settings, device):
    import torch
    with torch.no_grad():
        sources = torch.cat(list(train.batches(name, device)))
        queries = torch.cat(list(query.batches(name, device)))
        return influence(sources, queries, settings["damping_multiplier"]).double().cpu().numpy()


def score_indexes(directory, train, query, identity, settings, device, progress_factory):
    """Atomically checkpoint the cumulative score and layer count in one file."""
    import numpy as np
    if train.manifest["signature"] != query.manifest["signature"]:
        raise ValueError("LoGRA source/query projections and layer specifications must match")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "scores_progress.npz"
    names = list(train.specs)
    shape = (query.manifest["shards"][-1]["end"], train.manifest["shards"][-1]["end"])
    scores, completed = np.zeros(shape, dtype=np.float64), 0
    try:
        with np.load(path, allow_pickle=False) as saved:
            if json.loads(str(saved["identity"])) != identity:
                raise ValueError("LoGRA score identity changed")
            scores, completed = saved["scores"], int(saved["completed_layers"])
        if scores.shape != shape or scores.dtype != np.float64 or not np.isfinite(scores).all() or not 0 <= completed <= len(names):
            raise ValueError("Invalid LoGRA score progress")
    except FileNotFoundError:
        pass
    if completed == len(names):
        print("Reusing completed LoGRA influence scores", flush=True)
        return scores
    with progress_factory(total=len(names), initial=completed, desc="LoGRA curvature/scoring layers") as progress:
        for number in range(completed, len(names)):
            if hasattr(progress, "set_postfix_str"):
                progress.set_postfix_str(names[number].removeprefix("base_model.model.model."), refresh=True)
            scores += score_layer(train, query, names[number], settings, device)
            pending = path.with_name(path.name + ".tmp")
            with pending.open("wb") as stream:
                np.savez_compressed(stream, scores=scores, completed_layers=number + 1,
                                    identity=json.dumps(identity, sort_keys=True))
            pending.replace(path)
            progress.update(1)
    return scores


def run_logra(artifact_dir, source_path, query_path, *, output_dir="output", device="cuda",
              topk=(1, 5, 10), source_limit=None, query_limit=None):
    import gc
    import numpy as np
    import pandas as pd
    import torch
    from datasets import load_from_disk
    from tqdm import tqdm
    from .run_tda import _limit_by_label, _sha256_file

    topk = sorted(set(topk))
    if not topk or any(type(k) is not int or k <= 0 for k in topk):
        raise ValueError("topk must contain positive integers")
    if any(limit is not None and (type(limit) is not int or limit <= 0) for limit in (source_limit, query_limit)):
        raise ValueError("Source/query limits must be positive integers")
    artifact_dir, root = Path(artifact_dir), Path(output_dir)
    metadata = _read(artifact_dir / "metadata.json")
    dataset = load_from_disk(str(source_path))
    if dataset._fingerprint != metadata["source_fingerprint"]:
        raise ValueError("LoGRA source fingerprint differs from the training dataset")
    sources, queries = dataset.to_pandas(), pd.read_csv(query_path)
    for frame, columns in ((sources, ["prompts", "response", "label"]),
                           (queries, ["prompts", "expected_response", "label"])):
        if not set(columns) <= set(frame.columns) or frame.empty or frame[columns].isna().any().any():
            raise ValueError(f"Prepared LoGRA data needs nonempty values for {columns}")
    sources, queries = _limit_by_label(sources, source_limit), _limit_by_label(queries, query_limit)
    if topk[-1] > len(sources) or missing_query_labels(sources, queries):
        raise ValueError("Retrieval cutoffs and query labels must have supporting selected sources")
    target = torch.device(device)
    if metadata["load_in_4bit"] and target.type != "cuda":
        raise ValueError("LoGRA requires CUDA for this 4-bit training checkpoint")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model_identity = {"model_name": metadata["model_name"], "model_revision": metadata.get("model_revision"),
        "load_in_4bit": metadata["load_in_4bit"], "max_length": 512, "padding_side": metadata["padding_side"],
        "device": str(target), "loss": "response-token-sum-causal-cross-entropy",
        "module_selection": "attention-MLP-q-k-v-o-gate-up-down-weight-branches-v1",
        "artifact_sha256": {path.name: _sha256_file(path) for path in artifact_dir.iterdir() if path.name in (
            "adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", "tokenizer.json",
            "tokenizer_config.json", "chat_template.jinja")}}
    session, indexes, identities = [], [], []
    def session_factory():
        if not session:
            session.append(ModelSession(artifact_dir, metadata, target, SETTINGS))
        return session[0]
    try:
        for kind, frame, response in (("source", sources, "response"), ("query", queries, "expected_response")):
            rows = frame[["prompts", response, "label"]].to_json(orient="split", force_ascii=False)
            identity = {"version": "logra-dense-weight-gradients-v1", "model": model_identity,
                        "settings": {key: SETTINGS[key] for key in FEATURE_KEYS},
                        "kind": kind, "rows_sha256": hashlib.sha256(rows.encode()).hexdigest(), "rows": len(frame)}
            identities.append(identity)
            directory = root / "logra" / "features" / cache_key(identity)
            indexes.append(ensure_features(directory, frame, response, identity, session_factory, tqdm))
            row_map = frame[["label"]].copy()
            row_map.insert(0, "row_index", frame.index)
            _write(directory / "rows.json", json.loads(row_map.to_json(orient="records")))
    finally:
        if session:
            session[0].close()
        session.clear()
        gc.collect()
        if target.type == "cuda":
            torch.cuda.empty_cache()
    score_identity = {"version": "logra-empirical-fisher-dot-v1", "features": identities,
                      "signature": indexes[0].manifest["signature"], "settings": dict(SETTINGS)}
    score_dir = root / "logra" / "scores" / cache_key(score_identity)
    scores = score_indexes(score_dir, indexes[0], indexes[1], score_identity, SETTINGS, target, tqdm)
    result_dir = root / "logra" / f"epoch-{float(metadata['trained_epochs']):g}"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "complete.json").unlink(missing_ok=True)
    with (result_dir / "scores.npz.tmp").open("wb") as stream:
        np.savez_compressed(stream, scores=scores, source_indices=sources.index.to_numpy(dtype=np.int64),
                            query_indices=queries.index.to_numpy(dtype=np.int64))
    (result_dir / "scores.npz.tmp").replace(result_dir / "scores.npz")
    manifest = {"method": "logra", "epoch": metadata["trained_epochs"], "settings": dict(SETTINGS),
        "model": model_identity, "checkpoint_selection": "selected-completed-epoch",
        "source_features": str(indexes[0].directory.relative_to(root)),
        "query_features": str(indexes[1].directory.relative_to(root)), "score_cache": str(score_dir.relative_to(root)),
        "formula": "sum_layers query_gradient.T @ (G.T G / N + lambda I)^-1 @ source_gradient",
        "damping": "0.1 * trace(G.T G / N) / projected_dimension",
        "curvature_persistence": "none; cumulative scores checkpointed after each layer"}
    _write(result_dir / "feature_metadata.json", manifest)
    from src import TDAEvaluation
    metrics = pd.DataFrame([result.__dict__ for result in TDAEvaluation.compute_from_scores(
        sources, queries, scores, topk, "logra")])
    metrics.to_excel(result_dir / "metrics.xlsx", index=False)
    _write(result_dir / "complete.json", {"score_version": score_identity["version"]})
    print(f"Saved LoGRA features, scores, and metrics: {result_dir}", flush=True)
    return metrics
