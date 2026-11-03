"""LoRIF integration using the bundled official collector and curvature library."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from .gradient_cache import cache_key
from .relevance import missing_query_labels


SETTINGS = {"projection_factor": 16, "factor_rank": 1, "factor_iterations": 8,
            "projection_seed": 0, "projection_normalization": "legacy",
            "curvature_rank": 2048, "oversample": 10, "power_iterations": 3,
            "curvature_mode": "truncated", "damping_multiplier": 0.5,
            "compute_dtype": "float32", "storage_dtype": "bfloat16",
            "batch_size": 4, "shard_size": 128}

FACTOR_KEYS = ("projection_factor", "factor_rank", "factor_iterations", "projection_seed",
               "projection_normalization", "compute_dtype", "storage_dtype", "batch_size", "shard_size")


def _library():
    from modules.lorif.src import lorif
    return lorif


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(value, indent=2), encoding="utf-8")
    pending.replace(path)


def _save_torch(path, value):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    torch.save(value, pending)
    pending.replace(path)


def select_modules(model):
    """Collect transformer attention/MLP weight branches, excluding LoRA matrices."""
    import torch
    projections = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    names = []
    for name, module in model.named_modules():
        logical = name.removesuffix(".base_layer")
        if (isinstance(module, torch.nn.Linear) and logical.rsplit(".", 1)[-1] in projections
                and "lora_" not in name):
            names.append(name)
    if not names:
        raise ValueError("LoRIF needs supported attention/MLP Linear modules (q/k/v/o/gate/up/down_proj)")
    return names


def pad_batch(samples, pad_id, padding_side):
    """Combine already-tokenized examples; padding never contributes to the loss."""
    import torch
    length = max(sample["input_ids"].shape[1] for sample in samples)
    result = {}
    for key, value in (("input_ids", pad_id), ("attention_mask", 0), ("labels", -100)):
        rows = []
        for sample in samples:
            missing = length - sample[key].shape[1]
            padding = (missing, 0) if padding_side == "left" else (0, missing)
            rows.append(torch.nn.functional.pad(sample[key], padding, value=value))
        result[key] = torch.cat(rows)
    return result


class ModelSession:
    """Keep one frozen quantized adapter model alive while indexing missing rows."""

    method_name = "LoRIF"

    def __init__(self, artifact_dir, metadata, device, settings):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        from src import TDA, Params

        print(f"Loading model for {self.method_name} weight gradients (adapter gradients cannot be reused)...", flush=True)
        self.device, self.settings = device, settings
        self.tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir))
        self.tokenizer.padding_side = metadata["padding_side"]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        options = {"revision": metadata["model_revision"]} if metadata.get("model_revision") else {}
        if metadata["load_in_4bit"]:
            options.update(quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
                device_map={"": str(device)})
        base = AutoModelForCausalLM.from_pretrained(metadata["model_name"], **options)
        self.model = PeftModel.from_pretrained(base, str(artifact_dir), is_trainable=False)
        if not metadata["load_in_4bit"]:
            self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_disable()
        self.model.enable_input_require_grads()
        self.tda = TDA(self.model, self.tokenizer, Params(), device=str(device))
        self.collector = self.make_collector()
        self.signature = {"specs": [spec.to_dict() for spec in self.collector.specs.values()],
                          "projection_sha256": self.collector.projection_sha256()}

    def make_collector(self):
        import torch
        settings = self.settings
        return _library().GradientCollector(
            self.model, select_modules(self.model), projection_factor=settings["projection_factor"],
            factor_rank=settings["factor_rank"], factor_iterations=settings["factor_iterations"],
            projection_seed=settings["projection_seed"],
            projection_normalization=settings["projection_normalization"], compute_dtype=torch.float32,
        )

    def extract(self, frame, response):
        import torch
        samples = [self.tda.get_tokenized_text(self.tokenizer,
                   {"prompts": row["prompts"], "response": row[response]}, device=self.device)
                   for _, row in frame.iterrows()]
        batch = pad_batch(samples, self.tokenizer.pad_token_id, self.tokenizer.padding_side)
        counts = (batch["labels"][:, 1:] != -100).sum(dim=1)
        if (counts == 0).any():
            raise ValueError(f"{self.method_name} example has no response tokens after truncation")
        self.collector.clear()
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(**batch)
        # HF loss is mean over valid response tokens; collector requires a token sum.
        loss = outputs.loss * counts.sum()
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite {self.method_name} response-token loss")
        loss.backward()
        return self.collector.pop()

    def close(self):
        self.collector.close()
        self.model.disable_input_require_grads()


class FactorIndex:
    def __init__(self, directory, manifest):
        self.directory, self.manifest = Path(directory), manifest
        self.specs = {spec["name"]: spec for spec in manifest["signature"]["specs"]}

    def path(self, name, shard):
        return self.directory / cache_key(name) / f"shard-{shard:05d}.pt"

    def batches(self, name, device="cpu"):
        import torch
        Factors = _library().Factors
        spec = self.specs[name]
        rank = self.manifest["identity"]["settings"]["factor_rank"]
        for number, shard in enumerate(self.manifest["shards"]):
            saved = torch.load(self.path(name, number), map_location="cpu", weights_only=True)
            factors = Factors(saved["left"], saved["right"])
            if (factors.batch_size != shard["end"] - shard["start"] or factors.rank != rank
                    or factors.shape != (spec["projected_input_dim"], spec["projected_output_dim"])
                    or not torch.isfinite(factors.left).all() or not torch.isfinite(factors.right).all()):
                raise ValueError(f"Invalid LoRIF factor shard: {self.path(name, number)}")
            yield factors.to(device=device, dtype=torch.float32)


def ensure_factors(directory, frame, response, identity, session_factory, progress_factory):
    """Commit all layers of each shard together; recover from the last complete shard."""
    import torch
    library = _library()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "index.json"
    try:
        manifest = _read(manifest_path)
        if manifest["identity"] != identity:
            raise ValueError("LoRIF factor identity mismatch")
        index = FactorIndex(directory, manifest)
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
        print(f"Reusing LoRIF {identity['kind']} factors: {completed}/{len(frame)} rows", flush=True)
        return FactorIndex(directory, manifest)
    session = session_factory()
    if manifest is None:
        manifest = {"identity": identity, "signature": session.signature, "shards": []}
    elif manifest["signature"] != session.signature:
        raise ValueError("LoRIF projection tensors or selected modules changed")
    index = FactorIndex(directory, manifest)
    settings = identity["settings"]
    with progress_factory(total=len(frame), initial=completed, desc=f"LoRIF {identity['kind']} factors") as progress:
        for start in range(completed, len(frame), settings["shard_size"]):
            stop = min(start + settings["shard_size"], len(frame))
            buffer = {name: [] for name in index.specs}
            for row in range(start, stop, settings["batch_size"]):
                batch_stop = min(row + settings["batch_size"], stop)
                values = session.extract(frame.iloc[row:batch_stop], response)
                if set(values) != set(buffer) or any(value.batch_size != batch_stop - row for value in values.values()):
                    raise ValueError("LoRIF collector returned inconsistent layer/row counts")
                for name, value in values.items():
                    if not torch.isfinite(value.left).all() or not torch.isfinite(value.right).all():
                        raise ValueError("Nonfinite LoRIF factors")
                    buffer[name].append(value.to(device="cpu", dtype=torch.bfloat16))
                progress.update(batch_stop - row)
            for name, batches in buffer.items():
                value = library.Factors(torch.cat([batch.left for batch in batches]),
                                        torch.cat([batch.right for batch in batches]))
                _save_torch(index.path(name, len(manifest["shards"])), {"left": value.left, "right": value.right})
            manifest["shards"].append({"start": start, "end": stop})
            _write(manifest_path, manifest)
    return index


def score_layer(train, query, name, settings, device):
    """Official streamed curvature and Woodbury score, with reused query projection."""
    import torch
    library = _library()
    spec = train.specs[name]
    shape = (spec["projected_input_dim"], spec["projected_output_dim"])
    batches = lambda: train.batches(name, device)
    curvature = library.fit_curvature(
        batches, shape, num_examples=train.manifest["shards"][-1]["end"],
        rank=settings["curvature_rank"], oversample=settings["oversample"],
        num_power_iterations=settings["power_iterations"], mode=settings["curvature_mode"],
        seed=settings["projection_seed"], device=device, damping=1.0,
    )
    # Damping does not affect the fitted singular subspace. Apply the release's
    # trace-mean policy after fitting, so zero-training-gradient layers score zero.
    trace = curvature.spectrum_trace
    rows = query.manifest["shards"][-1]["end"]
    columns = train.manifest["shards"][-1]["end"]
    if trace == 0:
        return torch.zeros(rows, columns, dtype=torch.float64).numpy()
    curvature = replace(curvature, damping=settings["damping_multiplier"] * trace / (shape[0] * shape[1]),
                        damping_policy="trace-mean")
    parts = list(query.batches(name, device))
    queries = library.Factors(torch.cat([part.left for part in parts]), torch.cat([part.right for part in parts]))
    projected = library.project_factors(queries, curvature.right_vectors)
    scaled = projected / (curvature.damping + curvature.singular_values.square())
    result = torch.empty(rows, columns, dtype=torch.float64)
    offset = 0
    for factors in batches():
        train_projected = library.project_factors(factors, curvature.right_vectors)
        direct = library.factor_inner_products(queries, factors)
        scores = ((direct - projected @ train_projected.T) / curvature.damping
                  + scaled @ train_projected.T)
        if not torch.isfinite(scores).all():
            raise ValueError(f"Nonfinite LoRIF influence scores for {name}")
        result[:, offset:offset + factors.batch_size] = scores.double().cpu()
        offset += factors.batch_size
    return result.numpy()


def score_indexes(directory, train, query, identity, settings, device, progress_factory):
    """Atomically checkpoint the cumulative score and layer count in one file."""
    import numpy as np
    if train.manifest["signature"] != query.manifest["signature"]:
        raise ValueError("LoRIF source/query projections and layer specifications must match")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "scores_progress.npz"
    names = list(train.specs)
    shape = (query.manifest["shards"][-1]["end"], train.manifest["shards"][-1]["end"])
    scores, completed = np.zeros(shape, dtype=np.float64), 0
    try:
        with np.load(path, allow_pickle=False) as saved:
            if json.loads(str(saved["identity"])) != identity:
                raise ValueError("LoRIF score identity changed")
            scores, completed = saved["scores"], int(saved["completed_layers"])
        if scores.shape != shape or scores.dtype != np.float64 or not np.isfinite(scores).all() or not 0 <= completed <= len(names):
            raise ValueError("Invalid LoRIF score progress")
    except FileNotFoundError:
        pass
    if completed == len(names):
        print("Reusing completed LoRIF influence scores", flush=True)
        return scores
    with progress_factory(total=len(names), initial=completed, desc="LoRIF curvature/scoring layers") as progress:
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


def run_lorif(artifact_dir, source_path, query_path, *, output_dir="output", device="cuda",
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
        raise ValueError("LoRIF source fingerprint differs from the training dataset")
    sources, queries = dataset.to_pandas(), pd.read_csv(query_path)
    for frame, columns in ((sources, ["prompts", "response", "label"]),
                           (queries, ["prompts", "expected_response", "label"])):
        if not set(columns) <= set(frame.columns) or frame.empty or frame[columns].isna().any().any():
            raise ValueError(f"Prepared LoRIF data needs nonempty values for {columns}")
    sources, queries = _limit_by_label(sources, source_limit), _limit_by_label(queries, query_limit)
    if topk[-1] > len(sources) or missing_query_labels(sources, queries):
        raise ValueError("Retrieval cutoffs and query labels must have supporting selected sources")
    target = torch.device(device)
    if metadata["load_in_4bit"] and target.type != "cuda":
        raise ValueError("LoRIF requires CUDA for this 4-bit training checkpoint")
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
            identity = {"version": "lorif-weight-factors-v1", "model": model_identity,
                        "settings": {key: SETTINGS[key] for key in FACTOR_KEYS},
                        "kind": kind, "rows_sha256": hashlib.sha256(rows.encode()).hexdigest(), "rows": len(frame)}
            identities.append(identity)
            directory = root / "lorif" / "factors" / cache_key(identity)
            indexes.append(ensure_factors(directory, frame, response, identity, session_factory, tqdm))
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
    score_identity = {"version": "lorif-release-woodbury-v1", "factors": identities,
                      "signature": indexes[0].manifest["signature"], "settings": dict(SETTINGS)}
    score_dir = root / "lorif" / "scores" / cache_key(score_identity)
    scores = score_indexes(score_dir, indexes[0], indexes[1], score_identity, SETTINGS, target, tqdm)
    result_dir = root / "lorif" / f"epoch-{float(metadata['trained_epochs']):g}"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "complete.json").unlink(missing_ok=True)
    with (result_dir / "scores.npz.tmp").open("wb") as stream:
        np.savez_compressed(stream, scores=scores, source_indices=sources.index.to_numpy(dtype=np.int64),
                            query_indices=queries.index.to_numpy(dtype=np.int64))
    (result_dir / "scores.npz.tmp").replace(result_dir / "scores.npz")
    manifest = {"method": "lorif", "epoch": metadata["trained_epochs"], "settings": dict(SETTINGS),
        "model": model_identity, "checkpoint_selection": "selected-completed-epoch",
        "source_factors": str(indexes[0].directory.relative_to(root)),
        "query_factors": str(indexes[1].directory.relative_to(root)), "score_cache": str(score_dir.relative_to(root)),
        "formula": "sum_layers query_gradient.T @ (lambda I + V diag(sigma**2) V.T)^-1 @ source_gradient",
        "damping": "release: 0.5 * training_gradient_spectrum_trace / projected_dimension",
        "curvature_persistence": "none; cumulative scores checkpointed after each layer"}
    _write(result_dir / "feature_metadata.json", manifest)
    from src import TDAEvaluation
    metrics = pd.DataFrame([result.__dict__ for result in TDAEvaluation.compute_from_scores(
        sources, queries, scores, topk, "lorif")])
    metrics.to_excel(result_dir / "metrics.xlsx", index=False)
    _write(result_dir / "complete.json", {"score_version": score_identity["version"]})
    print(f"Saved LoRIF factors, scores, and metrics: {result_dir}", flush=True)
    return metrics
