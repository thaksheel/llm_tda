"""Single-model TRAK with the DsDm language-model log-odds output adaptation."""

import hashlib
import json
import math
from pathlib import Path

from .gradient_cache import cache_key
from .relevance import missing_query_labels


SETTINGS = {"projection_dim": 2048, "projection_seed": 0, "projection_block": 128,
            "projection_type": "rademacher", "model_id": 0, "batch_size": 16,
            "dtype": "float32", "output": "response-token-sum-log-odds-v1",
            "q": "mean-response-token-one-minus-probability",
            "solver": "thin-svd-moore-penrose-v1", "rtol": "max(n,k)*float32-eps"}

FEATURE_KEYS = ("projection_dim", "projection_seed", "projection_block", "projection_type",
                "model_id", "batch_size", "dtype", "output", "q")


def _write(path, value):
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(value, indent=2), encoding="utf-8")
    pending.replace(path)


def _save(path, **arrays):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    with pending.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    pending.replace(path)


def log_odds_output(logits, labels):
    """Causally shifted, response-masked token log-odds sum and mean (1-p)."""
    import torch
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or logits.shape[0] != 1 or logits.shape[-1] < 2:
        raise ValueError("TRAK expects one token sequence, matching labels, and at least two vocabulary classes")
    targets = labels[0, 1:]
    valid = targets != -100
    if not valid.any():
        raise ValueError("TRAK example has no response tokens after truncation")
    targets = targets[valid]
    values = logits[0, :-1][valid].float()
    correct = values.gather(1, targets.unsqueeze(1)).squeeze(1)
    other = values.scatter(1, targets.unsqueeze(1), -torch.inf).logsumexp(dim=1)
    margins = correct - other
    output = margins.sum()
    q = torch.sigmoid(-margins).mean().detach()
    if not torch.isfinite(output) or not torch.isfinite(q):
        raise ValueError("Nonfinite TRAK log-odds output or source confidence")
    return output, q


class ModelSession:
    def __init__(self, artifact_dir, metadata, device, settings):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        from modules import BasicProjector, ProjectionType
        from src import TDA, Params
        from .run_tda import _tracin_parameters

        print("Loading model for TRAK log-odds gradients...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir))
        self.tokenizer.padding_side = metadata["padding_side"]
        options = {"revision": metadata["model_revision"]} if metadata.get("model_revision") else {}
        if metadata["load_in_4bit"]:
            options.update(quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
                device_map={"": str(device)})
        base = AutoModelForCausalLM.from_pretrained(metadata["model_name"], **options)
        self.model = PeftModel.from_pretrained(base, str(artifact_dir), is_trainable=True)
        if not metadata["load_in_4bit"]:
            self.model.to(device)
        self.model.eval()
        self.model.config.use_cache = False
        self.names, self.dimensions = _tracin_parameters(self.model, artifact_dir)
        parameters = dict(self.model.named_parameters())
        self.parameters = [parameters[name] for name in self.names]
        self.tda = TDA(self.model, self.tokenizer, Params(), device=str(device))
        self.device, self.settings = device, settings
        self.projector = BasicProjector(
            grad_dim=self.dimensions, proj_dim=settings["projection_dim"], seed=settings["projection_seed"],
            proj_type=ProjectionType.rademacher, device=torch.device(device),
            block_size=settings["projection_block"], dtype=torch.float32,
        )

    def extract(self, frame, response):
        import torch
        gradients, confidence = [], []
        for _, row in frame.iterrows():
            inputs = self.tda.get_tokenized_text(self.tokenizer,
                {"prompts": row["prompts"], "response": row[response]}, device=self.device)
            self.model.zero_grad(set_to_none=True)
            forward_inputs = {key: value for key, value in inputs.items() if key != "labels"}
            output, q = log_odds_output(self.model(**forward_inputs).logits, inputs["labels"])
            values = torch.autograd.grad(output, self.parameters, allow_unused=True)
            vector = torch.cat([(value if value is not None else torch.zeros_like(parameter)).reshape(-1).float()
                                for value, parameter in zip(values, self.parameters, strict=True)])
            if not torch.isfinite(vector).all():
                raise ValueError("Nonfinite TRAK model-output gradients")
            gradients.append(vector.detach())
            confidence.append(q)
        projected = self.projector.project(torch.stack(gradients), model_id=self.settings["model_id"])
        projected = projected / math.sqrt(self.dimensions)
        if not torch.isfinite(projected).all():
            raise ValueError("Nonfinite TRAK projected gradients")
        return projected.cpu().numpy(), torch.stack(confidence).cpu().numpy()

    def close(self):
        self.projector.free_memory()


def ensure_features(directory, frame, response, identity, session_factory, progress_factory):
    """Commit projected gradients and Q together after each flushed batch."""
    import numpy as np
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / "gradients.npy", directory / "q.npy"]
    shapes = [(len(frame), identity["settings"]["projection_dim"]), (len(frame),)]
    progress_path = directory / "progress.json"
    matrices, completed = [], 0
    try:
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = saved["completed_rows"]
        if saved["identity"] != identity or type(completed) is not int or not 0 <= completed <= len(frame):
            raise ValueError("TRAK feature identity or row progress changed")
        for path, shape in zip(paths, shapes):
            matrix = np.load(path, mmap_mode="r" if completed == len(frame) else "r+", allow_pickle=False)
            if matrix.shape != shape or matrix.dtype != np.float32 or matrix.offset + matrix.nbytes != path.stat().st_size:
                raise ValueError("Invalid TRAK feature file")
            matrices.append(matrix)
    except (OSError, ValueError, KeyError, TypeError, EOFError):
        matrices.clear()
        matrix = None
        completed = 0
    if not matrices:
        progress_path.unlink(missing_ok=True)
        matrices = [np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
                    for path, shape in zip(paths, shapes)]
        _write(progress_path, {"identity": identity, "completed_rows": 0})
    if completed == len(frame):
        print(f"Reusing TRAK {identity['kind']} projected gradients: {completed}/{len(frame)} rows", flush=True)
        return matrices
    session = session_factory()
    with progress_factory(total=len(frame), initial=completed, desc=f"TRAK {identity['kind']} log-odds gradients") as progress:
        batch = identity["settings"]["batch_size"]
        for start in range(completed, len(frame), batch):
            stop = min(start + batch, len(frame))
            gradients, q = session.extract(frame.iloc[start:stop], response)
            if (gradients.shape != (stop - start, shapes[0][1]) or q.shape != (stop - start,)
                    or not np.isfinite(gradients).all() or not np.isfinite(q).all()
                    or (q < 0).any() or (q > 1).any()):
                raise ValueError("Invalid TRAK projected gradients or confidence values")
            matrices[0][start:stop], matrices[1][start:stop] = gradients, q
            for matrix in matrices:
                matrix.flush()
            _write(progress_path, {"identity": identity, "completed_rows": stop})
            progress.update(stop - start)
    _write(directory / "feature_metadata.json", identity)
    return matrices


def fit_kernel(directory, gradients, identity, device):
    """Stable unregularized least squares via thin SVD; no explicit Gram inverse."""
    import numpy as np
    import torch
    from zipfile import BadZipFile
    path = Path(directory) / "kernel.npz"
    try:
        with np.load(path, allow_pickle=False) as saved:
            if json.loads(str(saved["identity"])) != identity:
                raise ValueError("TRAK kernel identity changed")
            u, inverse, vh = saved["u"], saved["inverse"], saved["vh"]
        if (u.shape != (len(gradients), len(inverse)) or vh.shape != (len(inverse), gradients.shape[1])
                or any(not np.isfinite(value).all() for value in (u, inverse, vh)) or (inverse <= 0).any()):
            raise ValueError("Invalid TRAK kernel factorization")
        print("Reusing TRAK training kernel", flush=True)
        return u, inverse, vh
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
        pass
    print("Fitting TRAK training kernel with thin SVD...", flush=True)
    x = torch.from_numpy(np.array(gradients, copy=True)).to(device)
    if not torch.isfinite(x).all():
        raise ValueError("Nonfinite TRAK source features")
    u, singular, vh = torch.linalg.svd(x, full_matrices=False)
    rtol = max(x.shape) * torch.finfo(x.dtype).eps
    keep = singular > (singular[0] * rtol)
    u, inverse, vh = u[:, keep].cpu().numpy(), singular[keep].reciprocal().cpu().numpy(), vh[keep].cpu().numpy()
    _save(path, u=u, inverse=inverse, vh=vh, identity=json.dumps(identity, sort_keys=True), rtol=rtol)
    print(f"TRAK kernel retained {len(inverse)}/{min(x.shape)} singular directions", flush=True)
    return u, inverse, vh


def score_features(directory, source, query, kernel, identity, device):
    import numpy as np
    import torch
    from zipfile import BadZipFile
    path = Path(directory) / "scores.npz"
    shape = (len(query[0]), len(source[0]))
    try:
        with np.load(path, allow_pickle=False) as saved:
            if json.loads(str(saved["identity"])) != identity:
                raise ValueError("TRAK score identity changed")
            scores = saved["scores"]
        if scores.shape != shape or scores.dtype != np.float64 or not np.isfinite(scores).all():
            raise ValueError("Invalid TRAK scores")
        print("Reusing completed TRAK scores", flush=True)
        return scores
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
        pass
    print("Scoring TRAK kernel affinities and source confidence weights...", flush=True)
    u, inverse, vh = [torch.from_numpy(np.array(value, copy=True)).to(device) for value in kernel]
    q_source = torch.from_numpy(np.array(source[1], copy=True)).to(device)
    scores = np.empty(shape, dtype=np.float64)
    from tqdm import tqdm
    with tqdm(total=len(query[0]), desc="TRAK scoring", unit="query") as progress:
        for start in range(0, len(query[0]), 128):
            values = torch.from_numpy(np.array(query[0][start:start + 128], copy=True)).to(device)
            result = (((values @ vh.T) * inverse) @ u.T) * q_source
            if not torch.isfinite(result).all():
                raise ValueError("Nonfinite TRAK scores")
            scores[start:start + len(values)] = result.double().cpu().numpy()
            progress.update(len(values))
    _save(path, scores=scores, identity=json.dumps(identity, sort_keys=True))
    return scores


def run_trak(artifact_dir, source_path, query_path, *, output_dir="output", device="cuda",
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
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    dataset = load_from_disk(str(source_path))
    if dataset._fingerprint != metadata["source_fingerprint"]:
        raise ValueError("TRAK source fingerprint differs from the training dataset")
    sources, queries = dataset.to_pandas(), pd.read_csv(query_path)
    for frame, columns in ((sources, ["prompts", "response", "label"]),
                           (queries, ["prompts", "expected_response", "label"])):
        if not set(columns) <= set(frame.columns) or frame.empty or frame[columns].isna().any().any():
            raise ValueError(f"Prepared TRAK data needs nonempty values for {columns}")
    sources, queries = _limit_by_label(sources, source_limit), _limit_by_label(queries, query_limit)
    if topk[-1] > len(sources) or missing_query_labels(sources, queries):
        raise ValueError("Retrieval cutoffs and query labels must have supporting selected sources")
    target = torch.device(device)
    if metadata["load_in_4bit"] and target.type != "cuda":
        raise ValueError("TRAK requires CUDA for this 4-bit checkpoint")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model_identity = {"name": metadata["model_name"], "revision": metadata.get("model_revision"),
        "load_in_4bit": metadata["load_in_4bit"], "max_length": 512, "padding_side": metadata["padding_side"],
        "device": str(target), "parameter_groups": json.loads((artifact_dir / "optimizer_parameters.json").read_text()),
        "artifact_sha256": {path.name: _sha256_file(path) for path in artifact_dir.iterdir() if path.name in (
            "adapter_model.safetensors", "adapter_model.bin", "adapter_config.json", "tokenizer.json",
            "tokenizer_config.json", "chat_template.jinja")}}
    session, identities, features, directories = [], [], [], []
    def session_factory():
        if not session:
            session.append(ModelSession(artifact_dir, metadata, target, SETTINGS))
        return session[0]
    try:
        for kind, frame, response in (("source", sources, "response"), ("query", queries, "expected_response")):
            rows = frame[["prompts", response, "label"]].to_json(orient="split", force_ascii=False)
            identity = {"version": "trak-lm-output-gradients-v1", "model": model_identity,
                "settings": {key: SETTINGS[key] for key in FEATURE_KEYS}, "kind": kind,
                "rows_sha256": hashlib.sha256(rows.encode()).hexdigest(), "rows": len(frame)}
            identities.append(identity)
            directory = root / "trak" / "features" / cache_key(identity)
            directories.append(directory)
            features.append(ensure_features(directory, frame, response, identity, session_factory, tqdm))
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
    kernel_identity = {"version": SETTINGS["solver"], "source": identities[0], "rtol": SETTINGS["rtol"]}
    kernel_dir = root / "trak" / "kernels" / cache_key(kernel_identity)
    kernel = fit_kernel(kernel_dir, features[0][0], kernel_identity, target)
    score_identity = {"version": "trak-single-model-q-weighted-v1", "features": identities, "kernel": kernel_identity}
    score_dir = root / "trak" / "scores" / cache_key(score_identity)
    scores = score_features(score_dir, features[0], features[1], kernel, score_identity, target)
    result_dir = root / "trak" / f"epoch-{float(metadata['trained_epochs']):g}"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "complete.json").unlink(missing_ok=True)
    _save(result_dir / "scores.npz", scores=scores, source_indices=sources.index.to_numpy(dtype=np.int64),
          query_indices=queries.index.to_numpy(dtype=np.int64))
    _write(result_dir / "feature_metadata.json", {"method": "trak", "settings": dict(SETTINGS),
        "model": model_identity, "epoch": metadata["trained_epochs"], "num_models": 1,
        "checkpoint_selection": "selected-completed-epoch; no trajectory ensemble",
        "source_features": str(directories[0].relative_to(root)), "query_features": str(directories[1].relative_to(root)),
        "kernel_cache": str(kernel_dir.relative_to(root)), "score_cache": str(score_dir.relative_to(root)),
        "formula": "Phi_query @ pinv(Phi_source) @ diag(Q_source)", "retained_rank": len(kernel[1])})
    from src import TDAEvaluation
    metrics = pd.DataFrame([result.__dict__ for result in TDAEvaluation.compute_from_scores(sources, queries, scores, topk, "trak")])
    metrics.to_excel(result_dir / "metrics.xlsx", index=False)
    _write(result_dir / "complete.json", {"score_version": score_identity["version"]})
    print(f"Saved TRAK features, kernel, scores, and metrics: {result_dir}", flush=True)
    return metrics
