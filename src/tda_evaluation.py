import numpy as np
from typing import Literal, List, Optional, Dict, Any
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from peft import PeftModel
from tqdm import tqdm

from .utils import Result, Params
from .tda import TDA


class TDAEvaluation:
    def __init__(
        self,
        model,
        tokenizer,
        device: Literal["cuda", "cpu"],
        params: Params,
        display: bool = False,
    ):
        self.model: PeftModel = model
        self.device = torch.device(device)
        self.params = params
        self.tokenizer = tokenizer
        self.model.to(self.device)
        self.tda = TDA(
            self.model,
            tokenizer,
            params,
            self.device,
            display,
        )

    def _sim(self):
        return {
            "dot": lambda a, b: np.dot(a, b),
            "cosine": lambda a, b: np.dot(a, b)
            / (np.linalg.norm(a) * np.linalg.norm(b)),
        }

    def compute_attribution(
        self,
        source_data,
        eval_data,
        source_vector,
        eval_vector,
        topk,
        metric,
        method,
    ):
        scores = np.array([
            [self._sim()[metric](vector, source) for source in source_vector]
            for vector in eval_vector
        ])
        return self.compute_from_scores(source_data, eval_data, scores, topk, method)

    @staticmethod
    def compute_from_scores(source_data, eval_data, scores, topk, method):
        """Evaluate an already aggregated query-by-source score matrix."""
        if scores.shape != (len(eval_data), len(source_data)) or not np.isfinite(scores).all():
            raise ValueError("Attribution scores must be finite and match query/source counts")
        topk = sorted(topk, reverse=True)
        if not topk or min(topk) <= 0 or max(topk) > len(source_data):
            raise ValueError("Retrieval cutoffs must be positive and within source count")
        y_scores, y_trues, precision, recall, rrs = (
            {k: [] for k in topk},
            {k: [] for k in topk},
            {k: [] for k in topk},
            {k: [] for k in topk},
            {k: [] for k in topk},
        )
        for i, sim_score in enumerate(scores):
            y_trues_map = np.array(
                [
                    int(eval_data["label"].tolist()[i] == label)
                    for label in source_data["label"]
                ]
            )
            top_max_k_indices = np.argsort(sim_score)[-topk[0] :]
            sorted_idx = np.argsort(sim_score)[::-1]
            for k in topk:
                top_k_indices = top_max_k_indices[-k:]
                precision[k].append(y_trues_map[top_k_indices].sum() / k)
                correct_total = y_trues_map.sum()
                recall[k].append(
                    y_trues_map[top_k_indices].sum() / correct_total
                    if correct_total > 0
                    else 0
                )
                rr = 0
                for rank, idx in enumerate(sorted_idx):
                    if y_trues_map[idx] == 1:
                        rr = 1.0 / (rank + 1)
                        break
                rrs[k].append(rr)
                for j in top_k_indices:
                    y_scores[k].append(sim_score[j])
                    y_trues[k].append(y_trues_map[j])
        results = [
            Result(
                top_k=k,
                auPRC=average_precision_score(y_trues[k], y_scores[k]),
                P_at_k=np.mean(precision[k]),
                R_at_k=np.mean(recall[k]),
                MRR=np.mean(rrs[k]),
                method=method,
            )
            for k in topk
        ]
        return results

    def tracing(
        self,
        source_dataset: pd.DataFrame,
        eval_dataset: pd.DataFrame,
        method: Literal["rept", "tracin", "tracein", "rapid_in", "less", "lorif"],
        topk: List[int],
    ):
        source_signals, testing_signals = self.extract_features(
            source_dataset, eval_dataset, method
        )
        metric = "dot" if method in ("tracin", "tracein") else "cosine"
        return self.compute_attribution(
            source_dataset, eval_dataset, source_signals, testing_signals,
            topk, metric, method,
        )

    def extract_features(self, source_dataset, eval_dataset, method):
        """Compute source/query vectors separately from ranking metrics."""
        source_signals, testing_signals = [], []
        for i, row in tqdm(source_dataset.iterrows(), total=source_dataset.shape[0]):
            source_signals.append(
                self.tda.trace(
                    method=method,
                    prompt=row["prompts"],
                    expected_response=row["response"],
                )
            )
        for i, row in tqdm(eval_dataset.iterrows(), total=eval_dataset.shape[0]):
            testing_signals.append(
                self.tda.trace(
                    method=method,
                    prompt=row["prompts"],
                    expected_response=row["expected_response"],
                )
            )
        return source_signals, testing_signals
