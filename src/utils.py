from dataclasses import dataclass
from typing import Literal, Optional, Dict, List 


@dataclass
class Result:
    top_k: int
    auPRC: float
    P_at_k: float
    R_at_k: float
    MRR: float
    method: str

    def __repr__(self):
        return (
            f"Result(top_k={self.top_k}, "
            f"method='{self.method}', "
            f"auPRC={self.auPRC:.4f}, "
            f"P@k={self.P_at_k:.4f}, "
            f"R@k={self.R_at_k:.4f}, "
            f"MRR={self.MRR:.4f})"
        )

@dataclass
class Params:
    """Parameters for TDA in TDAEvaluation for each of use across self.traces evaluations for all methods."""
    layer_norm: bool = True
    layer: int = -1

