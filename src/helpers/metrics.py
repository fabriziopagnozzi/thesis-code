import numpy as np
from numpy.typing import NDArray


def fac_cov_score(
    selected_indices: NDArray[np.intp],
    sim_matrix: NDArray[np.float32],
) -> float:
    """fac(S) = (1/|D|) * sum_i max_{j in S} cos(e_i, e_j)"""
    if len(selected_indices) == 0:
        return 0.0
    coverage = sim_matrix[:, selected_indices].max(axis=1)
    return float(coverage.mean())


def avg_cos(
    selected_indices: NDArray[np.intp],
    sim_to_query: NDArray[np.float32],
) -> float:
    """AvgCos = (1/|S|) * sum_{j in S} cos(e_q, e_j)"""
    if len(selected_indices) == 0:
        return 0.0
    return float(sim_to_query[selected_indices].mean())


def jaccard(
    indices_a: NDArray[np.intp],
    indices_b: NDArray[np.intp],
) -> float:
    set_a = set(indices_a.tolist())
    set_b = set(indices_b.tolist())
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)
