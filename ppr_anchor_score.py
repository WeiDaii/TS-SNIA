from __future__ import annotations

from typing import Dict

import dgl
import numpy as np
import scipy.sparse as sp
import torch


def dgl_to_scipy_undirected_no_loop(G: dgl.DGLGraph) -> sp.csr_matrix:
    """
    Convert DGLGraph to scipy CSR adjacency.
    The returned graph is undirected and has no self-loops.
    """
    G_cpu = G.cpu()
    src, dst = G_cpu.edges()
    src = src.numpy()
    dst = dst.numpy()

    n = int(G_cpu.num_nodes())
    data = np.ones_like(src, dtype=np.float32)

    adj = sp.csr_matrix((data, (src, dst)), shape=(n, n))
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def rank_from_scores_exact(
    scores: torch.Tensor,
    candidate_nodes: torch.Tensor,
    topk: int,
) -> Dict[str, torch.Tensor]:
    """
    Return exactly the same number of candidate anchors as CausalScorer.

    Output:
        {
            "top_nodes": Tensor[num_return],
            "scores": Tensor[num_nodes]
        }

    where:
        num_return = min(topk, candidate_nodes.numel())
    """
    device = candidate_nodes.device
    candidate_nodes = candidate_nodes.long().to(device)
    scores = scores.to(device)

    num_return = min(int(topk), int(candidate_nodes.numel()))
    num_return = max(1, num_return)

    safe_scores = scores.clone()
    safe_scores[torch.isnan(safe_scores)] = -1e18
    safe_scores[torch.isposinf(safe_scores)] = 1e18
    safe_scores[torch.isneginf(safe_scores)] = -1e18

    cand_scores = safe_scores[candidate_nodes]

    # Deterministic tie-breaking: larger score first, smaller node id first.
    rank_values = cand_scores - candidate_nodes.float() * 1e-12
    order = torch.argsort(rank_values, descending=True)

    top_nodes = candidate_nodes[order[:num_return]]

    if top_nodes.numel() != num_return:
        raise RuntimeError(
            f"Expected {num_return} candidate anchors, got {top_nodes.numel()}."
        )

    return {
        "top_nodes": top_nodes,
        "scores": scores,
    }


@torch.no_grad()
def score_vulnerability_ppr(
    G: dgl.DGLGraph,
    candidate_nodes: torch.Tensor,
    topk: int,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    PPR-based anchor scoring.

    candidate_nodes:
        Usually correct_test_idx. Only these nodes can be returned.

    topk:
        Must be candidate_topk, not final topk.

    Output format is compatible with CausalScorer.score_vulnerability().
    """
    device = candidate_nodes.device
    adj = dgl_to_scipy_undirected_no_loop(G)
    n = adj.shape[0]

    seeds = candidate_nodes.detach().cpu().numpy().astype(np.int64)
    seeds = seeds[(seeds >= 0) & (seeds < n)]

    if seeds.size == 0:
        scores = torch.zeros(n, dtype=torch.float32, device=device)
        return rank_from_scores_exact(scores, candidate_nodes, topk)

    deg = np.asarray(adj.sum(axis=1)).reshape(-1)
    deg[deg <= 0] = 1.0

    # Row-stochastic transition matrix.
    P = sp.diags(1.0 / deg, format="csr").dot(adj)

    personalize = np.zeros(n, dtype=np.float64)
    personalize[seeds] = 1.0 / float(seeds.size)

    p = personalize.copy()
    alpha = float(alpha)

    for _ in range(int(max_iter)):
        p_next = (1.0 - alpha) * personalize + alpha * P.T.dot(p)
        if np.abs(p_next - p).sum() < float(tol):
            p = p_next
            break
        p = p_next

    scores = torch.tensor(p, dtype=torch.float32, device=device)
    return rank_from_scores_exact(scores, candidate_nodes, topk)