from __future__ import annotations

from typing import Dict

import dgl
import networkx as nx
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
    Return the same number of anchors as TopologySensitiveAnchorLocalizer.
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
def score_vulnerability_bridge_community(
    G: dgl.DGLGraph,
    candidate_nodes: torch.Tensor,
    topk: int,
    community_method: str = "louvain",
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Community-bridge anchor scoring.

    It scores nodes by cross-community bridge strength.
    Only candidate_nodes can be returned.

    community_method:
        "louvain" or "greedy"

    Output format matches TopologySensitiveAnchorLocalizer.score_topology_sensitivity().
    """
    device = candidate_nodes.device
    adj = dgl_to_scipy_undirected_no_loop(G)
    n = adj.shape[0]

    try:
        nx_g = nx.from_scipy_sparse_array(adj)
    except AttributeError:
        nx_g = nx.from_scipy_sparse_matrix(adj)

    if nx_g.number_of_nodes() == 0:
        scores = torch.zeros(n, dtype=torch.float32, device=device)
        return rank_from_scores_exact(scores, candidate_nodes, topk)

    method = str(community_method).lower()

    if method == "louvain" and hasattr(nx.community, "louvain_communities"):
        communities = list(
            nx.community.louvain_communities(
                nx_g,
                seed=int(seed),
            )
        )
    else:
        communities = list(nx.community.greedy_modularity_communities(nx_g))

    comm_id = np.full(n, -1, dtype=np.int64)
    for cid, comm in enumerate(communities):
        for v in comm:
            comm_id[int(v)] = int(cid)

    raw_scores = np.zeros(n, dtype=np.float32)

    for v in range(n):
        deg_v = nx_g.degree[v]
        if deg_v <= 0:
            continue

        cv = comm_id[v]
        cross = 0
        same = 0

        for u in nx_g.neighbors(v):
            if comm_id[int(u)] != cv:
                cross += 1
            else:
                same += 1

        cross_ratio = float(cross) / float(max(deg_v, 1))
        cross_log = np.log1p(float(cross))

        # bridge tendency + bridge capacity
        raw_scores[v] = 0.7 * cross_ratio + 0.3 * cross_log

    max_score = float(raw_scores.max())
    if max_score > 1e-12:
        raw_scores = raw_scores / max_score

    scores = torch.tensor(raw_scores, dtype=torch.float32, device=device)
    return rank_from_scores_exact(scores, candidate_nodes, topk)
