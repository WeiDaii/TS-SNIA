"""Topology-sensitive anchor localization from temporary local edge-deletion probes."""
from pathlib import Path
from typing import Optional
import torch
import dgl

class TopologySensitiveAnchorLocalizer:
    def __init__(self, surrogate, device, alpha: float = 0.6, drop_top: int = 5,
                 drop_strategy: str = "highest_degree", seed: int = 0):
        self.surrogate = surrogate
        self.device = device
        self.alpha = alpha
        self.drop_top = int(drop_top)  # ← 新增：每个节点“本地拓扑探测”要切掉的关键邻居条数
        self.drop_strategy = str(drop_strategy)
        self.seed = int(seed)

    @torch.inference_mode()
    def score_topology_sensitivity(self, G: dgl.DGLGraph, x: torch.Tensor, y: torch.Tensor,
                            node_idx: torch.Tensor, topk: Optional[int] = None) -> torch.Tensor:
        """
        用临时局部删边探测计算拓扑敏感度，并选择得分最高的 topk 个锚点。
        drop_top 控制每次探测最多删除多少条邻居边，topk 控制返回的锚点数。
        """
        G = G.to(self.device)
        x = x.to(self.device)
        y = y.to(self.device)

        node_idx = node_idx.to(self.device).view(-1).long()
        M = node_idx.shape[0]

        # 先对 node_idx 里的节点逐一打分
        local_scores = torch.zeros(M, device=self.device, dtype=torch.float32)
        for i, u in enumerate(node_idx.tolist()):
            s = self._local_confidence_drop(G, x, int(u), drop_top=self.drop_top)
            local_scores[i] = s

        # 把分数放回“全图”向量（未参与评估的节点设为 -inf，保证后续 topk 只会从 node_idx 中选）
        N = G.num_nodes()
        scores_full = torch.full((N,), float('-inf'), device=self.device)
        scores_full[node_idx] = local_scores

        # 选择 Top-K 作为锚点（全局ID）
        K = M if topk is None else min(int(topk), M)
        top_nodes = torch.topk(scores_full, k=K, largest=True).indices  # 全局ID

        return {
            "scores": scores_full,  # [N]，其余为 -inf
            "top_nodes": top_nodes  # [K]，按易损度降序
        }

    @torch.inference_mode()
    def score_probe_metric(
        self, G: dgl.DGLGraph, x: torch.Tensor, node_idx: torch.Tensor,
        metric: str, topk: Optional[int] = None, cache_path: str = "",
    ):
        """Score shared local topology probes with the selected sensitivity metric."""
        nodes = node_idx.to(self.device).view(-1).long()
        cache = None
        path = Path(cache_path) if cache_path else None
        if path is not None and path.exists():
            candidate = torch.load(path, map_location="cpu")
            if (int(candidate.get("num_nodes", -1)) == int(G.num_nodes())
                    and int(candidate.get("drop_top", -1)) == self.drop_top
                    and candidate.get("drop_strategy") == self.drop_strategy
                    and torch.equal(candidate["node_idx"], nodes.cpu())):
                cache = candidate
        cache_hit = cache is not None
        if cache is None:
            before, after = [], []
            for node in nodes.tolist():
                p0, p1 = self._local_probe_distributions(G, x, int(node), self.drop_top)
                before.append(p0.cpu())
                after.append(p1.cpu())
            cache = {
                "num_nodes": int(G.num_nodes()), "node_idx": nodes.cpu(),
                "drop_top": self.drop_top, "drop_strategy": self.drop_strategy,
                "before": torch.stack(before), "after": torch.stack(after),
            }
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(cache, path)

        p0 = cache["before"].to(self.device).clamp_min(1e-12)
        p1 = cache["after"].to(self.device).clamp_min(1e-12)
        if metric == "confidence_drop":
            if getattr(self.surrogate, "attack_output_mode", "full_probability") == "hard_label":
                scores = (p0.argmax(dim=1) != p1.argmax(dim=1)).float()
            else:
                scores = p0.max(dim=1).values - p1.max(dim=1).values
        elif metric == "entropy_increase":
            scores = -(p1 * p1.log()).sum(dim=1) + (p0 * p0.log()).sum(dim=1)
        elif metric == "margin_drop":
            margin0 = torch.topk(p0, k=min(2, p0.shape[1]), dim=1).values
            margin1 = torch.topk(p1, k=min(2, p1.shape[1]), dim=1).values
            m0 = margin0[:, 0] - margin0[:, -1]
            m1 = margin1[:, 0] - margin1[:, -1]
            scores = m0 - m1
        elif metric == "js_divergence":
            mid = 0.5 * (p0 + p1)
            scores = 0.5 * ((p0 * (p0.log() - mid.log())).sum(dim=1) +
                            (p1 * (p1.log() - mid.log())).sum(dim=1))
        else:
            raise ValueError(f"Unknown topology-probe metric: {metric}")

        scores_full = torch.full((G.num_nodes(),), float('-inf'), device=self.device)
        scores_full[nodes] = scores
        k = nodes.numel() if topk is None else min(int(topk), int(nodes.numel()))
        return {"scores": scores_full, "top_nodes": torch.topk(scores_full, k=k).indices,
                "probe_cache_hit": cache_hit}

    # ---------------- 内部：兼容获取 eid 的小工具 ----------------
    def _safe_edge_ids(self, G: dgl.DGLGraph, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """
        只为存在的 (src, dst) 取 eid。不存在的边自动过滤。
        兼容不同 DGL 版本的行为。
        """
        # 设备
        try:
            dev = G.device
        except Exception:
            # 从 ndata 推断
            dev = next((v.device for v in G.ndata.values() if isinstance(v, torch.Tensor)), torch.device("cpu"))

        src = torch.as_tensor(src, device=dev)
        dst = torch.as_tensor(dst, device=dev)
        if src.dim() == 0 and dst.dim() >= 1:
            src = src.repeat(dst.shape[0])

        # 过滤不存在的边
        exists = G.has_edges_between(src, dst)
        if exists.ndim > 0:
            if exists.sum() == 0:
                return torch.zeros(0, dtype=torch.int64, device=dev)
            src = src[exists]
            dst = dst[exists]
        elif not bool(exists):
            return torch.zeros(0, dtype=torch.int64, device=dev)

        # 取 eid
        eids = G.edge_ids(src, dst)
        if isinstance(eids, tuple):   # 旧版本某些情况会返回 (eids,)
            eids = eids[0]
        return eids

    @torch.inference_mode()
    def _select_cut(self, G, x, node: int, nbrs: torch.Tensor, count: int):
        if self.drop_strategy == "highest_degree":
            order = torch.topk(G.out_degrees(nbrs), k=count, largest=True).indices
        elif self.drop_strategy == "lowest_degree":
            order = torch.topk(G.out_degrees(nbrs), k=count, largest=False).indices
        elif self.drop_strategy == "random":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed * 1000003 + int(node))
            order = torch.randperm(int(nbrs.numel()), generator=generator)[:count].to(nbrs.device)
        elif self.drop_strategy == "feature_dissimilar":
            center = x[int(node)].view(1, -1)
            center = center / center.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            nearby = x[nbrs]
            nearby = nearby / nearby.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            similarity = (nearby * center).sum(dim=-1)
            order = torch.topk(similarity, k=count, largest=False).indices
        else:
            raise ValueError(f"Unknown anchor drop strategy: {self.drop_strategy}")
        return nbrs[order]

    @torch.inference_mode()
    def _local_confidence_drop(self, G: dgl.DGLGraph, x: torch.Tensor, node: int, drop_top: int = 5) -> float:
        """
        局部拓扑探测置信度下降：切掉该节点最“关键”的若干邻居边后，其预测置信度下降量。
        “关键”邻居这里仍按你原逻辑（比如度/注意力/ppMI 等）提供的 cut。
        这里我们只修正 eid 获取与删边的实现，避免 dgl.edge_ids 报错。
        """
        # 1) 原始置信度
        hard_label = getattr(self.surrogate, "attack_output_mode", "full_probability") == "hard_label"
        query_node = torch.tensor([node], device=x.device)
        if hard_label:
            p0_label = int(self.surrogate.predict_distribution(G, x, nodes=query_node).argmax(dim=-1).item())
        else:
            p0 = self.surrogate.predict_proba(G, x, nodes=query_node).item()

        # 2) 选择要切的邻居（保持你原来的选择逻辑；这里给个稳妥的默认：按度高的邻居）
        nbrs = G.successors(node)     # 出邻居；如需无向/入邻，可换 predecessors 或 union
        if nbrs.numel() == 0:
            return 0.0
        # 简单按邻居度排序（可替换为你原先的打分）
        topk = min(drop_top, int(nbrs.shape[0]))
        cut = self._select_cut(G, x, node, nbrs, topk)

        # 3) 取 eid 并删边
        eids = self._safe_edge_ids(G, torch.full_like(cut, node), cut)
        if eids.numel() == 0:
            return 0.0
        H = dgl.remove_edges(G, eids)

        # 4) 删边后的置信度
        if hard_label:
            p1_label = int(self.surrogate.predict_distribution(H, x, nodes=query_node).argmax(dim=-1).item())
            return float(p0_label != p1_label)
        p1 = self.surrogate.predict_proba(H, x, nodes=query_node).item()

        # 5) 局部拓扑探测置信度下降（下降越大越脆弱）
        return max(0.0, p0 - p1)

    @torch.inference_mode()
    def _local_probe_distributions(self, G, x, node: int, drop_top: int = 5):
        query_node = torch.tensor([node], device=x.device)
        p0 = self.surrogate.predict_distribution(G, x, nodes=query_node).squeeze(0)
        nbrs = G.successors(node)
        if nbrs.numel() == 0:
            return p0, p0.clone()
        count = min(int(drop_top), int(nbrs.numel()))
        cut = self._select_cut(G, x, node, nbrs, count)
        eids = self._safe_edge_ids(G, torch.full_like(cut, node), cut)
        if eids.numel() == 0:
            return p0, p0.clone()
        changed = dgl.remove_edges(G, eids)
        p1 = self.surrogate.predict_distribution(changed, x, nodes=query_node).squeeze(0)
        return p0, p1
