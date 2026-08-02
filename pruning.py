# -*- coding: utf-8 -*-
"""
注入后剪边：去掉最“可疑”的边（低同配/低相似度）
提升隐蔽性与面对防御的鲁棒性
"""
import torch, dgl
import torch.nn.functional as F

@torch.no_grad()
def prune_edges_by_dissim(G, injected_ids, ratio=0.3, feat_key='feat'):
    feats = G.ndata[feat_key]
    removed = 0
    for nid in injected_ids:
        # 只看 nid 的出边
        out_nbrs = dgl.sampling.sample_neighbors(G, torch.tensor([nid], device=G.device), -1).edges()[1]
        if out_nbrs.numel() <= 1: continue
        sims = F.cosine_similarity(feats[nid].unsqueeze(0).expand_as(feats[out_nbrs]), feats[out_nbrs], dim=1)
        k = int(round(ratio * out_nbrs.numel()))
        if k<=0: continue
        worst = torch.topk(-sims, k=k).indices
        to_cut = out_nbrs[worst]
        eids = dgl.edge_ids(G, torch.full_like(to_cut, nid), to_cut)
        G.remove_edges(eids)
        removed += int(to_cut.numel())
    return removed
