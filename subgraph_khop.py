# subgraph_khop.py
import torch, dgl
import numpy as np
from typing import Dict, List, Tuple

def _neighbors_one_by_one(g: dgl.DGLGraph, vs, direction='out') -> torch.Tensor:
    """逐点取邻居，统一返回到 g.device、int64。"""
    if torch.is_tensor(vs):
        vs_list = vs.reshape(-1).tolist()
    else:
        vs_list = [int(vs)]
    outs = []
    for v in vs_list:
        n = g.successors(v) if direction == 'out' else g.predecessors(v)
        if not torch.is_tensor(n):
            n = torch.as_tensor(n, dtype=torch.long)
        outs.append(n.to(g.device).long())
    if len(outs) == 0:
        return torch.empty(0, dtype=torch.long, device=g.device)
    return torch.unique(torch.cat(outs))


def khop_in_subgraph_compat(g: dgl.DGLGraph, nids, k: int) -> Tuple[dgl.DGLGraph, torch.Tensor]:
    """
    与 dgl.khop_in_subgraph 等价的兼容实现：
    返回 (子图, 中心点在子图中的位置 idx)。
    """
    dev = g.device
    if torch.is_tensor(nids):
        seeds = nids.reshape(-1).to(dev).long()
        center = int(seeds[0].item())
    else:
        center = int(nids)
        seeds = torch.tensor([center], device=dev, dtype=torch.long)

    visited = seeds.clone()
    frontier = seeds.clone()
    for _ in range(int(k)):
        try:
            succ = _neighbors_one_by_one(g, frontier, 'out')
            pred = _neighbors_one_by_one(g, frontier, 'in')
        except Exception:
            cg, fr = g.to('cpu'), frontier.to('cpu')
            succ = _neighbors_one_by_one(cg, fr, 'out').to(dev)
            pred = _neighbors_one_by_one(cg, fr, 'in').to(dev)
        nbrs = torch.unique(torch.cat([succ, pred, frontier]))
        visited = torch.unique(torch.cat([visited, nbrs]))
        frontier = nbrs

    sub_nodes = visited.to(dev).long()
    # 构子图（必要时回退 CPU 索引）
    try:
        subg = g.subgraph(sub_nodes)
    except Exception:
        subg = g.subgraph(sub_nodes.to('cpu'))

    # 中心点 idx（考虑 DGL 不同版本）
    try:
        orig = subg.ndata[dgl.NID]
    except Exception:
        orig = subg.ndata.get('_ID')
    if torch.is_tensor(orig):
        idx = (orig.cpu() == center).nonzero(as_tuple=True)[0]
    else:
        pos = np.where(np.asarray(orig) == center)[0]
        idx = torch.tensor([int(pos[0]) if len(pos) else 0])

    return subg, idx.to(subg.device)


def get_khop_candidates(
        G: dgl.DGLGraph,
        anchors: torch.Tensor,
        k_hop_edge: int,
        deg_per_node: int,
        fallback_random: bool = True
) -> Dict[int, List[int]]:
    """
    基于 k-hop 子图为每个 anchor 生成候选邻居列表：
    - 候选 = 子图中的所有节点（去掉 anchor 本身），再去重；
    - 若候选数 < deg_per_node，使用 2-hop / 随机节点补齐（与原逻辑一致，确保不越界）。
    返回：{anchor_id: [cand_id, ...]}
    """
    device = G.device
    N = int(G.num_nodes())
    cand_dict: Dict[int, List[int]] = {}

    for a in anchors.tolist():
        a = int(a)
        if k_hop_edge and k_hop_edge > 0:
            try:
                # 优先用官方 API，失败再走兼容
                sg, _ = dgl.khop_in_subgraph(G, a, k_hop_edge)
            except Exception:
                sg, _ = khop_in_subgraph_compat(G, a, k_hop_edge)

            # 子图里的原图节点 ID
            try:
                orig = sg.ndata[dgl.NID]
            except Exception:
                orig = sg.ndata.get('_ID')
            if torch.is_tensor(orig):
                cands = orig.to(device).long().tolist()
            else:
                cands = list(map(int, np.asarray(orig)))

            # 去掉 anchor 自身
            cands = [int(v) for v in cands if int(v) != a]
        else:
            # 全图：用 1/2-hop 做更稳健的“局部候选”
            out1 = _neighbors_one_by_one(G, a, 'out').tolist()
            in1  = _neighbors_one_by_one(G, a, 'in').tolist()
            hop1 = list(set([int(v) for v in (out1 + in1) if int(v) != a]))
            hop2 = []
            for v in hop1:
                hop2.extend(_neighbors_one_by_one(G, v, 'out').tolist())
                hop2.extend(_neighbors_one_by_one(G, v, 'in').tolist())
            cands = list(set([int(v) for v in (hop1 + hop2) if int(v) != a]))

        # 兜底：数量不足时再补
        if len(cands) < int(deg_per_node):
            if fallback_random:
                # 优先补“不与 a 相连”的点以提升隐蔽性
                nei_a = set(_neighbors_one_by_one(G, a, 'out').tolist() + _neighbors_one_by_one(G, a, 'in').tolist())
                pool = list(set(range(N)) - nei_a - {a} - set(cands))
                if len(pool) > 0:
                    need = int(deg_per_node) - len(cands)
                    idx = torch.randperm(len(pool), device=device)[:max(0, need)].tolist()
                    cands.extend([pool[i] for i in idx])
            # 若仍不足，重复填充以免后续 top-k 越界（与你原 attacker 对齐）
            if len(cands) == 0:
                cands = [a] * int(deg_per_node)
            elif len(cands) < int(deg_per_node):
                rep = (int(deg_per_node) + len(cands) - 1) // len(cands)
                cands = (cands * rep)[:int(deg_per_node)]

        cand_dict[a] = cands
    return cand_dict
def print_khop_subgraph_stats(G, anchors, k_hop_edge: int):
    sizes = []
    total = len(anchors)
    print(f"\n[Subgraph Stats] total anchors/subgraphs = {total}")
    for i, a in enumerate(anchors.tolist()):
        a = int(a)
        if k_hop_edge and k_hop_edge > 0:
            try:
                sg, _ = dgl.khop_in_subgraph(G, a, k_hop_edge)
            except Exception:
                sg, _ = khop_in_subgraph_compat(G, a, k_hop_edge)
            n_nodes = int(sg.num_nodes())
        else:
            # khop_edge==0：用 1/2-hop 近邻集合构一个“局部子图”
            hop1 = list(set(_neighbors_one_by_one(G, a, 'out').tolist()
                            + _neighbors_one_by_one(G, a, 'in').tolist()))
            hop2 = []
            for v in hop1:
                hop2.extend(_neighbors_one_by_one(G, v, 'out').tolist())
                hop2.extend(_neighbors_one_by_one(G, v, 'in').tolist())
            nodes = list(set([a] + hop1 + hop2))
            try:
                sg = G.subgraph(torch.tensor(nodes, device=G.device, dtype=torch.long))
            except Exception:
                sg = G.subgraph(torch.tensor(nodes, dtype=torch.long))
            n_nodes = int(sg.num_nodes())

        sizes.append(n_nodes)
        # print(f"  - Subgraph {i:03d} (anchor={a}): |V| = {n_nodes}")

    if sizes:
        print(f"[Summary] min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)/total:.2f}\n")