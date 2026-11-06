# -*- coding: utf-8 -*-
"""
TS-SNIA: Topology-Sensitivity-Aware Sparse Node Injection Attack.
Fixed-surrogate topology probes, anchor-guided construction, and sparse refinement.
"""

# import sys, io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import argparse, os, math, random, json, time
from pathlib import Path
from config import build_args, normalize_dataset_name, normalize_victim_name
from data import load_original_graph, apply_lcc_to_loaded_dgl_data, largest_connected_component_nodes_dgl
from ppr_anchor_score import score_vulnerability_ppr
from community_anchor_score import score_vulnerability_bridge_community
import numpy as np
import torch
from data import load_dgl_dataset, clone_graph_with_data, set_seed
from models import GNNModelWrapper
from anchor_localization import TopologySensitiveAnchorLocalizer
from utils import device_of, to_cpu
import dgl
# from attacker_sparse import build_surrogate_from_victim, stage4_sparse_optimize_per_subgraph
# ---------- Robust neighbor choosing helpers ----------
from injection_construction import sample_anchor_neighbors, construct_cross_block_injection,construct_internal_injected_edges
# from attacker_sparse import SparseInjectionAttack     # 稀疏矩阵可微优化攻击器（上一条消息给你的实现）
from subgraph_khop import print_khop_subgraph_stats, get_khop_candidates
from utils import extract_anchor_subgraphs_from_G, merge_anchor_maps, batch_subgraphs_from_injid_to_cands
# from causal_rl_stage1 import build_causal_rl_stage1_pack
import torch
# from subgraph_sparse_attack_final import run_subgraph_sparse_attack
# from innov_attack import run_innov_attack
# from subgraph_sparse_attack_modify import run_subgraph_sparse_attack_nes_modify
# from subgraph_sparse_attack import run_subgraph_sparse_attack_nes
# === 兼容预测输出为 [N,C] or [N] 的小工具 ===

def extract_clean_part_from_adv(G_adv: dgl.DGLGraph, X_adv: torch.Tensor, N0: int):
    """
    从对抗图中抽取前 N0 个原始节点诱导子图（OO 部分），并返回对应特征切片。
    注意：node_subgraph 会重排节点ID为 [0..N0-1]，原ID保存在 G_oo.ndata[dgl.NID]。
    """
    dev = G_adv.device
    keep = torch.arange(N0, device=dev)
    G_oo = dgl.node_subgraph(G_adv, keep)        # 只含原始节点与其间的边
    X_oo = X_adv[:N0].clone()                    # 原始节点特征部分
    return G_oo, X_oo

def _edges_to_set(G: dgl.DGLGraph, undirected: bool):
    u, v = G.edges()
    uu, vv = u.detach().cpu().tolist(), v.detach().cpu().tolist()
    if undirected:
        return set((min(a,b), max(a,b)) for a,b in zip(uu, vv))
    else:
        return set((a, b) for a,b in zip(uu, vv))


def count_injected_edge_types(G: dgl.DGLGraph, original_nodes: int):
    """Count unique undirected injected-to-original and injected-internal edges."""
    external = 0
    internal = 0
    for u, v in _edges_to_set(G, undirected=True):
        if u == v:
            continue
        u_injected = u >= original_nodes
        v_injected = v >= original_nodes
        if u_injected and v_injected:
            internal += 1
        elif u_injected or v_injected:
            external += 1
    return external, internal

import numpy as np
import torch


@torch.no_grad()
def spnia_simple_undirected_degree(G, num_nodes=None):
    """
    SpNIA-style degree statistics.
    Treat u->v and v->u as one undirected edge.
    """
    if num_nodes is None:
        num_nodes = int(G.num_nodes())
    else:
        num_nodes = int(num_nodes)

    src, dst = G.edges()
    src = src.detach().cpu().numpy()
    dst = dst.detach().cpu().numpy()

    edge_set = set()
    for u, v in zip(src, dst):
        u = int(u)
        v = int(v)
        if u == v:
            continue
        edge_set.add((min(u, v), max(u, v)))

    deg = np.zeros(num_nodes, dtype=np.int64)

    for u, v in edge_set:
        if u < num_nodes and v < num_nodes:
            deg[u] += 1
            deg[v] += 1

    return torch.from_numpy(deg), len(edge_set)


@torch.no_grad()
def spnia_degree_hist_similarity(G_clean, G_adv, N0):
    """
    SpNIA Eq. (16):
        Similarity(H(G), H'(G')) =
        1 - 1/n * sqrt(sum_i (h_i - h'_i)^2)

    Higher is better.
    """
    deg_clean, E_clean = spnia_simple_undirected_degree(G_clean, num_nodes=N0)
    deg_adv, E_adv = spnia_simple_undirected_degree(G_adv, num_nodes=G_adv.num_nodes())

    deg_clean_np = deg_clean.cpu().numpy()
    deg_adv_np = deg_adv.cpu().numpy()

    max_deg = int(max(deg_clean_np.max(), deg_adv_np.max()))
    h_clean = np.bincount(deg_clean_np, minlength=max_deg + 1).astype(np.float64)
    h_adv = np.bincount(deg_adv_np, minlength=max_deg + 1).astype(np.float64)

    sim = 1.0 - (np.sqrt(np.sum((h_clean - h_adv) ** 2)) / float(N0))

    return {
        "degree_similarity": float(sim),
        "degree_similarity_percent": float(sim * 100.0),
        "clean_undirected_edges": int(E_clean),
        "adv_undirected_edges": int(E_adv),
        "clean_avg_degree": float(deg_clean_np.mean()),
        "adv_avg_degree": float(deg_adv_np.mean()),
    }


@torch.no_grad()
def spnia_unnoticeability_eval(
    G_clean,
    G_adv,
    feats_clean,
    feats_adv,
    N0,
    inj_ids,
    epsilon,
    eps=1e-12,
):
    """
    Only evaluate SpNIA-style unnoticeability.
    Does not modify graph or features.
    """
    device = feats_adv.device

    if not isinstance(inj_ids, torch.Tensor):
        inj_ids = torch.tensor(list(inj_ids), dtype=torch.long, device=device)
    else:
        inj_ids = inj_ids.to(device).long().view(-1)

    N0 = int(N0)
    m = int(inj_ids.numel())

    deg_clean, E_clean = spnia_simple_undirected_degree(G_clean, num_nodes=N0)
    deg_adv, E_adv = spnia_simple_undirected_degree(G_adv, num_nodes=G_adv.num_nodes())

    deg_clean = deg_clean.to(device)
    deg_adv = deg_adv.to(device)

    orig_avg_deg = float(deg_clean.float().mean().item())

    if m > 0:
        inj_avg_deg = float(deg_adv[inj_ids].float().mean().item())
        inj_max_deg = int(deg_adv[inj_ids].max().item())
    else:
        inj_avg_deg = 0.0
        inj_max_deg = 0

    X_orig = feats_clean[:N0].to(device)
    X_inj = feats_adv[inj_ids].to(device) if m > 0 else feats_adv[:0]

    orig_feat_l0 = (X_orig.abs() > eps).float().sum(dim=1)
    orig_avg_feat_l0 = float(orig_feat_l0.mean().item())

    if m > 0:
        inj_feat_l0 = (X_inj.abs() > eps).float().sum(dim=1)
        inj_avg_feat_l0 = float(inj_feat_l0.mean().item())
        inj_max_feat_l0 = int(inj_feat_l0.max().item())
    else:
        inj_avg_feat_l0 = 0.0
        inj_max_feat_l0 = 0

    degree_sim = spnia_degree_hist_similarity(G_clean, G_adv, N0)

    inj_ratio = float(m) / float(N0)
    configured_node_budget = max(1, int(round(float(epsilon) * N0)))

    result = {
        "injection_ratio": inj_ratio,
        "injection_ratio_percent": inj_ratio * 100.0,
        "configured_node_budget": configured_node_budget,
        "node_budget_ok": m <= configured_node_budget,

        "original_avg_degree": orig_avg_deg,
        "injected_avg_degree": inj_avg_deg,
        "injected_max_degree": inj_max_deg,
        "degree_budget_ok": inj_avg_deg <= orig_avg_deg + 1e-6,

        "original_avg_feature_l0": orig_avg_feat_l0,
        "injected_avg_feature_l0": inj_avg_feat_l0,
        "injected_max_feature_l0": inj_max_feat_l0,
        "feature_budget_ok": inj_avg_feat_l0 <= orig_avg_feat_l0 + 1e-6,

        **degree_sim,
    }

    print("=" * 80)
    print("[SpNIA-style Unnoticeability Evaluation]")
    print(f"Injection ratio:          {result['injection_ratio_percent']:.2f}%")
    print(f"Node budget satisfied:    {'YES' if result['node_budget_ok'] else 'NO'}")
    print(f"Original avg degree:      {result['original_avg_degree']:.4f}")
    print(f"Injected avg degree:      {result['injected_avg_degree']:.4f}")
    print(f"Degree budget satisfied:  {'YES' if result['degree_budget_ok'] else 'NO'}")
    print(f"Original avg feat L0:     {result['original_avg_feature_l0']:.4f}")
    print(f"Injected avg feat L0:     {result['injected_avg_feature_l0']:.4f}")
    print(f"Feature budget satisfied: {'YES' if result['feature_budget_ok'] else 'NO'}")
    print(f"Degree similarity:        {result['degree_similarity_percent']:.2f}%")
    print("=" * 80)

    return result

def compare_graphs_ref_vs_advOO(
    G_ref: dgl.DGLGraph, X_ref: torch.Tensor,
    G_advOO: dgl.DGLGraph, X_advOO: torch.Tensor,
    undirected: bool = False, rtol=0.0, atol=0.0
):
    """
    对比“参考干净图(全原始节点)” vs “从 G_adv 索引出的原始部分(advOO)”：
    - 检查边集合是否完全一致（原始-原始之间不能被改动）
    - 检查原始节点特征是否一致（允许 rtol/atol 容差）
    """
    # 1) 边集合
    E_ref   = _edges_to_set(G_ref, undirected)
    E_advOO = _edges_to_set(G_advOO, undirected)
    removed = E_ref - E_advOO
    added   = E_advOO - E_ref
    edges_ok = (len(removed) == 0 and len(added) == 0)
    if edges_ok:
        print("[VERIFY][Edges] 原始-原始(OO) 边完全一致。")
    else:
        print(f"[VERIFY][Edges] 发现改动：被删={len(removed)}, 被加={len(added)}")
        if removed: print("  删掉的示例：", list(removed)[:10])
        if added:   print("  新增的示例：", list(added)[:10])

    # 2) 特征
    if X_ref.shape != X_advOO.shape:
        print(f"[VERIFY][X] 形状不一致: ref={tuple(X_ref.shape)}, advOO={tuple(X_advOO.shape)}")
        feats_ok = False
    else:
        diff = (X_ref - X_advOO).abs()
        feats_ok = bool(torch.all(diff <= (atol + rtol * X_ref.abs())).item())
        if feats_ok:
            print("[VERIFY][X] 原始节点特征一致（未被扰动）。")
        else:
            changed_rows = (diff.sum(dim=1) > 0).nonzero(as_tuple=True)[0].tolist()
            print(f"[VERIFY][X] 有 {len(changed_rows)} 个原始节点的特征发生改变，示例行：{changed_rows[:10]}")

    ok = edges_ok and feats_ok
    print(f"[VERIFY] 结论：{'通过' if ok else '不通过'}")
    return ok

def predict_labels_1d(surrogate, G, feats) -> torch.Tensor:
    with torch.no_grad():
        out = surrogate.predict(G, feats)
    if out.dim() == 2:           # [N, C] 浮点分数
        return out.argmax(dim=1)
    elif out.dim() == 1:         # [N] 整型标签
        return out.to(torch.long)
    else:
        raise ValueError(f"predict() output shape not supported: {tuple(out.shape)}")


def build_labels_with_injected(G, labels, anchor_to_injid, device):
    """
    基于原始 labels（长度=N_old），构造长度=N_current 的全图标签：
    - 原图节点：保持原始 labels
    - 恶性节点：标签 = 其对应锚点的标签

    参数
    ----
    G : dgl.DGLGraph          # 已经完成注入后的图（Stage-2b 之后）
    labels : torch.Tensor     # 原始图的真实标签，形状 [N_old]
    anchor_to_injid : dict    # {anchor_id: inj_id}
    device : torch.device

    返回
    ----
    labels_full : torch.Tensor  # 形状 [N_current]，恶性节点标签已赋值
    """
    N_old = int(labels.size(0))
    N_cur = int(G.num_nodes())
    labels_full = torch.full((N_cur,), -1, dtype=labels.dtype, device=device)

    # 1) 先写回原图节点的真实标签
    labels_full[:N_old] = labels.to(device)

    # 2) 给每个恶性节点赋“与锚点相同”的标签
    for a, inj in anchor_to_injid.items():
        a = int(a); inj = int(inj)
        labels_full[inj] = labels_full[a]  # 等同于 labels[a]，且在 device 上

    return labels_full








def diversify_anchor_selection(
    G, scores, candidate_nodes, topk, method, diversity_lambda=0.5, seed=0
):
    """Select anchors from a fixed score vector while varying only spatial diversity."""
    if method == "top_m" or scores is None:
        return candidate_nodes[torch.argsort(scores[candidate_nodes], descending=True)[:topk]] if scores is not None else candidate_nodes[:topk]

    ranked = candidate_nodes[torch.argsort(scores[candidate_nodes], descending=True)]
    if method in {"distance_greedy", "mmr_local"}:
        selected = []
        blocked = torch.zeros(G.num_nodes(), dtype=torch.bool, device=candidate_nodes.device)
        normalized = scores.clone()
        finite = torch.isfinite(normalized)
        lo, hi = normalized[finite].min(), normalized[finite].max()
        normalized[finite] = (normalized[finite] - lo) / (hi - lo).clamp_min(1e-12)
        available = torch.zeros_like(blocked)
        available[candidate_nodes] = True
        while len(selected) < topk and bool(available.any()):
            pool = available.nonzero(as_tuple=True)[0]
            if method == "distance_greedy":
                eligible = pool[~blocked[pool]]
                if eligible.numel() == 0:
                    eligible = pool
                chosen = eligible[torch.argmax(normalized[eligible])]
            else:
                penalty = blocked[pool].float() * float(diversity_lambda)
                chosen = pool[torch.argmax(normalized[pool] - penalty)]
            node = int(chosen.item())
            selected.append(node)
            available[node] = False
            neighbours = torch.unique(torch.cat([G.successors(node), G.predecessors(node)])).to(blocked.device)
            blocked[neighbours] = True
        return torch.tensor(selected, dtype=torch.long, device=candidate_nodes.device)

    if method == "community_stratified":
        import networkx as nx
        cpu = G.to("cpu")
        src, dst = cpu.edges()
        graph = nx.Graph()
        graph.add_nodes_from(range(cpu.num_nodes()))
        graph.add_edges_from(zip(src.tolist(), dst.tolist()))
        if hasattr(nx.community, "louvain_communities"):
            communities = nx.community.louvain_communities(graph, seed=int(seed))
        else:
            communities = nx.community.greedy_modularity_communities(graph)
        candidate_set = set(int(v) for v in candidate_nodes.detach().cpu().tolist())
        queues = []
        for community in communities:
            nodes = [v for v in community if int(v) in candidate_set]
            nodes.sort(key=lambda v: float(scores[int(v)].item()), reverse=True)
            if nodes:
                queues.append(nodes)
        queues.sort(key=lambda q: float(scores[int(q[0])].item()), reverse=True)
        selected = []
        offset = 0
        while len(selected) < topk and queues:
            next_queues = []
            for queue in queues:
                if offset < len(queue) and len(selected) < topk:
                    selected.append(int(queue[offset]))
                if offset + 1 < len(queue):
                    next_queues.append(queue)
            queues = next_queues
            offset += 1
        return torch.tensor(selected, dtype=torch.long, device=candidate_nodes.device)

    raise ValueError(f"Unknown anchor_selection: {method}")




def k_hop_nodes_from_pair(adj_bin: torch.Tensor, src: int, dst: int, k: int):
    """
    adj_bin: (N,N) 0/1 稀疏 or 稠密张量（torch.float/torch.bool 都可）
    返回：子图节点下标（1D LongTensor，去重）
    """
    N = adj_bin.size(0)
    dev = adj_bin.device
    # 邻接布尔化
    A = (adj_bin > 0).to(torch.bool)
    seeds = torch.zeros(N, dtype=torch.bool, device=dev)
    seeds[src] = True; seeds[dst] = True
    frontier = seeds.clone()
    visited = seeds.clone()
    for _ in range(int(k)):
        # 无向扩展：in 或 out 都并上
        neib_out = (A & frontier.view(1, -1)).any(dim=1)   # 有出边到 frontier 的点
        neib_in  = (A & frontier.view(-1, 1)).any(dim=0)   # 有入边自 frontier 的点
        nbrs = neib_out | neib_in | frontier
        new_frontier = nbrs & (~visited)
        visited = visited | new_frontier
        frontier = new_frontier
        if not frontier.any(): break
    return visited.nonzero(as_tuple=True)[0].long()

def build_in_out_masks(N: int, sub_idx: torch.Tensor, device):
    """
    生成“入度视图/出度视图”的遮罩矩阵：
    - 入度视图：保留所有“指向 sub_idx 节点”的边 => mask_in[:, sub_idx] = 1
    - 出度视图：保留所有“由 sub_idx 节点发出”的边 => mask_out[sub_idx, :] = 1
    """
    mask_in  = torch.zeros((N, N), dtype=torch.float32, device=device)
    mask_out = torch.zeros((N, N), dtype=torch.float32, device=device)
    mask_in[:,  sub_idx] = 1.0
    mask_out[sub_idx, :] = 1.0
    return mask_in, mask_out

def clamp01(x):  # 与 SAVAGE 一致的 [0,1] 投影
    return torch.clamp(x, 0.0, 1.0)

def bce_link_loss(model, X, A, src, dst, target, device):
    """
    适配你现有 SAVAGE 的链路预测：decode(z, [[dst],[src]])
    target: 标量 0/1 (float)
    """
    z = model(X, A)  # (1,N,dim) or (N,dim)
    out = model.decode(z.squeeze(0), torch.tensor([[dst],[src]], device=device).long()).view(-1)
    crit = torch.nn.BCEWithLogitsLoss()
    tgt = torch.tensor([float(target)], device=device, dtype=torch.float32)
    return crit(out, tgt), torch.sigmoid(out).detach().item()

# 若你是“节点分类”任务，把上面函数换成：
# def ce_node_loss(model, X, A, y, node_idx, target_labels):
#     logits = model(X, A)  # (N,C)
#     return F.cross_entropy(logits[node_idx], target_labels[node_idx]), logits[node_idx].softmax(-1).detach()



def _parse_target_ids(s):
    if not s: return None
    try:
        ids = [int(t.strip()) for t in s.split(',') if t.strip()!='']
        return sorted(set(ids))
    except:
        return None
def feature_range_norm(feats: torch.Tensor, method: str = "maxabs"):
    """
    计算 'Feature range (norm)'。
    feats: [N, F] 的 torch.Tensor（可在 GPU）
    method: "maxabs"（默认，按列 |x| 的最大值做缩放），
            也可选 "zscore" 或 "minmax"（见注释）。
    返回:
        feats_norm: 归一化后的特征 (同 dtype/device)
        fmin, fmax: 归一化后在全图上的最小/最大值（float）
    """
    assert feats.dim() == 2, "feats should be [N, F]"
    X = feats

    if method == "maxabs":              # 每一列除以该列的 max(|x|)
        scale = X.abs().max(dim=0).values.clamp(min=1e-12)
        Xn = X / scale
    elif method == "zscore":            # 每列 (x-mean)/std，然后统计整体 min/max
        mu  = X.mean(dim=0)
        std = X.std(dim=0, unbiased=False).clamp(min=1e-12)
        Xn = (X - mu) / std
    elif method == "minmax":            # 每列线性映射到 [-1, 1]
        xmin = X.min(dim=0).values
        xmax = X.max(dim=0).values
        denom = (xmax - xmin).clamp(min=1e-12)
        Xn = (X - xmin) / denom        # -> [0,1]
        Xn = Xn * 2 - 1                 # -> [-1,1]
    else:
        raise ValueError(f"unknown method: {method}")

    # 统计归一化后的全局范围
    fmin = float(Xn.min().detach().cpu())
    fmax = float(Xn.max().detach().cpu())
    return Xn, fmin, fmax
def run_single(args=None):
    experiment_started = time.perf_counter()
    args = build_args() if args is None else args
    if args.experiment_variant == 'no_structure_optimization':
        args.structure_mode = 'none'
    if not 0.0 < args.external_budget_ratio <= 1.0:
        raise ValueError('--external_budget_ratio must be in (0, 1]')
    if not 0.0 < args.query_budget_fraction <= 1.0:
        raise ValueError('--query_budget_fraction must be in (0, 1]')
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # 1) 加载数据与受害模型
    # G, feats, labels, train_idx, val_idx, test_idx = load_dgl_dataset(args.dataset, device)
    G, feats, labels, train_idx, val_idx, test_idx = load_original_graph(args.dataset, device, args)

    if args.use_lcc:
        G, feats, labels, train_idx, val_idx, test_idx = apply_lcc_to_loaded_dgl_data(
            G=G,
            feats=feats,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            device=device,
        )

    # Keep the data split and probe perturbations fixed through --seed while
    # allowing reviewer experiments to isolate surrogate initialization.
    model_seed = int(args.seed if args.model_seed is None else args.model_seed)
    set_seed(model_seed)

    print(len(train_idx),len(val_idx),len(test_idx))
    C = int(labels.max().item() + 1)
    print(f"Class num: {C}")
    surrogate = GNNModelWrapper(model_name=args.victim, in_dim=feats.shape[1],
                    n_classes=int(labels.max().item()+1),
                    lr=args.lr, wd=args.wd, args=args, device=device)

    print("\n[Stage 0] Train fixed surrogate on clean graph…")
    print('注入的是:{}网络'.format(args.victim))
    clean_initialization_counts = {"model_calls": 0, "node_predictions": 0}
    def count_clean_initialization(_module, _inputs, output):
        clean_initialization_counts["model_calls"] += 1
        if torch.is_tensor(output) and output.ndim > 0:
            clean_initialization_counts["node_predictions"] += int(output.shape[0])
    clean_init_hook = surrogate.model.register_forward_hook(count_clean_initialization)
    try:
        clean = surrogate.fit_eval(G, feats, labels, train_idx, val_idx, test_idx,
                                epochs=args.epochs, patience=args.patience, verbose=True,
                                fixed_epochs=args.fixed_surrogate_epochs)
    finally:
        clean_init_hook.remove()
    print(f"Clean test acc: {clean['test_acc']*100:.2f}%")
    print("注入了{}比例的节点".format(args.injection_ratio))
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        attack_memory_baseline = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        attack_memory_baseline = 0
    attack_resource_started = time.perf_counter()

    # 2) 拓扑敏感探测打分与锚点定位
    sensitivity_scorer = TopologySensitiveAnchorLocalizer(surrogate=surrogate, device=device, alpha=0.6,
                           drop_top=args.anchor_drop_top,
                           drop_strategy=args.anchor_drop_strategy, seed=args.seed)
    print("\n[Stage 1] Localizing topology-sensitive anchors…")
    # vul_scores = cscorer.score_vulnerability(G, feats, labels, test_idx, topk= min(G.num_nodes()*args.inj_node_budget, int(test_idx.shape[0])))
    all_idx = torch.arange(G.num_nodes(), device=device, dtype=torch.long)
    
    surrogate.query_stage = 'anchor_probe'
    if args.use_pseudo:
        with torch.no_grad():
            clean_pseudo_labels = surrogate.predict(G, feats)
        pseudo_labels = clean_pseudo_labels  # 关键：取 argmax 得到 1D 标签
    else:
        pseudo_labels = labels

    # 与 TCNIA 对齐：优先在干净模型已正确分类的原始节点中选择 anchor。
    # 若没有正确分类节点，则回退到全体节点，避免空候选集导致实验中断。
    # candidate_nodes = all_idx[(clean_pred[all_idx] == labels[all_idx]).nonzero(as_tuple=True)[0]]
    candidate_nodes = all_idx
    if candidate_nodes.numel() == 0:
        candidate_nodes = all_idx

    topk = max(1, int(round(G.num_nodes() * args.injection_ratio)))
    # topk = min(topk, int(candidate_nodes.numel()))

    if args.query_budget_fraction < 1.0:
        # Nested, deterministic probe subsets make the 25/50/75/100% curves
        # comparable while retaining the fixed 5% injected-node budget.
        generator = torch.Generator(device='cpu')
        generator.manual_seed(args.seed + 104729)
        probe_count = max(topk, int(round(candidate_nodes.numel() * args.query_budget_fraction)))
        probe_order = torch.randperm(int(candidate_nodes.numel()), generator=generator, device='cpu').to(device)
        candidate_nodes = candidate_nodes[probe_order[:probe_count]]

    if args.experiment_variant == 'random_anchors' or args.anchor_metric == 'random':
        generator = torch.Generator(device='cpu')
        generator.manual_seed(args.seed)
        order = torch.randperm(int(candidate_nodes.numel()), generator=generator, device='cpu').to(device)
        anchors = candidate_nodes[order[:topk]]
        vul_scores = {'scores': None, 'top_nodes': anchors}
    elif args.anchor_score_method == "topology_sensitivity" and (
        args.anchor_metric != 'confidence_drop' or args.anchor_probe_cache
    ):
        sensitivity_scorer = TopologySensitiveAnchorLocalizer(surrogate=surrogate, device=device, alpha=0.6,
                               drop_top=args.anchor_drop_top,
                               drop_strategy=args.anchor_drop_strategy, seed=args.seed)
        vul_scores = sensitivity_scorer.score_probe_metric(
            G, feats, candidate_nodes, metric=args.anchor_metric,
            topk=topk, cache_path=args.anchor_probe_cache,
        )
    elif args.anchor_score_method == "topology_sensitivity":
        sensitivity_scorer = TopologySensitiveAnchorLocalizer(surrogate=surrogate, device=device, alpha=0.6,
                               drop_top=args.anchor_drop_top,
                               drop_strategy=args.anchor_drop_strategy, seed=args.seed)
        vul_scores = sensitivity_scorer.score_topology_sensitivity(
            G,
            feats,
            pseudo_labels,
            candidate_nodes,
            topk=topk,
        )
    elif args.anchor_score_method == "ppr":
        vul_scores = score_vulnerability_ppr(
            G=G,
            candidate_nodes=candidate_nodes,
            topk=topk,
            alpha=args.ppr_alpha,
            max_iter=args.ppr_max_iter,
            tol=args.ppr_tol,
        )
    elif args.anchor_score_method == "bridge_community":
        vul_scores = score_vulnerability_bridge_community(
            G=G,
            candidate_nodes=candidate_nodes,
            topk=topk,
            community_method=args.community_method,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown anchor_score_method: {args.anchor_score_method}")
    anchors = vul_scores['top_nodes']  # 高易损节点做 anchor
    if args.anchor_selection != 'top_m' and vul_scores.get('scores') is not None:
        anchors = diversify_anchor_selection(
            G, vul_scores['scores'], candidate_nodes, topk,
            args.anchor_selection, args.anchor_diversity_lambda, args.seed,
        )
        vul_scores['top_nodes'] = anchors
    if args.anchor_output_path:
        anchor_path = os.path.abspath(args.anchor_output_path)
        os.makedirs(os.path.dirname(anchor_path), exist_ok=True)
        torch.save({
            'dataset': args.dataset, 'seed': int(args.seed), 'model_seed': model_seed,
            'selection': args.anchor_selection,
            'anchor_ids': anchors.detach().cpu(),
            'scores': None if vul_scores.get('scores') is None else vul_scores['scores'].detach().cpu(),
        }, anchor_path)
    if args.anchor_only:
        anchor_metrics = {
            'status': 'OK',
            'method': 'TS-SNIA',
            'dataset': args.dataset,
            'victim': args.victim,
            'seed': int(args.seed),
            'model_seed': model_seed,
            'use_lcc': bool(args.use_lcc),
            'anchor_metric': args.anchor_metric,
            'anchor_drop_top': int(args.anchor_drop_top),
            'anchor_drop_strategy': args.anchor_drop_strategy,
            'anchor_count': int(anchors.numel()),
            'configured_epochs': int(args.epochs),
            'epochs_trained': int(clean.get('epochs_trained', args.epochs)),
            'fixed_surrogate_epochs': bool(args.fixed_surrogate_epochs),
            'anchor_ids': [int(node) for node in anchors.detach().cpu().tolist()],
            'clean_accuracy': float(clean['test_acc']),
            'anchor_probe_model_calls': int(surrogate.query_counts.get('anchor_probe', {}).get('model_calls', 0)),
            'anchor_probe_node_predictions': int(surrogate.query_counts.get('anchor_probe', {}).get('node_predictions', 0)),
            'anchor_output_path': os.path.abspath(args.anchor_output_path) if args.anchor_output_path else '',
            'elapsed_seconds': float(time.perf_counter() - experiment_started),
        }
        if args.result_json:
            result_path = os.path.abspath(args.result_json)
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            with open(result_path, 'w', encoding='utf-8') as stream:
                json.dump(anchor_metrics, stream, ensure_ascii=False, indent=2, sort_keys=True)
        return
    surrogate.query_stage = None
    print(f"Picked {len(anchors)} anchors as vulnerable nodes.")
    base_nodes = G.num_nodes()
    base_edges = G.num_edges()

    G_clean_snapshot = G.clone()
    X_clean_snapshot = feats.clone()

    # 3) 注入规模/度预算
    N0 = G.num_nodes()
    # inj_nodes = max(1, int(round(N0 * args.inj_ratio)))
    inj_nodes = len(anchors)  # ← 每个子图 1 个注入节点（原来是按 inj_ratio）
    avg_deg = int(round(float(G.num_edges())/N0))*2
    # args.nes_edge_budget = avg_deg/2
    print("平均度为:{}".format(avg_deg))
    print("边预算为{}".format(args.sparse_action_budget))
    deg_per_node = avg_deg if args.deg_per_node=='auto' else int(args.deg_per_node)
    total_new_edges = inj_nodes * deg_per_node
    print(f"[Budget] inject nodes = ({args.injection_ratio*100:.1f}%), "
          f"deg per injected = {deg_per_node}, total new edges ≈ {total_new_edges}")
    # 如果命令行没给预算，就用一个小而稳的默认
    if args.sparse_action_budget is None:
        args.sparse_action_budget = max(2, int(0.3 * deg_per_node))  # 例如 0.3*deg_per_node
    print(f"  [Original] graph size N={base_nodes}, E={base_edges} (no structural change)")

    print("\n[Stage 2] Building k-hop candidate pools…")
    base_pool = get_khop_candidates(
        G=G,
        anchors=anchors,  # Stage 1 得到的脆弱锚点（tensor）
        k_hop_edge=args.candidate_hops,  # 例如 1 或 2；0 表示用全图的 1/2-hop 近邻集
        deg_per_node=int(round(args.external_budget_ratio * deg_per_node)),
        fallback_random=True
    )
    if args.experiment_variant == 'no_anchor_guidance':
        # Preserve each candidate-pool size but remove all anchor/locality
        # guidance.  Sampling is deterministic under the experiment seed.
        random_pool = {}
        for anchor, values in base_pool.items():
            pool_size = max(int(round(args.external_budget_ratio * deg_per_node)), len(values))
            allowed = torch.arange(N0, device=device)
            allowed = allowed[allowed != int(anchor)]
            random_pool[int(anchor)] = allowed[torch.randperm(allowed.numel(), device=device)[:pool_size]].tolist()
        base_pool = random_pool
    print("  Candidate pool ready (k-hop).")
    print_khop_subgraph_stats(G, anchors, k_hop_edge=args.candidate_hops)
   
    base_nodes = G.num_nodes()
    base_edges = G.num_edges()
    print(f"  [Stage-1 done] graph size N={base_nodes}, E={base_edges} (no structural change)")

    # if args.use_pseudo:
    #     with torch.no_grad():
    #         pseudo_logits = victim.predict(G, feats)
    #     labels_for_sampler = pseudo_logits  # 关键：取 argmax 得到 1D 标签
    # else:
    #     labels_for_sampler = labels
    # weights = dict(homo=1.0, degree=0.3, cos=0.5)  # 起步比较稳的权重
    # weights = dict(homo=1.6, degree=0.1, cos=0.8)
    weights = dict(homo=args.cue_homo_weight, degree=args.cue_degree_weight, cos=args.cue_cos_weight)
    candidate = sample_anchor_neighbors(
        G=G,
        anchors=anchors,
        base_pool=base_pool,
        # K=int(round(0.7*deg_per_node)),
        K=deg_per_node,
        labels=pseudo_labels,
        feats=feats,
        weights=weights,
        # temp=1.0,
        temp=args.sampling_temperature,
        exact_k=True
    )
    G, inj_ids, anchor_to_injid = construct_cross_block_injection(
        G=G,
        anchors=anchors,
        candidate=candidate,
        deg_per_node=int(round(args.external_budget_ratio*deg_per_node)),
        feats=feats,  # 若 G.ndata['feat'] 已有也可不传
        undirected=True,
        verbose=True
    )
    # 同步外部 feats 引用（后续统一用 G.ndata['feat'] 更稳）
    feats = G.ndata['feat']
    base_nodes = G.num_nodes()
    base_edges = G.num_edges()
    print(f"[After Stage 2b] N={G.num_nodes()}, E={G.num_edges()}")
    print("  Candidate pool refined by Enhanced-Bernoulli (homo/degree/cos mixed).")

    per_node_inner_used = None
    # print(inj_ids,candidate)
    labels_full = build_labels_with_injected(G, labels, anchor_to_injid, device)
    G_before_E = base_edges
    # 如果后面需要“误导视图”的标签（labels_for_sampler）以此为起点：
    labels_for_True = labels_full.clone()
    if args.experiment_variant == 'no_internal_edges':
        inj_pairs = []
        print('[Stage 3] Ablation: internal injected-to-injected edges disabled.')
    else:
        G, inj_pairs = construct_internal_injected_edges(
            G=G,
            inj_ids=inj_ids,
            deg_per_node=deg_per_node,
            intra_ratio=1.0-args.external_budget_ratio,
            per_node_inner_used=per_node_inner_used,
            undirected=False,
            seed=args.seed,
            verbose=True,
            alpha_intra=args.alpha_intra if args.alpha_intra > 0 else 2.0,
            tgt_pref="indegree",
            tgt_gamma=1.5
        )

    print(f"[After Stage 3] N={G.num_nodes()}, E={G.num_edges()} (+{G.num_edges() - G_before_E} edges)")
    use_pseudo_after3 = bool(args.use_pseudo)  # 布尔就用布尔，不和字符串比较
    device = feats.device
   
    pseudo_labels = pseudo_labels.to(device=feats.device, dtype=torch.long)


    # 类原型（特征对齐用）
    protos = []
    for c in range(C):
        idx = (pseudo_labels == c).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            protos.append(torch.zeros(feats.size(1), device=device))
        else:
            protos.append(feats[idx].mean(dim=0))
    class_protos = torch.stack(protos, dim=0).to(device)

    print(f"[After Stage 3] pseudo_labels ready (mode={use_pseudo_after3}).")

    # ===== 2) 锚点-注入映射合并 & 子图抽取（保留你原来的打印，做健壮处理）=====
    merged_by_anchor, injid_to_cands, edge_pairs = merge_anchor_maps(anchor_to_injid, base_pool)

    print(f"共提取子图数量: {len(merged_by_anchor)}")

    G_before_Edge = G.num_edges()

    with torch.no_grad():
        # logits_final = victim.predict(G, feats)
        y_hat = predict_labels_1d(surrogate, G, feats)
    # y_hat = logits_final.argmax(dim=1)
    gt = labels_for_True.to(y_hat.device)

    # （1）测试集准确率（节点分类）
    if 'test_idx' in locals() and test_idx is not None:
        surrogate_test_accuracy_after_attack = (y_hat[test_idx] == gt[test_idx]).float().mean().item()
        print(f"[Analysis] Fixed-surrogate test accuracy after attack: {surrogate_test_accuracy_after_attack * 100:.2f}%")
    
    if args.attack_memory_only:
        # This diagnostic retraining is not part of poisoned-graph generation.
        pre_nes_poison_result = {"test_acc": float("nan")}
    else:
        labels_poison_pre = labels_for_True.to(device)
        train_idx_poison_pre = train_idx.to(device)
        val_idx_poison_pre = val_idx.to(device)
        test_idx_poison_pre = test_idx.to(device)
        pre_nes_poison_victim = GNNModelWrapper(
            model_name=args.victim,
            in_dim=feats.shape[1],
            n_classes=int(labels.max().item() + 1),
            lr=args.lr,
            wd=args.wd,
            args=args,
            device=device,
        )
        pre_nes_poison_result = pre_nes_poison_victim.fit_eval(
            G, feats, labels_poison_pre,
            train_idx_poison_pre, val_idx_poison_pre, test_idx_poison_pre,
            epochs=args.epochs, patience=args.patience, verbose=True,
        )
        print("=" * 80)
        print("[Pre-NES Poison Eval Result]")
        print(f"Clean test acc:        {clean['test_acc'] * 100:.2f}%")
        print(f"Pre-NES poison acc:    {pre_nes_poison_result['test_acc'] * 100:.2f}%")
        print(f"Pre-NES poison drop:   {(clean['test_acc'] - pre_nes_poison_result['test_acc']) * 100:.2f}%")
        print("=" * 80)
        
        
    from sparse_refinement import run_sparse_refinement

    # The NES action budget controls how many edits may be considered, while
    # this value enforces the paper's final per-injected-node degree budget.
    args.deg_per_node_landing = int(deg_per_node)
    args.optimize_nes_features = args.experiment_variant != 'no_nes_features'
    surrogate.query_stage = 'nes_feature'
    G_adv, feats_adv = run_sparse_refinement(
        G=G,
        feats=feats,
        surrogate=surrogate,
        merged_by_anchor=merged_by_anchor,
        pseudo_labels=pseudo_labels,
        class_protos=class_protos,
        args=args,
        T_homo=0.7,
        log_every=args.log_every
    )
    surrogate.query_stage = None

    print(f"[After SubgraphSparseAttack] N={G_adv.num_nodes()}, E={G_adv.num_edges()}")
    if args.attack_memory_only:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            attack_peak_allocated = int(torch.cuda.max_memory_allocated(device))
        else:
            attack_peak_allocated = 0
        actual_external_edges, actual_internal_edges = count_injected_edge_types(G_adv, N0)
        resource_metrics = {
            "status": "OK",
            "method": "TS-SNIA",
            "dataset": args.dataset,
            "victim": args.victim,
            "seed": int(args.seed),
            "injection_ratio": float(args.injection_ratio),
            "injected_nodes": int(len(inj_ids)),
            "degree_per_injected_node": int(deg_per_node),
            "actual_external_edges": int(actual_external_edges),
            "actual_internal_edges": int(actual_internal_edges),
            "attack_seconds": float(time.perf_counter() - attack_resource_started),
            "memory_metric": "attack-stage additional peak torch CUDA memory_allocated",
            "attack_memory_baseline_mb": attack_memory_baseline / (1024 ** 2),
            "attack_peak_allocated_mb": attack_peak_allocated / (1024 ** 2),
            "attack_additional_peak_gpu_mb": max(
                0.0, (attack_peak_allocated - attack_memory_baseline) / (1024 ** 2)
            ),
        }
        if not args.result_json:
            raise ValueError('--attack_memory_only requires --result_json')
        result_path = os.path.abspath(args.result_json)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, 'w', encoding='utf-8') as stream:
            json.dump(resource_metrics, stream, ensure_ascii=False, indent=2, sort_keys=True)
        return
    
    spnia_stealth = spnia_unnoticeability_eval(
    G_clean=G_clean_snapshot,
    G_adv=G_adv,
    feats_clean=X_clean_snapshot,
    feats_adv=feats_adv,
    N0=N0,
    inj_ids=inj_ids,
    epsilon=args.injection_ratio,
)
    
    print("\n[Poison Eval] Retrain victim on poisoned graph...")

# 1) 构造 poisoned labels，长度必须和 G_adv.num_nodes() 一致
# 原始节点使用真实标签；注入节点不参与监督训练，标签填 -1 即可
    N_orig = int(labels.size(0))
    N_poison = int(G_adv.num_nodes())

    # labels_poison = torch.full(
    #     (N_poison,),
    #     2,
    #     dtype=labels.dtype,
    #     device=device,
    # )
    # labels_poison[:N_orig] = labels.to(device)
    labels_poison = labels_for_True.to(device)
    # 2) 训练/验证/测试索引仍然使用原始节点索引
    # 这样 poisoned graph 只改变消息传播结构，不把注入节点加入训练监督
    train_idx_poison = train_idx.to(device)
    val_idx_poison = val_idx.to(device)
    test_idx_poison = test_idx.to(device)

    # 3) 重新初始化一个 victim；不要复用已经 clean-trained 的 victim 参数
    poison_victim = GNNModelWrapper(
        model_name=args.victim,
        in_dim=feats_adv.shape[1],
        n_classes=int(labels.max().item() + 1),
        lr=args.lr,
        wd=args.wd,
        args=args,
        device=device,
    )

    # 4) 在 poisoned graph 上重新训练，并在原始 test_idx 上测试
    poison_result = poison_victim.fit_eval(
        G_adv,
        feats_adv,
        labels_poison,
        train_idx_poison,
        val_idx_poison,
        test_idx_poison,
        epochs=args.epochs,
        patience=args.patience,
        verbose=True,
    )

    print("=" * 80)
    print("[Poison Eval Result]")
    print(f"Clean test acc:    {clean['test_acc'] * 100:.2f}%")
    print(f"Poison test acc:   {poison_result['test_acc'] * 100:.2f}%")
    print(f"Poison acc drop:   {(clean['test_acc'] - poison_result['test_acc']) * 100:.2f}%")
    print("=" * 80)
    
    G_adv00, X_adv00 = extract_clean_part_from_adv(G_adv, feats_adv, N0)
    _ = compare_graphs_ref_vs_advOO(
        G_ref=G_clean_snapshot, X_ref=X_clean_snapshot,
        G_advOO=G_adv00, X_advOO=X_adv00,
        undirected=False  # 如果你的图逻辑上当作无向，这里设 True
    )
    # 使用对抗图/特征
    G, feats = G_adv, feats_adv

    # ===== 4) 最终评测（务必用真实标签 labels）=====
    with torch.no_grad():
        # logits_final = victim.predict(G, feats)
        y_hat = predict_labels_1d(surrogate, G, feats)
    # y_hat = logits_final.argmax(dim=1)
    gt = labels_for_True.to(y_hat.device)

    # （1）测试集准确率（节点分类）
    if 'test_idx' in locals() and test_idx is not None:
        surrogate_test_accuracy_after_attack = (y_hat[test_idx] == gt[test_idx]).float().mean().item()
        print(f"[Analysis] Fixed-surrogate test accuracy after attack: {surrogate_test_accuracy_after_attack * 100:.2f}%")

    # （2）全图准确率（可选）
    # overall_acc_final = (y_hat == gt).float().mean().item()
    # print(f"[Final] Overall accuracy (GT labels): {overall_acc_final * 100:.2f}%")

    # （3）固定代理模型在锚点上的误分类比例（非目标攻击成功率）
    asr = (y_hat[anchors] != labels_for_True[anchors].to(y_hat.device)).float().mean().item()
    print(f"[Analysis] Fixed-surrogate anchor misclassification rate: {asr * 100:.2f}%")

    # （4）结构改动量（相对 Stage-3 前）
    print(f"[Delta] Added edges after sparse attack (approx): {G.num_edges() - G_before_Edge}")
    print(G.num_edges())
    print(G.num_nodes())

    actual_external_edges, actual_internal_edges = count_injected_edge_types(G_adv, N0)
    metrics = {
        "status": "OK",
        "method": "TS-SNIA",
        "experiment_variant": args.experiment_variant,
        "structure_mode": args.structure_mode,
        "query_budget_fraction": float(args.query_budget_fraction),
        "dataset": args.dataset,
        "victim": args.victim,
        "seed": int(args.seed),
        "model_seed": model_seed,
        "use_lcc": bool(args.use_lcc),
        "anchor_score_method": args.anchor_score_method,
        "anchor_metric": args.anchor_metric,
        "anchor_drop_top": int(args.anchor_drop_top),
        "anchor_drop_strategy": args.anchor_drop_strategy,
        "anchor_selection": args.anchor_selection,
        "anchor_diversity_lambda": float(args.anchor_diversity_lambda),
        "attack_output_mode": args.attack_output_mode,
        "attack_output_bits": int(args.attack_output_bits),
        "attack_output_topk": int(args.attack_output_topk),
        "cue_homo_weight": float(args.cue_homo_weight),
        "cue_degree_weight": float(args.cue_degree_weight),
        "cue_cos_weight": float(args.cue_cos_weight),
        "sampling_temperature": float(args.sampling_temperature),
        "lambda_l1": float(args.feature_l1_weight),
        "lambda_class": float(args.prototype_alignment_weight),
        "anchor_probe_cache_hit": bool(vul_scores.get('probe_cache_hit', False)),
        "anchor_ids": [int(node) for node in anchors.detach().cpu().tolist()],
        "anchor_average_degree": float(G_clean_snapshot.out_degrees(anchors).float().mean().item()),
        "injection_ratio": float(args.injection_ratio),
        "degree_per_injected_node": int(deg_per_node),
        "external_budget_ratio": float(args.external_budget_ratio),
        "internal_budget_ratio": float(1.0 - args.external_budget_ratio),
        "external_degree_budget_per_node": int(round(args.external_budget_ratio * deg_per_node)),
        "internal_degree_budget_per_node": int(round((1.0 - args.external_budget_ratio) * deg_per_node)),
        "actual_external_edges": int(actual_external_edges),
        "actual_internal_edges": int(actual_internal_edges),
        "sparse_action_budget": int(args.sparse_action_budget),
        "clean_accuracy": float(clean["test_acc"]),
        "clean_guidance_initialization_model_calls": int(clean_initialization_counts["model_calls"]),
        "clean_guidance_initialization_node_predictions": int(clean_initialization_counts["node_predictions"]),
        "pre_nes_poison_accuracy": float(pre_nes_poison_result["test_acc"]),
        "poison_accuracy": float(poison_result["test_acc"]),
        "accuracy_drop": float(clean["test_acc"] - poison_result["test_acc"]),
        "surrogate_anchor_misclassification_rate": float(asr),
        "anchor_asr": float(asr),  # Historical result-key alias.,
        "original_nodes": int(N0),
        "injected_nodes": int(len(inj_ids)),
        "original_edges": int(G_clean_snapshot.num_edges()),
        "poisoned_edges": int(G_adv.num_edges()),
        "injected_avg_degree_actual": float(spnia_stealth["injected_avg_degree"]),
        "injected_max_degree_actual": int(spnia_stealth["injected_max_degree"]),
        "configured_degree_budget_ok": bool(spnia_stealth["injected_max_degree"] <= deg_per_node),
        "budget_ok": bool(spnia_stealth["injected_max_degree"] <= deg_per_node),
        "spnia_average_degree_check_ok": bool(spnia_stealth["degree_budget_ok"]),
        "node_budget_ok": bool(spnia_stealth["node_budget_ok"]),
        "degree_similarity_percent": float(spnia_stealth["degree_similarity_percent"]),
        "elapsed_seconds": float(time.perf_counter() - experiment_started),
        "query_counts": surrogate.query_counts,
        "anchor_probe_model_calls": int(surrogate.query_counts.get('anchor_probe', {}).get('model_calls', 0)),
        "anchor_probe_node_predictions": int(surrogate.query_counts.get('anchor_probe', {}).get('node_predictions', 0)),
        "nes_feature_model_calls": int(surrogate.query_counts.get('nes_feature', {}).get('model_calls', 0)),
        "nes_feature_node_predictions": int(surrogate.query_counts.get('nes_feature', {}).get('node_predictions', 0)),
        "structure_model_calls": int(surrogate.query_counts.get('structure', {}).get('model_calls', 0)),
        "structure_node_predictions": int(surrogate.query_counts.get('structure', {}).get('node_predictions', 0)),
    }
    metrics["total_query_model_calls"] = sum(int(v.get("model_calls", 0)) for v in surrogate.query_counts.values())
    metrics["total_query_node_predictions"] = sum(int(v.get("node_predictions", 0)) for v in surrogate.query_counts.values())
    metrics.update(getattr(args, 'attack_stats', {}))
    if args.mode == 'defense' or args.defense or args.defenses:
        from defense import evaluate_defense_pairs
        selected_defenses = args.defenses or ([args.defense] if args.defense else None)
        metrics['defense_results'] = evaluate_defense_pairs(
            G_clean_snapshot, X_clean_snapshot, labels,
            G_adv, feats_adv, labels_poison,
            train_idx, val_idx, test_idx, seed=args.seed, names=selected_defenses,
            epochs=args.defense_epochs, patience=args.defense_patience,
            hidden=args.defense_hidden, jaccard_args=args,
        )
    if args.result_json:
        result_path = os.path.abspath(args.result_json)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, 'w', encoding='utf-8') as stream:
            json.dump(metrics, stream, ensure_ascii=False, indent=2, sort_keys=True)
    if args.artifact_path:
        artifact_path = os.path.abspath(args.artifact_path)
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        graph_cpu = G_adv.to('cpu')
        graph_cpu.ndata['feat'] = feats_adv.detach().cpu()
        graph_cpu.ndata['label'] = labels_for_True.detach().cpu()
        dgl.save_graphs(
            artifact_path,
            [graph_cpu],
            {
                'train_idx': train_idx.detach().cpu(),
                'val_idx': val_idx.detach().cpu(),
                'test_idx': test_idx.detach().cpu(),
                'original_nodes': torch.tensor([N0], dtype=torch.long),
            },
        )
    return metrics


def _write_run_tables(results_dir, mode):
    import csv
    from collections import defaultdict
    from statistics import mean, stdev
    rows = []
    for path in sorted((results_dir / mode).glob('*.json'), key=lambda item: item.stat().st_mtime_ns):
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('status') != 'OK':
            continue
        if mode == 'train' and not all(key in record for key in
                                      ['dataset', 'victim', 'seed', 'clean_accuracy', 'poison_accuracy']):
            # Anchor-only and memory-only diagnostics are not training cases.
            continue
        if mode == 'defense':
            for item in record.get('defense_results', []):
                rows.append({'dataset': record['dataset'], 'victim': record['victim'],
                             'seed': record['seed'], 'defense': item['defense'],
                             'clean_accuracy': item['clean_accuracy'],
                             'poison_accuracy': item['poison_accuracy']})
        else:
            rows.append({key: record[key] for key in
                         ['dataset', 'victim', 'seed', 'clean_accuracy', 'poison_accuracy']})
    # A case may occur in a single-defense run and a later multi-defense run.
    # Keep the latest observation of each dataset/model/defense/seed.
    unique = {(row['dataset'], row['victim'], row.get('defense', ''), row['seed']): row for row in rows}
    rows = list(unique.values())
    if not rows:
        return
    with (results_dir / f'{mode}_raw.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['victim'], row.get('defense', ''))].append(row)
    summaries = []
    for (dataset, victim, defense), group in groups.items():
        clean_values = [row['clean_accuracy'] for row in group]
        values = [row['poison_accuracy'] for row in group]
        summaries.append({'dataset':dataset,'victim':victim,'defense':defense,'runs':len(values),
                          'clean_mean':mean(clean_values),
                          'clean_sample_std':stdev(clean_values) if len(values)>1 else '',
                          'poison_mean':mean(values),
                          'poison_sample_std':stdev(values) if len(values)>1 else ''})
    with (results_dir / f'{mode}_summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def _evaluate_saved_artifact(args):
    from defense import evaluate_saved_artifact
    return evaluate_saved_artifact(args)


def _run_configuration(args):
    """Identify compatible completed runs without changing attack randomness."""
    import hashlib
    excluded = {'overwrite', 'result_json', 'artifact_path', 'results_dir',
                'datasets', 'models', 'seeds', 'defense', 'defenses'}
    settings = {name: str(value) if isinstance(value, Path) else value
                for name, value in vars(args).items() if name not in excluded}
    if args.mode == 'defense':
        from config import DEFENSES
        settings['selected_defenses'] = sorted(args.defenses or ([args.defense] if args.defense else DEFENSES))
    if args.artifact_input:
        settings['artifact_sha256'] = hashlib.sha256(args.artifact_input.read_bytes()).hexdigest()
    return settings


def main(argv=None, *, legacy_single=False):
    import copy
    import sys
    from config import DATASETS, VICTIMS, SEEDS, LCC_DATASETS
    cli = list(sys.argv[1:] if argv is None else argv)
    args = build_args(cli)
    provided = {word.split('=')[0] for word in cli if word.startswith('--')}
    datasets = args.datasets or ([args.dataset] if legacy_single or '--dataset' in provided else DATASETS)
    victims = args.models or ([args.victim] if legacy_single or '--victim' in provided else VICTIMS)
    seeds = args.seeds or ([args.seed] if legacy_single or '--seed' in provided else SEEDS)
    if args.defense or args.defenses or args.artifact_input:
        args.mode = 'defense'
    cases = [(dataset, victim, seed) for dataset in datasets for victim in victims for seed in seeds]
    if len(cases) > 1 and (args.result_json or args.artifact_path or args.artifact_input):
        raise ValueError('Use one dataset/model/seed with an explicit result or artifact path.')
    args.results_dir = args.results_dir.resolve()
    output_dir = args.results_dir / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    for dataset, victim, seed in cases:
        case = copy.copy(args)
        case.dataset, case.victim, case.seed = dataset, victim, seed
        if not legacy_single and not ({'--use_lcc','--no-use_lcc'} & provided):
            case.use_lcc = dataset in LCC_DATASETS
        suffix = ('_' + '-'.join(case.defenses or ([case.defense] if case.defense else []))) if case.mode == 'defense' else ''
        run_id = f'{dataset}_{victim}_seed{seed}{suffix}'
        target = Path(case.result_json) if case.result_json else output_dir / f'{run_id}.json'
        case.result_json = str(target)
        artifact = args.results_dir / 'artifacts' / f'{dataset}_{victim}_seed{seed}.bin'
        case.artifact_path = case.artifact_path or str(artifact)
        configuration = _run_configuration(case)
        if target.exists() and not case.overwrite:
            try:
                record = json.loads(target.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                record = {}
            artifact_available = case.artifact_input or Path(case.artifact_path).is_file()
            if record.get('status') == 'OK' and record.get('run_configuration') == configuration and artifact_available:
                print(f'[Skip] completed {run_id}: {target}')
                completed.append(record)
                continue
        print(f'[Run] {case.mode}: {run_id}', flush=True)
        if case.artifact_input:
            record = _evaluate_saved_artifact(case)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(record, indent=2, sort_keys=True), encoding='utf-8')
        else:
            record = run_single(case)
        if record is not None:
            record['run_configuration'] = configuration
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(record, indent=2, sort_keys=True), encoding='utf-8')
        completed.append(record)
    _write_run_tables(args.results_dir, args.mode)
    return completed


if __name__ == '__main__':
    from pathlib import Path
    main()
