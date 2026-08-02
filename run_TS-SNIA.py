# -*- coding: utf-8 -*-
"""
C2SNI 主入口脚本
论文依据：
- 黑盒 NES: Ilyas et al., ICML'18
- 因果: Pearl '09
"""

# import sys, io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import argparse, os, math, random
from ppr_anchor_score import score_vulnerability_ppr
from community_anchor_score import score_vulnerability_bridge_community
import numpy as np
import torch
from data_utils import load_dgl_dataset, clone_graph_with_data, set_seed
from victim_models import Victim
from causal_score import CausalScorer
from pruning import prune_edges_by_dissim
from utils import device_of, to_cpu
import dgl
# from attacker_sparse import build_surrogate_from_victim, stage4_sparse_optimize_per_subgraph
# ---------- Robust neighbor choosing helpers ----------
from EdgeSampler import augmstru_enhanced_bernoulli, stage2b_inject_per_subgraph,stage3_connect_between_injected   # 通用增强型 Bernoulli（你已放在 EdgeSampler.py）
# from attacker_sparse import SparseInjectionAttack     # 稀疏矩阵可微优化攻击器（上一条消息给你的实现）
from subgraph_khop import print_khop_subgraph_stats, get_khop_candidates
from utils import extract_anchor_subgraphs_from_G, load_original_graph, merge_anchor_maps, batch_subgraphs_from_injid_to_cands
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

    result = {
        "injection_ratio": inj_ratio,
        "injection_ratio_percent": inj_ratio * 100.0,
        "node_budget_ok": inj_ratio <= float(epsilon) + 1e-12,

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

def predict_labels_1d(victim, G, feats) -> torch.Tensor:
    with torch.no_grad():
        out = victim.predict(G, feats)
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


def largest_connected_component_nodes_dgl(G: dgl.DGLGraph):
    """
    Return old node ids in the largest connected component of a DGL graph.
    The graph is treated as undirected, which is consistent with citation-network preprocessing.
    """
    import scipy.sparse as sp

    G_cpu = G.to("cpu")
    src, dst = G_cpu.edges()
    src = src.numpy()
    dst = dst.numpy()

    n = int(G_cpu.num_nodes())
    data = np.ones(len(src), dtype=np.float32)
    adj = sp.coo_matrix((data, (src, dst)), shape=(n, n), dtype=np.float32).tocsr()

    # Treat the graph as undirected and unweighted.
    adj = adj + adj.T
    adj.data[:] = 1.0
    adj.setdiag(0)
    adj.eliminate_zeros()

    _, component_indices = sp.csgraph.connected_components(adj, directed=False)
    component_sizes = np.bincount(component_indices)
    largest_component = int(np.argmax(component_sizes))

    keep_nodes = np.where(component_indices == largest_component)[0]
    print(
        f"[LCC] Selecting largest connected component: "
        f"{len(keep_nodes)} / {n} nodes kept."
    )

    return torch.from_numpy(keep_nodes).long()

def apply_lcc_to_loaded_dgl_data(
    G,
    feats,
    labels,
    train_idx,
    val_idx,
    test_idx,
    device,
):
    """
    Apply largest connected component filtering after loading a DGL dataset.
    This function also remaps train/val/test indices from old node ids to new compact ids.
    """
    keep_cpu = largest_connected_component_nodes_dgl(G)

    old_num_nodes = int(G.num_nodes())
    old2new = torch.full((old_num_nodes,), -1, dtype=torch.long)
    old2new[keep_cpu] = torch.arange(keep_cpu.numel(), dtype=torch.long)

    def remap_idx(idx):
        idx_cpu = idx.detach().cpu().long()
        valid_mask = old2new[idx_cpu] >= 0
        new_idx = old2new[idx_cpu[valid_mask]]
        return new_idx.to(device)

    keep = keep_cpu.to(device)

    # DGL will relabel kept nodes into compact ids [0, ..., N_lcc-1].
    G_lcc = dgl.node_subgraph(G, keep)
    G_lcc = G_lcc.to(device)

    feats_lcc = feats[keep].clone().to(device)
    labels_lcc = labels[keep].clone().to(device)

    train_idx_lcc = remap_idx(train_idx)
    val_idx_lcc = remap_idx(val_idx)
    test_idx_lcc = remap_idx(test_idx)

    G_lcc.ndata["feat"] = feats_lcc

    print(
        f"[LCC] Split sizes after remapping: "
        f"train={len(train_idx_lcc)}, val={len(val_idx_lcc)}, test={len(test_idx_lcc)}"
    )

    return G_lcc, feats_lcc, labels_lcc, train_idx_lcc, val_idx_lcc, test_idx_lcc

def normalize_dataset_name(value: str) -> str:
    """Normalize the only supported dataset alias: cora-ml -> cora_ml."""
    return value.strip().lower().replace('-', '_')


def normalize_victim_name(value: str) -> str:
    """Normalize user-facing victim names to their internal implementations."""
    name = value.strip().lower().replace('-', '_')
    aliases = {
        'sac': 'sgc',
        'jaccardgcn': 'gcn_jaccard',
        'jaccard_gcn': 'gcn_jaccard',
        'graphsage': 'sage',
        'graph_sage': 'sage',
    }
    return aliases.get(name, name)


def build_args():
    p = argparse.ArgumentParser()
    # 数据/受害模型
    p.add_argument(
        '--dataset',
        type=normalize_dataset_name,
        default='citeseer',
        choices=['cora', 'citeseer', 'cora_ml', 'pubmed'],
    )
    p.add_argument(
        '--victim',
        type=normalize_victim_name,
        default='gcn',
        choices=['gcn', 'sgc', 'gcn_jaccard', 'gin', 'sage'],
        help='Victim model. Aliases: SAC=SGC, jaccardgcn=GCN-Jaccard, GraphSAGE=sage.',
    )
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--wd', type=float, default=5e-4)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--feat_dropout', type=float, default=0.6)
    p.add_argument('--attn_dropout', type=float, default=0.6)
    p.add_argument('--num_heads1', type=int, default=8)
    p.add_argument('--num_heads2', type=int, default=1)
    p.add_argument('--sage_agg_type', type=str, default='mean')
    p.add_argument('--appnp_hidden', type=int, default=64)
    p.add_argument('--K', type=int, default=10)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--gin_hidden', type=int, default=64)
    p.add_argument('--gin_agg_type', type=str, default='sum')
    # 攻击目标
    p.add_argument('--attack_objective', type=str, default='untargeted',
                   choices=['untargeted','targeted'])
    p.add_argument('--target_ids', type=str, default='')  # 逗号分隔

    # 预算与稀疏
    p.add_argument('--deg_per_node', type=str, default='auto')  # 'auto' 或者整数
    p.add_argument('--max_rounds', type=int, default=1)
    # 超参可调
    p.add_argument('--opt_steps', type=int, default=200)
    p.add_argument('--opt_lr', type=float, default=0.1)
    p.add_argument('--hid_sur', type=int, default=64)

    p.add_argument('--lam_l1', type=float, default=1e-3)
  
    p.add_argument('--nes_edge_budget', type=int, default=2,
                   help='每个锚点最多新增的外边数；None 表示不做硬预算')
    p.add_argument('--nes_hard_topk', action='store_true',default=True,
                   help='在采样和落地时用 top-k 严格卡住边数（强烈建议打开）')
    p.add_argument('--nes_edge_tau', type=float, default=1.0,
                   help='边 logits 的温度（>1 更平缓，<1 更尖锐）')
    p.add_argument('--nes_edge_thr', type=float, default=0.5,
                   help='阈值二值化时的阈值（配合非 hard_topk 时用）')
    p.add_argument('--bb_queries', type=int, default=2)
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--sigma', type=float, default=0.25)
    p.add_argument('--adam_lr', type=float, default=0.05)
    p.add_argument('--tau', type=float, default=0.5)     # 对比温度
    p.add_argument('--mmd_w', type=float, default=1.0)
    p.add_argument('--degkl_w', type=float, default=0.0)
    p.add_argument('--causal_w', type=float, default=1.0)
    p.add_argument('--l0_w', type=float, default=1e-3)
    p.add_argument('--khop_edge', type=int, default=1,
                        help='围绕锚点进行子图采样的 hop 数；0 表示不用采样，靠 1/2-hop 近邻集')
    p.add_argument('--p_stru', type=float, default=0.8,
                   help='AugmStru 的 Bernoulli 边保留概率（0~1，越大越贴近原图）')
    p.add_argument('--use_pseudo', action='store_true',default=True,
                   help='用于 AugmStru 的同类判定是否使用受害模型伪标签补全')
    # 剪边（隐蔽性）
    p.add_argument('--prune_ratio', type=float, default=0.3)

    # 训练/日志
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)

    p.add_argument("--lam_in", default=0.08, type=float)
    p.add_argument("--lam_out", default=0.1, type=float)
    p.add_argument("--lam_global", default=0.1, type=float)
    p.add_argument("--alpha_intra", default=0.0, type=float, help="0 表示自动用 CCDF 拟合；>0 则固定幂律指数")
    p.add_argument('--lam_edgecls', type=float, default=0.8,
                   help='子图稀疏优化中：优先连向目标类的边约束权重')

    p.add_argument('--log_every', type=int, default=10)

    # ===== 损失形态 =====
    p.add_argument('--loss_type', type=str, default='margin', choices=['margin', 'ce'],
                   help='anchor 分类项：margin(推荐) 或 ce')
    p.add_argument('--lam_class', type=float, default=1.2,
                   help='注入特征与目标类原型对齐的权重')
   
    p.add_argument('--inj_node_budget', type=float, default=0.05,
                   help='采样阶段放宽的边预算倍数（落地仍严格按预算）')
    
     # Anchor scoring strategy copied from TCNIA.
    p.add_argument('--anchor_score_method', type=str, default='causal',
                   choices=['causal', 'ppr', 'bridge_community'],
                   help='Anchor selection strategy: causal vulnerability, PPR, or bridge/community score.')
    p.add_argument('--ppr_alpha', type=float, default=0.85,
                   help='Damping factor used by the PPR anchor scorer.')
    p.add_argument('--ppr_max_iter', type=int, default=100,
                   help='Maximum iterations used by the PPR anchor scorer.')
    p.add_argument('--ppr_tol', type=float, default=1e-6,
                   help='Convergence tolerance used by the PPR anchor scorer.')
    p.add_argument('--community_method', type=str, default='louvain',
                   choices=['louvain', 'greedy'],
                   help='Community detection backend used by bridge_community anchor scoring.')
    #GCN-Jaccard
    p.add_argument('--jaccard_threshold', type=float, default=0.001)
    #Simpgcn
    p.add_argument('--simpgcn_lambda_self', type=float, default=5.0)
    p.add_argument('--simpgcn_gamma', type=float, default=0.1)
    p.add_argument('--simpgcn_k', type=int, default=20)
    p.add_argument('--simpgcn_m_pairs', type=int, default=5)
    p.add_argument(
    '--use_lcc',
    action='store_true',
    default=True,
    help='Keep only the largest connected component before training and attack.'
)
    return p.parse_args()


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
def main():
    args = build_args()
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

    print(len(train_idx),len(val_idx),len(test_idx))
    C = int(labels.max().item() + 1)
    print(f"Class num: {C}")
    victim = Victim(model_name=args.victim, in_dim=feats.shape[1],
                    n_classes=int(labels.max().item()+1),
                    lr=args.lr, wd=args.wd, args=args, device=device)

    print("\n[Stage 0] Train victim on clean graph…")
    print('注入的是:{}网络'.format(args.victim))
    clean = victim.fit_eval(G, feats, labels, train_idx, val_idx, test_idx,
                            epochs=args.epochs, patience=args.patience, verbose=True)
    print(f"Clean test acc: {clean['test_acc']*100:.2f}%")
    print("注入了{}比例的节点".format(args.inj_node_budget))
    # 2) 因果打分 & 锚点
    cscorer = CausalScorer(victim=victim, device=device, alpha=0.6, drop_top=5)
    print("\n[Stage 1] Scoring node vulnerability (causal proxy)…")
    # vul_scores = cscorer.score_vulnerability(G, feats, labels, test_idx, topk= min(G.num_nodes()*args.inj_node_budget, int(test_idx.shape[0])))
    all_idx = torch.arange(G.num_nodes(), device=device, dtype=torch.long)
    
    if args.use_pseudo:
        with torch.no_grad():
            pseudo_logits = victim.predict(G, feats)
        labels_for_sampler = pseudo_logits  # 关键：取 argmax 得到 1D 标签
    else:
        labels_for_sampler = labels

    # 与 TCNIA 对齐：优先在干净模型已正确分类的原始节点中选择 anchor。
    # 若没有正确分类节点，则回退到全体节点，避免空候选集导致实验中断。
    # candidate_nodes = all_idx[(clean_pred[all_idx] == labels[all_idx]).nonzero(as_tuple=True)[0]]
    candidate_nodes = all_idx
    if candidate_nodes.numel() == 0:
        candidate_nodes = all_idx

    topk = max(1, int(round(G.num_nodes() * args.inj_node_budget)))
    # topk = min(topk, int(candidate_nodes.numel()))

    if args.anchor_score_method == "causal":
        cscorer = CausalScorer(victim=victim, device=device, alpha=0.6, drop_top=5)
        vul_scores = cscorer.score_vulnerability(
            G,
            feats,
            labels_for_sampler,
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
        )
    else:
        raise ValueError(f"Unknown anchor_score_method: {args.anchor_score_method}")
    anchors = vul_scores['top_nodes']  # 高易损节点做 anchor
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
    print("边预算为{}".format(args.nes_edge_budget))
    deg_per_node = avg_deg if args.deg_per_node=='auto' else int(args.deg_per_node)
    total_new_edges = inj_nodes * deg_per_node
    print(f"[Budget] inject nodes = ({args.inj_node_budget*100:.1f}%), "
          f"deg per injected = {deg_per_node}, total new edges ≈ {total_new_edges}")
    # 如果命令行没给预算，就用一个小而稳的默认
    if args.nes_edge_budget is None:
        args.nes_edge_budget = max(2, int(0.3 * deg_per_node))  # 例如 0.3*deg_per_node
    print(f"  [Original] graph size N={base_nodes}, E={base_edges} (no structural change)")

    print("\n[Stage 2] Building ER-guided candidate pool…")
    base_pool = get_khop_candidates(
        G=G,
        anchors=anchors,  # Stage 1 得到的脆弱锚点（tensor）
        k_hop_edge=args.khop_edge,  # 例如 1 或 2；0 表示用全图的 1/2-hop 近邻集
        deg_per_node=int(round(0.7*deg_per_node)),  # 你的预算：每个注入点要连多少条边
        fallback_random=True
    )
    print("  Candidate pool ready (k-hop).")
    print_khop_subgraph_stats(G, anchors, k_hop_edge=args.khop_edge)
   
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
    weights = dict(homo=1, degree=1, cos=1)
    candidate = augmstru_enhanced_bernoulli(
        G=G,
        anchors=anchors,
        base_pool=base_pool,
        # K=int(round(0.7*deg_per_node)),
        K=deg_per_node,
        labels=labels_for_sampler,
        feats=feats,
        weights=weights,
        # temp=1.0,
        temp=0.7,
        exact_k=True
    )
    G, inj_ids, anchor_to_injid = stage2b_inject_per_subgraph(
        G=G,
        anchors=anchors,
        candidate=candidate,
        deg_per_node=int(round(0.7*deg_per_node)),
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
    G, inj_pairs = stage3_connect_between_injected(
        G=G,
        inj_ids=inj_ids,
        deg_per_node=deg_per_node,
        intra_ratio=0.3,
        per_node_inner_used=per_node_inner_used,
        undirected=False,  # ← 可写可不写，已忽略
        seed=0,
        verbose=True,
        # alpha_intra=args.alpha_intra,  # 0=自动拟合；>0=固定指数
        alpha_intra=args.alpha_intra if args.alpha_intra > 0 else 2.0,
        tgt_pref="indegree",  # 或 "uniform"
        # tgt_gamma=1.0
        tgt_gamma=1.5
    )

    print(f"[After Stage 3] N={G.num_nodes()}, E={G.num_edges()} (+{G.num_edges() - G_before_E} edges)")
    use_pseudo_after3 = bool(args.use_pseudo)  # 布尔就用布尔，不和字符串比较
    device = feats.device
   
    labels_for_sampler = labels_for_sampler.to(device=feats.device, dtype=torch.long)


    # 类原型（特征对齐用）
    protos = []
    for c in range(C):
        idx = (labels_for_sampler == c).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            protos.append(torch.zeros(feats.size(1), device=device))
        else:
            protos.append(feats[idx].mean(dim=0))
    class_protos = torch.stack(protos, dim=0).to(device)

    print(f"[After Stage 3] labels_for_sampler ready (mode={use_pseudo_after3}).")

    # ===== 2) 锚点-注入映射合并 & 子图抽取（保留你原来的打印，做健壮处理）=====
    merged_by_anchor, injid_to_cands, edge_pairs = merge_anchor_maps(anchor_to_injid, base_pool)

    print(f"共提取子图数量: {len(merged_by_anchor)}")

    G_before_Edge = G.num_edges()

    with torch.no_grad():
        # logits_final = victim.predict(G, feats)
        y_hat = predict_labels_1d(victim, G, feats)
    # y_hat = logits_final.argmax(dim=1)
    gt = labels_for_True.to(y_hat.device)

    # （1）测试集准确率（节点分类）
    if 'test_idx' in locals() and test_idx is not None:
        test_acc_final = (y_hat[test_idx] == gt[test_idx]).float().mean().item()
        print(f"[Final] Test accuracy (GT labels): {test_acc_final * 100:.2f}%")
    
    labels_poison_pre = labels_for_True.to(device)

    train_idx_poison_pre = train_idx.to(device)
    val_idx_poison_pre = val_idx.to(device)
    test_idx_poison_pre = test_idx.to(device)

    # 2) 重新初始化一个 victim；不要复用 clean-trained victim 参数
    pre_nes_poison_victim = Victim(
        model_name=args.victim,
        in_dim=feats.shape[1],
        n_classes=int(labels.max().item() + 1),
        lr=args.lr,
        wd=args.wd,
        args=args,
        device=device,
    )

    # 3) 在进入 NES 前的 injected graph 上重新训练
    pre_nes_poison_result = pre_nes_poison_victim.fit_eval(
        G,
        feats,
        labels_poison_pre,
        train_idx_poison_pre,
        val_idx_poison_pre,
        test_idx_poison_pre,
        epochs=args.epochs,
        patience=args.patience,
        verbose=True,
    )

    print("=" * 80)
    print("[Pre-NES Poison Eval Result]")
    print(f"Clean test acc:        {clean['test_acc'] * 100:.2f}%")
    print(f"Pre-NES poison acc:    {pre_nes_poison_result['test_acc'] * 100:.2f}%")
    print(f"Pre-NES poison drop:   {(clean['test_acc'] - pre_nes_poison_result['test_acc']) * 100:.2f}%")
    print("=" * 80)    
        
        
    from Subgraph_Sparse_Attack import run_subgraph_sparse_attack_nes

    G_adv, feats_adv = run_subgraph_sparse_attack_nes(
        G=G,
        feats=feats,
        victim=victim,
        merged_by_anchor=merged_by_anchor,
        labels_for_sampler=labels_for_sampler,
        class_protos=class_protos,
        args=args,
        T_homo=0.7,
        log_every=args.log_every
    )

    print(f"[After SubgraphSparseAttack] N={G_adv.num_nodes()}, E={G_adv.num_edges()}")
    
    spnia_stealth = spnia_unnoticeability_eval(
    G_clean=G_clean_snapshot,
    G_adv=G_adv,
    feats_clean=X_clean_snapshot,
    feats_adv=feats_adv,
    N0=N0,
    inj_ids=inj_ids,
    epsilon=args.inj_node_budget,
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
    poison_victim = Victim(
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
        y_hat = predict_labels_1d(victim, G, feats)
    # y_hat = logits_final.argmax(dim=1)
    gt = labels_for_True.to(y_hat.device)

    # （1）测试集准确率（节点分类）
    if 'test_idx' in locals() and test_idx is not None:
        test_acc_final = (y_hat[test_idx] == gt[test_idx]).float().mean().item()
        print(f"[Final] Test accuracy (GT labels): {test_acc_final * 100:.2f}%")

    # （2）全图准确率（可选）
    # overall_acc_final = (y_hat == gt).float().mean().item()
    # print(f"[Final] Overall accuracy (GT labels): {overall_acc_final * 100:.2f}%")

    # （3）锚点定向成功率：预测==误导标签（分析指标）
    asr = (y_hat[anchors] != labels_for_True[anchors].to(y_hat.device)).float().mean().item()
    print(f"[Analysis] Targeted success on anchors (toward misguide labels): {asr * 100:.2f}%")

    # （4）结构改动量（相对 Stage-3 前）
    print(f"[Delta] Added edges after sparse attack (approx): {G.num_edges() - G_before_Edge}")
    print(G.num_edges())
    print(G.num_nodes())
if __name__ == '__main__':
    main()
