# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch as th
import dgl
import torch
import scipy.sparse as sp
import numpy as np
import copy
from sklearn.model_selection import train_test_split
import os
import sys
import pickle as pkl
import networkx as nx

def batch_subgraphs_from_injid_to_cands(G: dgl.DGLGraph, injid_to_cands: dict):
    """
    对每个 inj_id 构造一个列表 [inj_id] + cands 并抽子图。
    返回 dict: {inj_id: (subg, nodes_t, nid_map)}
    """
    out = {}
    for inj_id, cands in injid_to_cands.items():
        subg, nodes_t, nid_map = subgraph_from_inj_and_cands(G, inj_id, cands)
        out[int(inj_id)] = (subg, nodes_t, nid_map)
    return out

def subgraph_from_inj_and_cands(G: dgl.DGLGraph, inj_id, cands):
    """
    把 inj_id 和 cands 合到一个列表，一次性从全图 G 抽子图（有向/无向均可）。
    返回：subg（紧凑编号）、nodes_t（列表对应的全局ID）、nid_map（subg节点->全局ID）
    """
    device = G.device if hasattr(G, "device") else torch.device("cpu")
    N = G.num_nodes()

    # 1) 合并成一个列表（去重、过滤非法、保持 inj_id 在最前）
    if isinstance(cands, torch.Tensor):
        cands = cands.detach().cpu().tolist()
    cands = [int(v) for v in cands if 0 <= int(v) < N and int(v) != int(inj_id)]

    # 去重但保序
    seen, c_uniq = set(), []
    for v in cands:
        if v not in seen:
            seen.add(v); c_uniq.append(v)

    node_list = [int(inj_id)] + c_uniq
    nodes_t = torch.tensor(node_list, dtype=torch.long, device=device)

    # 2) 用这个【一个列表】从全图索引子图
    subg = dgl.node_subgraph(G, nodes_t)

    # 3) 子图 -> 全局ID 映射（紧凑编号还原到原图编号）
    nid_map = subg.ndata[dgl.NID].to(device).long()
    return subg, nodes_t, nid_map


def merge_anchor_maps(anchor_to_injid: dict, candidate: dict, ensure_anchor_in_cands: bool = True):
    """
    anchor_to_injid: {anchor -> injected_id}
    candidate:       {anchor -> [c1, c2, ...]}
    ensure_anchor_in_cands: 若 True，确保候选列表里包含 anchor 本人（若无则放到最前）

    返回：
      merged_by_anchor: {anchor: {"inj_id": inj_id or None, "cands": [..]}}
      injid_to_cands:   {inj_id: [..]}
      edge_pairs:       [(inj_id, v), ...]  # 仅对有 inj_id 的 anchor 生成
    """
    # 所有出现过的锚点
    all_anchors = set(anchor_to_injid.keys()) | set(candidate.keys())

    merged_by_anchor = {}
    injid_to_cands = {}
    edge_pairs = []

    for a in sorted(all_anchors):
        inj = anchor_to_injid.get(a, None)
        cands_raw = candidate.get(a, [])

        # 去重但保序
        seen = set()
        cands = []
        for v in cands_raw:
            if v not in seen:
                seen.add(v)
                cands.append(v)

        # 可选：确保候选里包含 anchor 自身
        if ensure_anchor_in_cands and a not in cands:
            cands = [a] + cands

        merged_by_anchor[a] = {"inj_id": inj, "cands": cands}

        if inj is not None:
            injid_to_cands[inj] = cands
            edge_pairs.extend((inj, v) for v in cands)

    return merged_by_anchor, injid_to_cands, edge_pairs

@torch.no_grad()
def accuracy(logits, y):
    return (logits.argmax(-1)==y).float().mean().item()
# Graph and tensor utilities used by TS-SNIA.

import random
from typing import Any, Optional
import numpy as np
import torch
import dgl

__all__ = [
    "set_seed",
    "device_of",
    "to_cpu",
    "to_device",
    "clone_graph_with_data",
]

import dgl
import torch
import dgl
import torch
import numpy as np

def _to_long_1d(x, device, N=None):
    """把多种节点集合表示统一为 1D LongTensor（全局节点ID）。"""
    if isinstance(x, torch.Tensor):
        t = x.to(device)
        if t.dtype == torch.bool:
            idx = torch.nonzero(t, as_tuple=True)[0].long()
            return idx
        return t.long().view(-1)
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.tensor(x, dtype=torch.long, device=device).view(-1)
    if isinstance(x, np.ndarray):
        if x.dtype == bool:
            idx = np.nonzero(x)[0].astype(np.int64)
            return torch.from_numpy(idx).to(device)
        return torch.from_numpy(x.astype(np.int64)).to(device).view(-1)
    # 未识别则返回空
    return torch.empty(0, dtype=torch.long, device=device)

def _normalize_candidate(candidate, anchors_1d: torch.Tensor, device, N):
    """
    规范化为 dict[anchor_id] -> 1D LongTensor(节点ID列表)
    支持:
      - dict: {anchor: list/tensor/mask 或 {chosen/pool/...}}
      - list/tuple: 与 anchors 对齐或通用一份
      - torch.Tensor:
           2D: (num_anchors, K)
           1D: (K,)  —— 所有 anchors 共用同一份
           bool mask: 同上规则
    """
    mapping = {}
    anchors_list = anchors_1d.tolist()

    # dict
    if isinstance(candidate, dict):
        for a in anchors_list:
            item = candidate.get(int(a), [])
            if isinstance(item, dict):
                for key in ['chosen', 'nodes', 'pool', 'cands', 'neighbors']:
                    if key in item:
                        item = item[key]; break
            mapping[int(a)] = _to_long_1d(item, device, N)
        return mapping

    # Tensor
    if isinstance(candidate, torch.Tensor):
        cand = candidate.to(device)
        if cand.dim() == 2:
            assert cand.size(0) == len(anchors_list),\
                f"candidate第一维({cand.size(0)})应与anchors长度({len(anchors_list)})一致"
            for i, a in enumerate(anchors_list):
                row = cand[i]
                if row.dtype == torch.bool:
                    idx = torch.nonzero(row, as_tuple=True)[0].long()
                else:
                    idx = row.long().view(-1)
                mapping[int(a)] = idx
            return mapping
        elif cand.dim() == 1:
            if cand.dtype == torch.bool:
                idx = torch.nonzero(cand, as_tuple=True)[0].long()
            else:
                idx = cand.long().view(-1)
            for a in anchors_list:
                mapping[int(a)] = idx
            return mapping

    # list/tuple
    if isinstance(candidate, (list, tuple)):
        if len(candidate) == len(anchors_list) and len(candidate) > 0 and isinstance(candidate[0], (list, tuple, torch.Tensor, np.ndarray)):
            for i, a in enumerate(anchors_list):
                mapping[int(a)] = _to_long_1d(candidate[i], device, N)
            return mapping
        else:
            idx = _to_long_1d(candidate, device, N)
            for a in anchors_list:
                mapping[int(a)] = idx
            return mapping

    # 兜底：全空
    for a in anchors_list:
        mapping[int(a)] = torch.empty(0, dtype=torch.long, device=device)
    return mapping

def extract_anchor_subgraphs_from_G(
    G: dgl.DGLGraph,
    anchors,
    candidate,
    anchor_to_injid: dict = None,
    include_anchor: bool = True,
    include_injected: bool = False,  # 若要把注入节点也放进子图，置 True 并传入 anchor_to_injid
):
    """
    返回：
      subgraphs: dict[anchor] -> DGL 子图（紧凑编号）
      node_maps: dict[anchor] -> subg.ndata[dgl.NID]（子图节点 -> 原图节点ID）
      used_nodes: dict[anchor] -> 该 anchor 的全局节点ID集合（1D LongTensor）
    """
    device = G.device if hasattr(G, "device") else torch.device("cpu")
    N = G.num_nodes()

    if not isinstance(anchors, torch.Tensor):
        anchors = torch.tensor(list(anchors), dtype=torch.long, device=device)
    else:
        anchors = anchors.to(device).long()

    # 统一 candidate 结构
    cand_map = _normalize_candidate(candidate, anchors, device, N)

    subgraphs, node_maps, used_nodes = {}, {}, {}
    for a in anchors.tolist():
        nodes = cand_map.get(int(a), torch.empty(0, dtype=torch.long, device=device))

        # 追加 anchor / injected（可选）
        extra = []
        if include_anchor:
            extra.append(int(a))
        if include_injected and (anchor_to_injid is not None) and (int(a) in anchor_to_injid):
            extra.append(int(anchor_to_injid[int(a)]))

        if len(extra) > 0:
            extra_t = torch.tensor(extra, dtype=torch.long, device=device)
            nodes = torch.cat([nodes, extra_t], dim=0)

        # 清理非法/重复
        if nodes.numel() > 0:
            nodes = nodes[(nodes >= 0) & (nodes < N)]
            nodes = torch.unique(nodes)

        if nodes.numel() == 0:
            # 至少包含 anchor 本人，避免空子图
            nodes = torch.tensor([int(a)], dtype=torch.long, device=device)

        # 诱导子图 + 原ID映射
        subg = dgl.node_subgraph(G, nodes)
        nid_map = subg.ndata[dgl.NID].to(device).long()

        subgraphs[int(a)] = subg
        node_maps[int(a)] = nid_map
        used_nodes[int(a)] = nodes

    return subgraphs, node_maps, used_nodes









import os
import numpy as np
import scipy.sparse as sp
import torch
import dgl

# ---------- helpers (与“别人”的实现等价/兼容) ----------







# ---------- 修改你的入口 ----------




def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 让 cuDNN 更可复现（如需极致复现可打开下面两行）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False



def _try_tensor_device_from_graph(g: dgl.DGLGraph) -> Optional[torch.device]:
    # 旧版 DGL 没有 g.device；尽量从 ndata 拿一个 tensor 的 device
    try:
        if hasattr(g, "device") and g.device is not None:
            return g.device
    except Exception:
        pass
    try:
        for k, v in g.ndata.items():
            if isinstance(v, torch.Tensor):
                return v.device
    except Exception:
        pass
    return None


def device_of(x: Any, default: str | torch.device = "cpu") -> torch.device:
    """
    返回对象所在 device：
      - torch.Tensor：直接返回 x.device
      - dgl.DGLGraph：优先 g.device，否则从 ndata 的任意张量推断
      - 其他：返回 default
    """
    if isinstance(x, torch.Tensor):
        return x.device
    if isinstance(x, dgl.DGLGraph):
        dev = _try_tensor_device_from_graph(x)
        if dev is not None:
            return dev
    return torch.device(default)


def to_cpu(x: Any):
    """把 Tensor / DGLGraph 安全迁移到 CPU。其他类型原样返回。"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dgl.DGLGraph):
        return x.to(torch.device("cpu"))
    try:
        return x.to("cpu")
    except Exception:
        return x


def to_device(x: Any, dev: str | torch.device):
    """把 Tensor / DGLGraph 迁移到 dev。其他类型尝试 .to，失败则原样返回。"""
    if not isinstance(dev, torch.device):
        dev = torch.device(dev)
    if isinstance(x, torch.Tensor):
        return x.to(dev)
    if isinstance(x, dgl.DGLGraph):
        return x.to(dev)
    try:
        return x.to(dev)
    except Exception:
        return x


def clone_graph_with_data(g: dgl.DGLGraph,
                          feats: Optional[torch.Tensor] = None,
                          labels: Optional[torch.Tensor] = None) -> dgl.DGLGraph:
    """
    复制图结构与数据。兼容旧版 DGL（不用 g.clone()）。
    会复制 ndata/edata（逐 key clone），并在给定时覆盖 'feat'/'label'。
    """
    u, v = g.edges()
    newg = dgl.graph((u, v), num_nodes=g.num_nodes())
    # 复制图级属性
    try:
        for k in g.ndata:
            if isinstance(g.ndata[k], torch.Tensor):
                newg.ndata[k] = g.ndata[k].clone()
            else:
                newg.ndata[k] = g.ndata[k]
    except Exception:
        pass
    try:
        for k in g.edata:
            if isinstance(g.edata[k], torch.Tensor):
                newg.edata[k] = g.edata[k].clone()
            else:
                newg.edata[k] = g.edata[k]
    except Exception:
        pass
    if feats is not None:
        newg.ndata["feat"] = feats.clone()
    if labels is not None:
        newg.ndata["label"] = labels.clone()
    # 尽量放回原 device
    dev = _try_tensor_device_from_graph(g)
    if dev is not None:
        newg = newg.to(dev)
    return newg

# Shared dataset utilities.
from data import (_find_existing_file, _np_load_any, _split_to_indices_flexible, index_to_mask, load_cora_ml_npz_graph, load_data_citation, load_npz, load_original_graph, normalize, parse_index_file, preprocess_features, random_splits, sparse_mx_to_torch_sparse_tensor)
