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
# c2sni/utils.py

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
            assert cand.size(0) == len(anchors_list), \
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


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def random_splits(labels, num_classes, percls_trn, val_lb, seed=42):
    index=[i for i in range(0,labels.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(labels.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]

    train_mask = index_to_mask(train_idx,size=len(labels))
    val_mask = index_to_mask(val_idx,size=len(labels))
    test_mask = index_to_mask(test_idx,size=len(labels))
    return train_mask, val_mask, test_mask

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features



file_dir_citation = os.getcwd() + '/data'
def load_data_citation(dataset_str='cora'):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(file_dir_citation, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(file_dir_citation, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize(features)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # G = nx.from_dict_of_lists(graph, create_using=nx.DiGraph())
    # adj = nx.adjacency_matrix(G)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    # #原来写法
    # D1 = np.array(adj.sum(axis=1)) ** (-0.5)
    # D2 = np.array(adj.sum(axis=0)) ** (-0.5)
    # D1 = sp.diags(D1[:, 0], format='csr')
    # D2 = sp.diags(D2[0, :], format='csr')
    #
    # A = adj.dot(D1)
    # A = D2.dot(A)

    #新写法

    deg_row = np.asarray(adj.sum(axis=1)).reshape(-1)
    deg_col = np.asarray(adj.sum(axis=0)).reshape(-1)
    with np.errstate(divide='ignore'):
        D1_vec = np.power(deg_row, -0.5)
        D2_vec = np.power(deg_col, -0.5)
    D1_vec[~np.isfinite(D1_vec)] = 0.0
    D2_vec[~np.isfinite(D2_vec)] = 0.0
    D1 = sp.diags(D1_vec, format='csr')
    D2 = sp.diags(D2_vec, format='csr')
    A = D2.dot(adj.dot(D1))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]  # onehot

    features = torch.FloatTensor(np.array(features.todense()))
    # features_norm = torch.FloatTensor(np.array(features_norm.todense()))
    labels = torch.LongTensor(np.argmax(labels, -1))
    A = sparse_mx_to_torch_sparse_tensor(A)
    label_classes = labels.numpy()

    # 训练/验证/测试集划分比例
    train_size = 0.1
    val_size = 0.1
    test_size = 0.8

    # 全部节点索引
    all_idx = np.arange(len(label_classes))

    # 先划分训练集
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        all_idx, label_classes, train_size=train_size, stratify=label_classes, random_state=42)

    # 再从剩下的中划分验证与测试
    relative_val_size = val_size / (val_size + test_size)
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, train_size=relative_val_size, stratify=y_temp, random_state=42)

    # 转换成 torch Tensor
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    print(idx_test)
    print(idx_train)
    print(idx_val)
    return A, features, labels, idx_train, idx_val, idx_test, adj
import os
import numpy as np
import scipy.sparse as sp
import torch
import dgl

# ---------- helpers (与“别人”的实现等价/兼容) ----------
def _np_load_any(path, **kwargs):
    """兼容别人代码里的 np.aload；没有就用 np.load(allow_pickle=True)"""
    if hasattr(np, 'aload'):
        return np.aload(path)
    return np.load(path, allow_pickle=True, **kwargs)

def load_npz(file_path):
    """Load a graph2gauss-style NPZ file (used by Cora-ML)."""
    if not file_path.endswith('.npz'):
        file_path += '.npz'
    with _np_load_any(file_path) as loader:
        loader = dict(loader)
        adj = sp.csr_matrix(
            (loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
            shape=loader['adj_shape']
        )
        if 'attr_data' in loader:
            feats = sp.csr_matrix(
                (loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                shape=loader['attr_shape']
            )
        else:
            feats = None
        labels = loader.get('labels')
    return adj, feats, labels

def _split_to_indices_flexible(split_obj):
    """
    兼容 split.npy 的多种保存格式：
    1. dict: {'train': ..., 'val': ..., 'test': ...}
    2. dict: {'idx_train': ..., 'idx_val': ..., 'idx_test': ...}
    3. list/tuple: [train, val, test]
    """
    if isinstance(split_obj, np.ndarray) and split_obj.shape == ():
        split_obj = split_obj.item()

    if isinstance(split_obj, np.ndarray) and split_obj.dtype == object:
        if split_obj.shape == ():
            split_obj = split_obj.item()
        else:
            split_obj = split_obj.tolist()

    def _to_index(x):
        x = np.asarray(x)
        if x.dtype == bool or x.dtype == np.bool_:
            return torch.LongTensor(np.where(x)[0])
        return torch.LongTensor(x.astype(np.int64))

    if isinstance(split_obj, dict):
        train = split_obj.get('train', split_obj.get('idx_train', split_obj.get('train_idx', None)))
        val = split_obj.get('val', split_obj.get('valid', split_obj.get('idx_val', split_obj.get('val_idx', None))))
        test = split_obj.get('test', split_obj.get('idx_test', split_obj.get('test_idx', None)))

        if train is None or val is None or test is None:
            raise KeyError(f"Unknown split keys: {list(split_obj.keys())}")

        return _to_index(train), _to_index(val), _to_index(test)

    if isinstance(split_obj, (list, tuple)) and len(split_obj) >= 3:
        return _to_index(split_obj[0]), _to_index(split_obj[1]), _to_index(split_obj[2])

    if isinstance(split_obj, np.ndarray) and split_obj.ndim >= 1 and len(split_obj) >= 3:
        return _to_index(split_obj[0]), _to_index(split_obj[1]), _to_index(split_obj[2])

    raise ValueError(f"Unsupported split format: type={type(split_obj)}")


def _find_existing_file(candidates):
    for p in candidates:
        if p is not None and os.path.exists(p):
            return p
    raise FileNotFoundError("None of these files exists:\n" + "\n".join(str(x) for x in candidates))


def load_cora_ml_npz_graph(device, args):
    """
    Load Cora-ML from graph2gauss / PyG CitationFull npz format.

    Expected file:
        ./datasets/cora_ml.npz
    Optional split file:
        ./datasets/cora_ml_split.npy

    If split file does not exist, use stratified 10%/10%/80% split,
    consistent with your load_data_citation() setting.
    """
    data_root = getattr(args, "data_root", None)
    if data_root is None:
        data_root = os.path.join(os.getcwd(), "datasets")
    data_root = os.path.expanduser(str(data_root))

    npz_path = _find_existing_file([
        os.path.join(data_root, "cora_ml.npz"),
        os.path.join(data_root, "Cora_ML.npz"),
        os.path.join(os.getcwd(), "datasets", "cora_ml.npz"),
        os.path.join(os.getcwd(), "data", "cora_ml.npz"),
    ])

    split_candidates = [
        os.path.join(data_root, "cora_ml_split.npy"),
        os.path.join(os.getcwd(), "datasets", "cora_ml_split.npy"),
        os.path.join(os.getcwd(), "data", "cora_ml_split.npy"),
    ]
    split_path = None
    for p in split_candidates:
        if os.path.exists(p):
            split_path = p
            break

    # 1. 读取 graph2gauss npz
    adj, features, labels_np = load_npz(npz_path)

    if features is None:
        raise RuntimeError(f"{npz_path} does not contain node features.")
    if labels_np is None:
        raise RuntimeError(f"{npz_path} does not contain labels.")

    # 2. 邻接矩阵处理：对称化，去重。
    # 注意：这里默认不加 self-loop，以保持和你普通 cora/citeseer/pubmed loader 更一致。
    n = adj.shape[0]
    adj = adj.tocsr()
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj.data[:] = 1.0
    adj.setdiag(0)
    adj.eliminate_zeros()

    # 3. 特征处理：row-normalize，和 load_data_citation() 中 features = normalize(features) 对齐。
    if sp.issparse(features):
        features = normalize(features)
        feats = torch.FloatTensor(np.asarray(features.todense(), dtype=np.float32))
    else:
        features = sp.csr_matrix(np.asarray(features, dtype=np.float32))
        features = normalize(features)
        feats = torch.FloatTensor(np.asarray(features.todense(), dtype=np.float32))

    # 4. 标签
    labels_np = np.asarray(labels_np)
    if labels_np.ndim > 1:
        labels_np = labels_np.argmax(axis=1)
    labels = torch.LongTensor(labels_np.astype(np.int64)).view(-1)

    # 5. split：优先读取 cora_ml_split.npy；没有则按 10/10/80 分层随机划分
    if split_path is not None:
        split_obj = np.load(split_path, allow_pickle=True)
        idx_train, idx_val, idx_test = _split_to_indices_flexible(split_obj)
        split_info = f"loaded split={split_path}"
    else:
        label_classes = labels.cpu().numpy()
        all_idx = np.arange(len(label_classes))

        train_size = float(getattr(args, "train_ratio", 0.1))
        val_size = float(getattr(args, "val_ratio", 0.1))
        seed = int(getattr(args, "seed", 42))

        idx_train_np, idx_temp_np, _, y_temp = train_test_split(
            all_idx,
            label_classes,
            train_size=train_size,
            stratify=label_classes,
            random_state=seed,
        )

        relative_val_size = val_size / (1.0 - train_size)

        idx_val_np, idx_test_np = train_test_split(
            idx_temp_np,
            train_size=relative_val_size,
            stratify=y_temp,
            random_state=seed,
        )

        idx_train = torch.LongTensor(idx_train_np)
        idx_val = torch.LongTensor(idx_val_np)
        idx_test = torch.LongTensor(idx_test_np)

        split_info = (
            f"generated stratified split | "
            f"train={train_size:.2f}, val={val_size:.2f}, "
            f"test={1.0 - train_size - val_size:.2f}, seed={seed}"
        )

    # 6. DGL graph
    g = dgl.from_scipy(adj).to(device)

    feats = feats.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    g.ndata["feat"] = feats
    g.ndata["label"] = labels

    # 7. 打印统计
    undirected_edges = adj.nnz // 2
    print(
        f"[Dataset] Cora-ML | "
        f"N={g.num_nodes()}, E_entries={g.num_edges()}, E_undir={undirected_edges}, "
        f"F={feats.size(1)}, C={int(labels.max().item() + 1)} | "
        f"train={idx_train.numel()}, val={idx_val.numel()}, test={idx_test.numel()} | "
        f"{split_info}"
    )

    return g, feats, labels, idx_train, idx_val, idx_test

# ---------- 修改你的入口 ----------
def load_original_graph(dataset: str, device, args):
    dataset_name = dataset.lower().replace('-', '_')

    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        _, feats, labs, tr, va, te, adj = load_data_citation(dataset_name)
        g = dgl.from_scipy(adj).to(device)
        num_edges = g.num_edges()

        print(f"图中的边数: {num_edges}")
        g.ndata['feat']  = feats.to(device)
        g.ndata['label'] = labs.to(device)
        return g, feats.to(device), labs.to(device), tr.to(device), va.to(device), te.to(device)
    
    if dataset_name == "cora_ml":
        return load_cora_ml_npz_graph(device, args)

    raise ValueError(
        f"Unsupported dataset {dataset!r}; choose cora, citeseer, cora_ml, or pubmed."
    )




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
