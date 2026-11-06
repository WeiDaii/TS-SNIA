"""Graph loading and the existing TS-SNIA data-split and random-seed protocol."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
import copy
import os
import sys
import pickle as pkl
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch as th
import dgl
import dgl.data
import networkx as nx
from sklearn.model_selection import train_test_split
from config import DATA_ROOT

file_dir_citation = str(DATA_ROOT)

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

    if dataset_name in {"amazon_computers", "amazon_photo"}:
        from dgl.data import AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset

        dataset_cls = AmazonCoBuyComputerDataset if dataset_name == "amazon_computers" else AmazonCoBuyPhotoDataset
        raw_dir = str(DATA_ROOT / "dgl_amazon")
        graph = dataset_cls(raw_dir=raw_dir, verbose=True)[0]
        graph = dgl.to_simple(dgl.to_bidirected(graph, copy_ndata=True))
        features = graph.ndata["feat"].float()
        labels = graph.ndata["label"].long()

        # Deterministic class-stratified 10/10/80 split.  The same loader is
        # used by TS-SNIA and every baseline, so all methods receive one split.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(getattr(args, "seed", 0)))
        train_parts, val_parts, test_parts = [], [], []
        for class_id in range(int(labels.max().item()) + 1):
            members = (labels.cpu() == class_id).nonzero(as_tuple=True)[0]
            members = members[torch.randperm(int(members.numel()), generator=generator)]
            n_train = max(1, int(round(0.10 * members.numel())))
            n_val = max(1, int(round(0.10 * members.numel())))
            train_parts.append(members[:n_train])
            val_parts.append(members[n_train:n_train + n_val])
            test_parts.append(members[n_train + n_val:])
        train_idx = torch.cat(train_parts).long()
        val_idx = torch.cat(val_parts).long()
        test_idx = torch.cat(test_parts).long()

        graph = graph.to(device)
        features = features.to(device)
        labels = labels.to(device)
        train_idx = train_idx.to(device)
        val_idx = val_idx.to(device)
        test_idx = test_idx.to(device)
        graph.ndata["feat"] = features
        graph.ndata["label"] = labels
        print(
            f"[Dataset] {dataset_name} | N={graph.num_nodes()}, E={graph.num_edges()}, "
            f"F={features.shape[1]}, C={int(labels.max().item() + 1)} | "
            f"train={train_idx.numel()}, val={val_idx.numel()}, test={test_idx.numel()}"
        )
        return graph, features, labels, train_idx, val_idx, test_idx

    raise ValueError(
        f"Unsupported dataset {dataset!r}."
    )

def set_seed(seed):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_dgl_dataset(name='cora', device='cpu'):
    if name=='cora':
        ds = dgl.data.CoraGraphDataset(); G = ds[0]
    elif name=='citeseer':
        ds = dgl.data.CiteseerGraphDataset(); G = ds[0]
    elif name=='pubmed':
        ds = dgl.data.PubmedGraphDataset(); G = ds[0]
    else:
        raise ValueError(
            f"Unsupported dataset {name!r}; choose cora, citeseer, cora_ml, or pubmed."
        )
    G = dgl.remove_self_loop(G); G = dgl.add_self_loop(G)
    feats = G.ndata['feat'].to(device)
    labels = G.ndata['label'].to(device)
    train_mask = G.ndata['train_mask']; val_mask = G.ndata['val_mask']; test_mask = G.ndata['test_mask']
    tr = torch.nonzero(train_mask, as_tuple=False).view(-1).to(device)
    va = torch.nonzero(val_mask, as_tuple=False).view(-1).to(device)
    te = torch.nonzero(test_mask, as_tuple=False).view(-1).to(device)
    G = G.to(device)
    return G, feats, labels, tr, va, te

def clone_graph_with_data(G, feats, labels):
    H = dgl.graph((G.edges()[0], G.edges()[1]), num_nodes=G.num_nodes(), device=G.device)
    H.ndata['feat'] = feats.clone()
    H.ndata['label'] = labels.clone()
    return H

def to_device(x, device): return x.to(device)

def to_cpu(x): return x.detach().cpu()

def device_of(G): return G.device


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
