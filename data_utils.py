# -*- coding: utf-8 -*-
import torch, dgl
import dgl.data
import numpy as np

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
