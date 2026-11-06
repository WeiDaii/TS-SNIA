"""The five empirical defenses and the unified seven-defense interface.

GNNGuard, RobustGCN, ProGNN and GTrans retain the implementations used by
the paper experiments. GCN-Jaccard retains DeepRobust 0.2.11 preprocessing
and GCN training. Original source references are retained below.
Node-Aware and Collective-LP are optional, independent packages, imported
only when explicitly selected. Their reported accuracies are empirical,
not certified robustness bounds.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from types import SimpleNamespace
from typing import Dict

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dgl_nn
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import dropout_edge
import torch_sparse
from torch_sparse import coalesce
from sklearn.preprocessing import normalize
from numba import njit
from tqdm import tqdm
from deeprobust.graph.defense import GCN as _DeepRobustGCN
from deeprobust.graph import utils as _deeprobust_utils

from config import DEFENSES, DEFENSE_NAMES, DEFENSE_TRAINING


# gnnguard: preserved paper-experiment implementation.
def convert_adjacency_matrix(g):
    row, col = g.edges()
    row = row.detach().cpu().numpy()
    col = col.detach().cpu().numpy()
    data = torch.ones(len(row), dtype=torch.float32).numpy()
    adj = sp.csr_matrix((data, (row, col)), shape=(g.num_nodes(), g.num_nodes()))
    return adj


class GNNGuard(nn.Module):

    def __init__(self, threshold: float = 0.1, add_self_loop: bool = False):
        super().__init__()
        self.threshold = threshold
        self.add_self_loop = add_self_loop

    def forward(self, g, feat=None):
        g = g.remove_self_loop()
        if feat is None:
            feat = g.ndata.get('feat', None)
            if feat is None:
                raise ValueError(
                    f"The node feature matrix is not specified, please add argument 'feat' during forward or specify `g.ndata['feat']=feat`"
                )

        row, col = g.edges()
        A = feat[row]
        B = feat[col]
        att_score = F.cosine_similarity(A, B)
        att_score[att_score < self.threshold] = 0.
        # adj = g.adjacency_matrix(scipy_fmt='csr')
        adj = convert_adjacency_matrix(g)
        row, col = row.cpu().tolist(), col.cpu().tolist()
        adj[row, col] = att_score.cpu().tolist()
        adj = normalize(adj, axis=1, norm='l1')
        att_score_norm = torch.tensor(adj[row, col]).to(feat).view(-1)

        if self.add_self_loop:
            degree = (adj != 0).sum(1).A1
            self_weight = torch.tensor(1.0 / (degree + 1)).to(feat)
            att_score_norm = torch.cat([att_score_norm, self_weight])
            g = g.add_self_loop()

        att_score_norm = att_score_norm.exp()
        g.edata['ew'] = att_score_norm

        return g

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}, add_self_loop={self.add_self_loop}"


class GCNGuard(nn.Module):
    """Graph Convolution Network (GCN) with GNNGUARD
    as in `GNNGuard: Defending Graph Neural Networks against Adversarial Attacks`
    ref: https://github.com/EdisonLeeeee/GreatX/blob/dgl/graphwar/defense/model_level/gnnguard.py

    Example
    -------
    # GCNGUARD with one hidden layer
    >>> model = GCNGuard(100, 10)
    # GCNGUARD with two hidden layers
    >>> model = GCNGuard(100, 10, hids=[32, 16])
    # GCNGUARD with two hidden layers, without activation at the first layer
    >>> model = GCNGuard(100, 10, hids=[32, 16])
    """

    def __init__(self, in_feats: int, out_feats: int, hids: list = [16],
                 dropout: float = 0.5, bn: bool = False, bias: bool = True,
                 norm: str = 'both', threshold: float = 0.1, add_self_loop: bool = True):
        r"""
        Parameters
        ----------
        in_feats : int,
            the input dimensions of model
        out_feats : int,
            the output dimensions of model
        hids : list, optional
            the number of hidden units of each hidden layer, by default [16]
        dropout : float, optional
            the dropout ratio of model, by default 0.5
        bias : bool, optional
            whether to use bias in the layers, by default True
        bn: bool, optional
            whether to use `BatchNorm1d` after the convolution layer, by default False
        norm : str, optional
            How to apply the normalizer.  Can be one of the following values:

            * ``both``, where the messages are scaled with :math:`1/c_{ji}`,
            where :math:`c_{ji}` is the product of the square root of node degrees
            (i.e.,  :math:`c_{ji} = \sqrt{|\mathcal{N}(j)|}\sqrt{|\mathcal{N}(i)|}`).

            * ``square``, where the messages are scaled with :math:`1/c_{ji}^2`, where
            :math:`c_{ji}` is defined as above.

            * ``right``, to divide the aggregated messages by each node's in-degrees,
            which is equivalent to averaging the received messages.

            * ``none``, where no normalization is applied.

            * ``left``, to divide the messages sent out from each node by its out-degrees,
            equivalent to random walk normalization.
        threshold: float, optional
            threshold in `GNNGuard` class
        add_self_loop: bool, optional
            add_self_loop in `GNNGuard` class
        """

        super().__init__()

        conv = [GNNGuard(threshold, add_self_loop)]
        for hid in hids:
            conv.append(
                dgl_nn.GraphConv(in_feats, hid, bias=bias, norm=norm, activation=None, allow_zero_in_degree=True)
            )
            if bn:
                conv.append(nn.BatchNorm1d(hid))
            conv.append(nn.ReLU())
            conv.append(nn.Dropout(dropout))
            conv.append(GNNGuard(threshold, add_self_loop))
            in_feats = hid

        # `loc=1` specifies the location of features.
        self.conv1 = nn.Sequential(*conv)
        self.conv2 = dgl_nn.GraphConv(in_feats, out_feats, bias=bias, norm=norm)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    @torch.no_grad()
    def test_forward(self, g, feat):
        g = dgl.add_self_loop(g)
        for layer in self.conv1:
            if isinstance(layer, GNNGuard):
                g = layer(g, feat)
            elif isinstance(layer, dgl_nn.GraphConv):
                feat = layer(g, feat, edge_weight=g.edata['ew'])
            else:
                feat = layer(feat)

        return self.conv2(g, feat, edge_weight=g.edata['ew'])

    def forward(self, g, feat, no_grad=False):
        if no_grad:
            return self.test_forward(g, feat)

        g = dgl.add_self_loop(g)
        for layer in self.conv1:
            if isinstance(layer, GNNGuard):
                g = layer(g, feat)
            elif isinstance(layer, dgl_nn.GraphConv):
                feat = layer(g, feat, edge_weight=g.edata['ew'])
            else:
                feat = layer(feat)

        return self.conv2(g, feat, edge_weight=g.edata['ew'])


# rgcn: preserved paper-experiment implementation.
class RobustGCNConv(nn.Module):
    def __init__(self, in_feats, out_feats, bias=False, gamma=1.0, activation=None):
        super().__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight_mean = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.weight_var = nn.Parameter(torch.Tensor(in_feats, out_feats))

        if bias:
            self.bias_mean = nn.Parameter(torch.Tensor(out_feats))
            self.bias_var = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias_mean', None)
            self.register_parameter('bias_var', None)

        self._gamma = gamma
        self._activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized as in the
        `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__
        where the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
        and the bias is initialized to be zero.

        """
        init.xavier_uniform_(self.weight_mean)
        init.xavier_uniform_(self.weight_var)
        if self.bias_mean is not None:
            init.zeros_(self.bias_mean)
            init.zeros_(self.bias_var)

    def forward(self, graph, feat):
        if not isinstance(feat, tuple):
            feat = (feat, feat)

        mean = torch.matmul(feat[0], self.weight_mean)
        var = torch.matmul(feat[1], self.weight_var)

        if self.bias_mean is not None:
            mean = mean + self.bias_mean
            var = var + self.bias_var

        mean = F.relu(mean)
        var = F.relu(var)

        attention = torch.exp(-self._gamma * var)

        degs = graph.in_degrees().float().clamp(min=1)
        norm1 = torch.pow(degs, -0.5).to(mean.device).unsqueeze(1)
        norm2 = norm1.square()

        with graph.local_scope():
            graph.ndata['mean'] = mean * attention * norm1
            graph.ndata['var'] = var * attention * attention * norm2
            graph.update_all(fn.copy_u('mean', 'm'), fn.sum('m', 'mean'))
            graph.update_all(fn.copy_u('var', 'm'), fn.sum('m', 'var'))

            mean = graph.ndata.pop('mean') * norm1
            var = graph.ndata.pop('var') * norm2

            if self._activation is not None:
                mean = self._activation(mean)
                var = self._activation(var)

        return mean, var


class RobustGCN(nn.Module):
    """
    Robust Graph Convolutional Networks (`RobustGCN <https://pengcui.thumedialab.com/papers/RGCN.pdf>`__)
    """

    def __init__(self, in_features: int, out_features: int,
                 n_hids: list = [16], dropout: float = 0.5, bias: bool = True, gamma: float = 1.0):
        r"""
        Parameters
        ----------
        in_features : int,
            the input dimensions of model
        out_features : int,
            the output dimensions of model
        n_hids : list, optional
            the number of hidden units of each hidden layer, by default [16]
        acts : list, optional
            the activation function of each hidden layer, by default ['relu']
        dropout : float, optional
            the dropout ratio of model, by default 0.5
        bias : bool, optional
            whether to use bias in the layers, by default True
        gamma : float, optional
            the attention weight, by default 1.0
        """

        super().__init__()

        self.mean, self.var = None, None
        assert len(n_hids) > 0
        self.conv1 = RobustGCNConv(in_features, n_hids[0], bias=bias, activation=F.relu)

        conv2 = nn.ModuleList()
        in_features = n_hids[0]
        for hid in n_hids[1:]:
            conv2.append(RobustGCNConv(in_features, hid, bias=bias, gamma=gamma, activation=F.relu))
            in_features = hid

        conv2.append(RobustGCNConv(in_features, out_features, gamma=gamma, bias=bias))
        self.conv2 = conv2
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.conv2:
            conv.reset_parameters()

    @torch.no_grad()
    def test_forward(self, g, x):
        x = self.dropout(x)
        mean, var = self.conv1(g, x)
        self.mean, self.var = mean, var

        for conv in self.conv2:
            mean, var = self.dropout(mean), self.dropout(var)
            mean, var = conv(g, (mean, var))

        std = torch.sqrt(var + 1e-8)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)
        return z

    def forward(self, g, x, no_grad=False):
        if no_grad:
            return self.test_forward(g, x)

        x = self.dropout(x)
        mean, var = self.conv1(g, x)
        self.mean, self.var = mean, var

        for conv in self.conv2:
            mean, var = self.dropout(mean), self.dropout(var)
            mean, var = conv(g, (mean, var))

        std = torch.sqrt(var + 1e-8)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)
        return z


# prognn: preserved paper-experiment implementation.
class ReconAdj(nn.Module):

    def __init__(self, epochs=1, lr=0.01, gamma=1.0):
        super(ReconAdj, self).__init__()
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma

    def recon_edge_weight(self, model, feat, edge_index, train_mask, labels):
        model.eval()

        num_edges = edge_index.shape[1]
        estimated_edge_weight = nn.Parameter(torch.ones((num_edges,), dtype=torch.float32, device=feat.device))
        estimated_edge_weight.requires_grad = True
        optimizer = torch.optim.Adam([estimated_edge_weight], lr=self.lr)
        for idx in range(self.epochs):
            optimizer.zero_grad()
            logits = model(feat, edge_index, edge_weight=estimated_edge_weight)

            loss_gcn = F.cross_entropy(logits[train_mask], labels[train_mask])
            loss_fro = torch.norm(estimated_edge_weight, p='fro')
            loss = - loss_fro + self.gamma * loss_gcn

            loss.backward()
            optimizer.step()
            estimated_edge_weight.data.copy_(torch.clamp(estimated_edge_weight, 0, 1))
        return estimated_edge_weight.detach()


class _ProGNNBackbone(nn.Module):

    def __init__(self, in_feats, h_feats, num_classes):
        super(_ProGNNBackbone, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, num_classes)

    def get_embedding(self, in_feat, edge_index, edge_weight):
        return self.conv1(in_feat, edge_index, edge_weight)

    def forward(self, in_feat, edge_index, edge_weight):
        h = self.conv1(in_feat, edge_index, edge_weight)
        h = F.relu(h)
        h = self.conv2(h, edge_index, edge_weight)
        return h


class ProGNN(nn.Module):
    """ ProGNN (Properties Graph Neural Network).
        See more details in Graph Structure Learning for Robust Graph Neural Networks, KDD 2020, https://arxiv.org/abs/2005.10203.
    """

    def __init__(self, in_feats, h_feats, num_classes, device='cuda'):
        super(ProGNN, self).__init__()
        self.backbone = _ProGNNBackbone(in_feats, h_feats, num_classes).to(device)
        self.recon_model = ReconAdj()

    def forward(self, g, in_feat, no_grad=False):
        # Get the estimated edge weight
        row, col = g.edges()
        edge_index = torch.cat(
            [row.reshape(1, -1), col.reshape(1, -1)],
            dim=0
        )
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        in_feat = in_feat.detach()
        estimated_edge_weight = self.recon_model.recon_edge_weight(self.backbone, in_feat, edge_index, train_mask, labels)
        if no_grad:
            with torch.no_grad():
                return self.backbone(in_feat, edge_index, estimated_edge_weight)
        return self.backbone(in_feat, edge_index, estimated_edge_weight)


# gtrans: preserved paper-experiment implementation.
def linear_to_triu_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
    row_idx = (n - 2 - torch.floor(torch.sqrt(-8 * lin_idx.double() + 4 * n * (n - 1) - 7) / 2.0 - 0.5)).long()
    col_idx = lin_idx + row_idx + 1 - n * (n - 1) // 2 + (n - row_idx) * ((n - row_idx) - 1) // 2
    return torch.stack((row_idx, col_idx))


def inner(t1, t2):
    t1 = t1 / (t1.norm(dim=1).view(-1, 1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1, 1) + 1e-15)
    return (1 - (t1 * t2).sum(1)).mean()


def to_symmetric(edge_index, edge_weight, n, op='mean'):
    symmetric_edge_index = torch.cat(
        (edge_index, edge_index.flip(0)), dim=-1
    )

    symmetric_edge_weight = edge_weight.repeat(2)
    symmetric_edge_index, symmetric_edge_weight = coalesce(
        symmetric_edge_index,
        symmetric_edge_weight,
        m=n,
        n=n,
        op=op
    )
    return symmetric_edge_index, symmetric_edge_weight


def grad_with_checkpoint(outputs, inputs):
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    for input in inputs:
        if not input.is_leaf:
            input.retain_grad()
    torch.autograd.backward(outputs)

    grad_outputs = []
    for input in inputs:
        grad_outputs.append(input.grad.clone())
        input.grad.zero_()
    return grad_outputs


def bisection(edge_weights, a, b, n_perturbations, epsilon=1e-5, iter_max=1e5):
    def func(x):
        return torch.clamp(edge_weights - x, 0, 1).sum() - n_perturbations

    miu = a
    for i in range(int(iter_max)):
        miu = (a + b) / 2
        # Check if middle point is root
        if func(miu) == 0.0:
            break
        # Decide the side to repeat the steps
        if func(miu) * func(a) < 0:
            b = miu
        else:
            a = miu
        if (b - a) <= epsilon:
            break
    return miu


class OptimizeFeatureAndEdge(object):

    def __init__(self, num_nodes, num_features, epochs, loop_feat, loop_adj,
                 ratio=0.1, lr_feat=0.01, lr_adj=0.01, device='cuda'):
        self.ratio = ratio
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.eps = 1e-7
        self.lr_adj, self.lr_feat = lr_adj, lr_feat
        self.device = device
        self.epochs, self.loop_feat, self.loop_adj = epochs, loop_feat, loop_adj

    def __call__(self, model, feat, edge_index, edge_weight, train_mask, labels):
        model.eval()

        n_perturbations = int(self.ratio * edge_index.shape[1] // 2)
        modified_edge_index, perturbed_edge_weight = OptimizeFeatureAndEdge.sample_random_block(self.num_nodes, self.eps, edge_index)
        perturbed_edge_weight.requires_grad = True
        optimizer_adj = torch.optim.Adam([perturbed_edge_weight], lr=self.lr_adj)

        delta_feat = Parameter(torch.FloatTensor(self.num_nodes, self.num_features).to(self.device))
        delta_feat.data.fill_(0)
        optimizer_feat = torch.optim.Adam([delta_feat], lr=self.lr_feat)

        for it in range(self.epochs // (self.loop_adj + self.loop_feat)):
            delta_feat.requires_grad = True
            for j in range(self.loop_feat):
                optimizer_feat.zero_grad()
                cur_feat = feat + delta_feat
                loss = self.test_time_loss(model, cur_feat, edge_index, train_mask, labels, edge_weight)

                loss.backward()

                optimizer_feat.step()
                delta_feat.data = torch.clamp(delta_feat.data, min=-1., max=1.)

            n_feat = (feat + delta_feat).detach()
            for j in range(self.loop_adj):
                perturbed_edge_weight.requires_grad = True

                edge_index, edge_weight = self.get_modified_adj(edge_index, edge_weight, modified_edge_index, perturbed_edge_weight, self.num_nodes)
                loss = self.test_time_loss(model, n_feat, edge_index, train_mask, labels, edge_weight)
                gradient = grad_with_checkpoint(loss, perturbed_edge_weight)[0]
                gradient = torch.where(torch.isnan(gradient), self.eps, gradient)

                with torch.no_grad():
                    optimizer_adj.zero_grad()
                    perturbed_edge_weight.grad = gradient
                    optimizer_adj.step()
                    perturbed_edge_weight.data[perturbed_edge_weight < self.eps] = self.eps
                    perturbed_edge_weight = self.project(n_perturbations, perturbed_edge_weight, self.eps)
                    edge_index, edge_weight = self.get_modified_adj(edge_index, edge_weight, modified_edge_index, perturbed_edge_weight, self.num_nodes)

        n_feat = feat + delta_feat
        return n_feat.detach(), edge_index.detach(), edge_weight.detach()

    def project(self, n_perturbations, values, eps, inplace=False, bisec=True):
        if not inplace:
            values = values.clone()

        if not bisec:
            values.data.copy_(torch.clamp(
                values, min=eps, max=1 - eps
            ))
            return values

        if torch.clamp(values, 0, 1).sum() > n_perturbations:
            left = (values - 1).min()
            right = values.max()
            miu = bisection(values, left, right, n_perturbations)
            values.data.copy_(torch.clamp(
                values - miu, min=eps, max=1 - eps
            ))
        else:
            values.data.copy_(torch.clamp(
                values, min=eps, max=1 - eps
            ))
        return values

    def get_modified_adj(self, edge_index, edge_weight, modified_edge_index, perturbed_edge_weight, num_nodes):
        modified_edge_index, modified_edge_weight = to_symmetric(
            modified_edge_index, perturbed_edge_weight, num_nodes
        )
        edge_index = torch.cat((edge_index.to(self.device), modified_edge_index), dim=-1)
        edge_weight = torch.cat((edge_weight.to(self.device), modified_edge_weight))

        edge_index, edge_weight = torch_sparse.coalesce(edge_index, edge_weight, m=num_nodes, n=num_nodes, op='sum')

        # Allow removal of edges
        edge_weight[edge_weight >= 1] = 2 - edge_weight[edge_weight >= 1]
        return edge_index, edge_weight

    def test_time_loss(self, model, feat, edge_index, train_mask, labels, edge_weight=None):
        output = model(feat, edge_index, edge_weight)
        loss = F.cross_entropy(output[train_mask], labels[train_mask])

        output1 = OptimizeFeatureAndEdge.augment(model, feat, strategy='dropedge', p=0.5, edge_index=edge_index, edge_weight=edge_weight)
        output2 = OptimizeFeatureAndEdge.augment(model, feat, strategy='dropedge', p=0.0, edge_index=edge_index, edge_weight=edge_weight)
        output3 = OptimizeFeatureAndEdge.augment(model, feat, strategy='shuffle', edge_index=edge_index, edge_weight=edge_weight)
        loss += (inner(output1, output2) - inner(output2, output3))
        return loss

    @staticmethod
    def augment(model, feat, strategy='dropedge', p=0.5, edge_index=None, edge_weight=None):
        output = None
        if strategy == 'shuffle':
            idx = np.random.permutation(feat.shape[0])
            shuf_fts = feat[idx, :]
            output = model.get_embedding(shuf_fts, edge_index, edge_weight)
        if strategy == "dropedge":
            edge_index, edge_mask = dropout_edge(edge_index, p=p)
            edge_weight = edge_weight[edge_mask]
            output = model.get_embedding(feat, edge_index, edge_weight)
        return output

    @staticmethod
    def sample_random_block(n, eps, edge_index):
        edge_index = edge_index.clone()
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]

        row, col = edge_index[0], edge_index[1]
        edge_index_id = (2 * n - row - 1) * row // 2 + col - row - 1  # // is important to get the correct result
        edge_index_id = edge_index_id.long()
        current_search_space = edge_index_id

        modified_edge_index = linear_to_triu_idx(n, current_search_space)
        perturbed_edge_weight = torch.full_like(current_search_space, eps, dtype=torch.float32, requires_grad=True)
        return modified_edge_index, perturbed_edge_weight


class _GTransBackbone(nn.Module):

    def __init__(self, in_feats, h_feats, num_classes):
        super(_GTransBackbone, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, num_classes)

    def get_embedding(self, in_feat, edge_index, edge_weight):
        return self.conv1(in_feat, edge_index, edge_weight)

    def forward(self, in_feat, edge_index, edge_weight):
        h = self.conv1(in_feat, edge_index, edge_weight)
        h = F.relu(h)
        h = self.conv2(h, edge_index, edge_weight)
        return h


class GTrans(nn.Module):

    def __init__(self, in_feats, h_feats, num_classes, epochs=2, loop_feat=1, loop_adj=1):
        super(GTrans, self).__init__()
        self.in_feats, self.h_feats = in_feats, h_feats
        self.backbone = _GTransBackbone(in_feats, h_feats, num_classes)
        self.purify_model = None
        self.epochs, self.loop_feat, self.loop_adj = epochs, loop_feat, loop_adj

    def purify_graph(self, g, in_feat):
        num_nodes = in_feat.shape[0]
        num_features = self.in_feats
        self.purify_model = OptimizeFeatureAndEdge(num_nodes, num_features, self.epochs, self.loop_feat, self.loop_adj, device=in_feat.device)

        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']

        row, col = g.edges()
        edge_index = torch.cat(
            [row.reshape(1, -1), col.reshape(1, -1)],
            dim=0
        )
        num_edges = edge_index.shape[1]
        edge_weight = torch.ones(size=(num_edges,), dtype=torch.float32, device=edge_index.device)

        in_feat, edge_index, edge_weight, train_mask, labels = in_feat.detach(), edge_index.detach(), edge_weight.detach(), \
            train_mask.detach(), labels.detach()
        n_feat, n_edge_index, n_edge_weight = self.purify_model(self.backbone, in_feat, edge_index, edge_weight, train_mask, labels)
        return n_feat, n_edge_index, n_edge_weight

    def forward(self, g, in_feat, no_grad=False):
        in_feat, edge_index, edge_weight = self.purify_graph(g, in_feat)
        if no_grad:
            with torch.no_grad():
                return self.backbone(in_feat, edge_index, edge_weight)
        return self.backbone(in_feat, edge_index, edge_weight)


# GCN-Jaccard: DeepRobust 0.2.11, preprocessing kept unchanged.
# MIT License
# Copyright (c) 2020 Yaxin Li, Wei Jin, Han Xu and Jiliang Tang.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
class GCNJaccard(_DeepRobustGCN):
    """GCNJaccard first preprocesses input graph via droppining dissimilar
    edges and train a GCN based on the processed graph. See more details in
    Adversarial Examples on Graph Data: Deep Insights into Attack and Defense,
    https://arxiv.org/pdf/1903.01610.pdf.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    dropout : float
        dropout rate for GCN
    lr : float
        learning rate for GCN
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN. When `with_relu` is True, `weight_decay` will be set to 0.
    with_relu : bool
        whether to use relu activation function. If False, GCN will be linearized.
    with_bias: bool
        whether to include bias term in GCN weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
	We can first load dataset and then train GCNJaccard.

    >>> from deeprobust.graph.data import PrePtbDataset, Dataset
    >>> from deeprobust.graph.defense import GCNJaccard
    >>> # load clean graph data
    >>> data = Dataset(root='/tmp/', name='cora', seed=15)
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # load perturbed graph data
    >>> perturbed_data = PrePtbDataset(root='/tmp/', name='cora')
    >>> perturbed_adj = perturbed_data.adj
    >>> # train defense model
    >>> model = GCNJaccard(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu').to('cpu')
    >>> model.fit(features, perturbed_adj, labels, idx_train, idx_val, threshold=0.03)

    """
    def __init__(self, nfeat, nhid, nclass, binary_feature=True, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, device='cpu'):

        super(GCNJaccard, self).__init__(nfeat, nhid, nclass, dropout, lr, weight_decay, with_relu, with_bias, device=device)
        self.device = device
        self.binary_feature = binary_feature

    def fit(self, features, adj, labels, idx_train, idx_val=None, threshold=0.01, train_iters=200, initialize=True, verbose=True, **kwargs):
        """First drop dissimilar edges with similarity smaller than given
        threshold and then train the gcn model on the processed graph.
        When idx_val is not None, pick the best model according to the
        validation loss.

        Parameters
        ----------
        features :
            node features. The format can be numpy.array or scipy matrix
        adj :
            the adjacency matrix.
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        threshold : float
            similarity threshold for dropping edges. If two connected nodes with similarity smaller than threshold, the edge between them will be removed.
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        """

        self.threshold = threshold
        modified_adj = self.drop_dissimilar_edges(features, adj)
        # modified_adj_tensor = _deeprobust_utils.sparse_mx_to_torch_sparse_tensor(self.modified_adj)
        features, modified_adj, labels = _deeprobust_utils.to_tensor(features, modified_adj, labels, device=self.device)
        self.modified_adj = modified_adj
        self.features = features
        self.labels = labels
        super().fit(features, modified_adj, labels, idx_train, idx_val, train_iters=train_iters, initialize=initialize, verbose=verbose)

    def drop_dissimilar_edges(self, features, adj, metric='similarity'):
        """Drop dissimilar edges.(Faster version using numba)
        """
        if not sp.issparse(adj):
            adj = sp.csr_matrix(adj)

        adj_triu = sp.triu(adj, format='csr')

        if sp.issparse(features):
            features = features.todense().A # make it easier for njit processing

        if metric == 'distance':
            removed_cnt = dropedge_dis(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=self.threshold)
        else:
            if self.binary_feature:
                removed_cnt = dropedge_jaccard(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=self.threshold)
            else:
                removed_cnt = dropedge_cosine(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=self.threshold)
        print('removed %s edges in the original graph' % removed_cnt)
        modified_adj = adj_triu + adj_triu.transpose()
        return modified_adj

    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized adjacency

        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.


        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCNJaccard
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            adj = self.drop_dissimilar_edges(features, adj)
            if type(adj) is not torch.Tensor:
                features, adj = _deeprobust_utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if _deeprobust_utils.is_sparse_tensor(adj):
                self.adj_norm = _deeprobust_utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = _deeprobust_utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)

    def _drop_dissimilar_edges(self, features, adj):
        """Drop dissimilar edges. (Slower version)
        """
        if not sp.issparse(adj):
            adj = sp.csr_matrix(adj)
        modified_adj = adj.copy().tolil()

        # preprocessing based on features
        print('=== GCN-Jaccrad ===')
        edges = np.array(modified_adj.nonzero()).T
        removed_cnt = 0
        for edge in tqdm(edges):
            n1 = edge[0]
            n2 = edge[1]
            if n1 > n2:
                continue

            if self.binary_feature:
                J = self._jaccard_similarity(features[n1], features[n2])

                if J < self.threshold:
                    modified_adj[n1, n2] = 0
                    modified_adj[n2, n1] = 0
                    removed_cnt += 1
            else:
                # For not binary feature, use cosine similarity
                C = self._cosine_similarity(features[n1], features[n2])
                if C < self.threshold:
                    modified_adj[n1, n2] = 0
                    modified_adj[n2, n1] = 0
                    removed_cnt += 1
        print('removed %s edges in the original graph' % removed_cnt)
        return modified_adj

    def _jaccard_similarity(self, a, b):
        intersection = a.multiply(b).count_nonzero()
        J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)
        return J

    def _cosine_similarity(self, a, b):
        inner_product = (a * b).sum()
        C = inner_product / (np.sqrt(np.square(a).sum()) * np.sqrt(np.square(b).sum()) + 1e-10)
        return C


@njit
def dropedge_jaccard(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            intersection = np.count_nonzero(a*b)
            J = intersection * 1.0 / (np.count_nonzero(a) + np.count_nonzero(b) - intersection)

            if J < threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt


@njit
def dropedge_cosine(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            inner_product = (a * b).sum()
            C = inner_product / (np.sqrt(np.square(a).sum()) * np.sqrt(np.square(b).sum()) + 1e-8)

            if C < threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt


@njit
def dropedge_dis(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            C = np.linalg.norm(features[n1] - features[n2])
            if C > threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1

    return removed_cnt


@dataclass
class DefenseTrainingResult:
    clean_or_poison: str
    defense: str
    best_val_acc: float | None
    test_acc: float
    epochs_ran: int


def _seed_defense(seed):
    from data import set_seed
    set_seed(int(seed))
    dgl.seed(int(seed))
    dgl.random.seed(int(seed))


@lru_cache(maxsize=2)
def _smoothing_adapter(name):
    """Import only the explicitly requested optional defense package."""
    package = {'node_aware': 'Node_Aware', 'collective_lp': 'Collective_LP'}[name]
    try:
        return import_module(package + '.model')
    except ModuleNotFoundError as error:
        if error.name in {package, package + '.model'}:
            raise ImportError(
                f'{name} is optional and requires the {package}/ directory. '
                f'The default defenses are: {", ".join(DEFENSES)}.'
            ) from error
        raise


def build_defense_model(name, in_dim, out_dim, hidden=64, dropout=0.5,
                        threshold=0.1, device=None):
    """Build one of the local GNN defense implementations."""
    if name not in DEFENSE_NAMES:
        raise ValueError(f'Unknown defense {name!r}; choose from {DEFENSE_NAMES}')
    if name == 'gcn_jaccard':
        return GCNJaccard(nfeat=in_dim, nhid=hidden, nclass=out_dim,
                          dropout=dropout, device=str(device or 'cpu'))
    if name in {'node_aware', 'collective_lp'}:
        model = _smoothing_adapter(name).MODEL(in_dim, hidden, out_dim, dropout)
        return model.to(device or 'cpu')
    if name == 'gnnguard':
        model = GCNGuard(in_dim, out_dim, hids=[hidden], dropout=dropout, threshold=threshold)
    elif name == 'rgcn':
        model = RobustGCN(in_dim, out_dim, n_hids=[hidden], dropout=dropout)
    elif name == 'prognn':
        model = ProGNN(in_feats=in_dim, h_feats=hidden, num_classes=out_dim, device=str(device or 'cpu'))
    else:  # gtrans
        model = GTrans(in_feats=in_dim, h_feats=hidden, num_classes=out_dim, epochs=2, loop_feat=1, loop_adj=1)
    return model.to(device) if device is not None else model


def train_defense_model(name, graph, features, labels, idx_train, idx_val, idx_test,
                        epochs=200, patience=30, lr=0.01, weight_decay=5e-4,
                        hidden=64, dropout=0.5, threshold=0.1, tag='poison',
                        *, seed=None, jaccard_args=None):
    """Keep the paper training protocol and the unified call signature."""
    if name not in DEFENSE_NAMES:
        raise ValueError(f'Unsupported defense: {name}')
    if seed is not None:
        _seed_defense(seed)
    if name == 'gcn_jaccard':
        from models import GNNModelWrapper
        args = copy.copy(jaccard_args) if jaccard_args is not None else SimpleNamespace()
        # Preserve the paper's hidden=256 and threshold=0.001 for this defense.
        args.hidden = getattr(args, 'hidden', 256)
        args.jaccard_threshold = getattr(args, 'jaccard_threshold', 0.001)
        model = GNNModelWrapper(name, int(features.shape[1]), int(labels.max())+1,
                                lr, weight_decay, args, features.device)
        result = model.fit_eval(graph, features, labels, idx_train, idx_val, idx_test,
                                epochs=epochs, patience=patience, verbose=False)
        return model, DefenseTrainingResult(tag, name, None,
                                            float(result['test_acc']), int(result['epochs_trained']))
    if name in {'node_aware', 'collective_lp'}:
        adapter = _smoothing_adapter(name)
        settings = adapter.DEFAULT_CONFIG
        upper = adapter.unique_upper_edges(graph, features.device)
        model, info = adapter.train_smoothed(
            features, labels, upper, idx_train, idx_val, seed=int(seed or 0),
            p_e=settings['p_e'], p_n=settings['p_n'], epochs=epochs,
            patience=patience, validation_interval=10, hidden=hidden,
            dropout=dropout, lr=lr, weight_decay=weight_decay,
        )
        model.eval()
        with torch.no_grad():
            edge_index = torch.cat((upper, upper.flip(0)), dim=1)
            accuracy = adapter.accuracy(model(features, edge_index), labels, idx_test)
        return model, DefenseTrainingResult(tag, name, info['best_validation_accuracy'],
                                            float(accuracy), int(info['epochs_ran']))
    return _train_empirical_defense(name, graph, features, labels, idx_train, idx_val, idx_test,
                 epochs, patience, lr, weight_decay, hidden, dropout, threshold, tag)


def evaluate_defense_pairs(clean_graph, clean_features, clean_labels,
                           poison_graph, poison_features, poison_labels,
                           train_idx, val_idx, test_idx, *, seed, names=None,
                           epochs=200, patience=30, hidden=64, jaccard_args=None):
    rows = []
    for name in names or DEFENSES:
        results = []
        for phase, graph, features, labels in (
            ('clean', clean_graph, clean_features, clean_labels),
            ('poison', poison_graph, poison_features, poison_labels),
        ):
            _, result = train_defense_model(
                name, graph.clone(), features, labels, train_idx, val_idx, test_idx,
                epochs, patience, DEFENSE_TRAINING['lr'], DEFENSE_TRAINING['weight_decay'],
                hidden, DEFENSE_TRAINING['dropout'], DEFENSE_TRAINING['threshold'], phase,
                seed=seed, jaccard_args=jaccard_args,
            )
            results.append(result)
        row = {'defense':name, 'seed':int(seed),
               'clean_accuracy':results[0].test_acc, 'poison_accuracy':results[1].test_acc,
               'accuracy_drop':results[0].test_acc-results[1].test_acc,
               'clean_epochs_ran':results[0].epochs_ran, 'poison_epochs_ran':results[1].epochs_ran,
               'evaluation':'empirical_accuracy', 'victim_queries_during_attack':0}
        rows.append(row)
        print(f'[Defense] {name}: clean={row["clean_accuracy"]:.6f}, '
              f'poison={row["poison_accuracy"]:.6f}', flush=True)
    return rows


def evaluate_saved_artifact(args):
    """Train defended models on a saved attack while verifying the data split."""
    from data import load_original_graph, apply_lcc_to_loaded_dgl_data, set_seed
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    clean = load_original_graph(args.dataset, device, args)
    if args.use_lcc:
        clean = apply_lcc_to_loaded_dgl_data(*clean, device=device)
    clean_graph, clean_features, clean_labels, train_idx, val_idx, test_idx = clean
    graphs, metadata = dgl.load_graphs(str(args.artifact_input))
    poison_graph = graphs[0].to(device)
    for key, expected in [('train_idx',train_idx),('val_idx',val_idx),('test_idx',test_idx)]:
        if not torch.equal(metadata[key].cpu(), expected.cpu()):
            raise ValueError(f'Artifact {key} does not match the requested clean-data split.')
    if int(metadata['original_nodes'].item()) != clean_graph.num_nodes():
        raise ValueError('Artifact original-node count does not match the clean graph.')
    names = args.defenses or ([args.defense] if args.defense else None)
    poison_labels = torch.zeros(poison_graph.num_nodes(), dtype=torch.long, device=device)
    poison_labels[:clean_labels.numel()] = clean_labels
    rows = evaluate_defense_pairs(
        clean_graph, clean_features, clean_labels, poison_graph, poison_graph.ndata['feat'],
        poison_labels, train_idx, val_idx, test_idx, seed=args.seed, names=names,
        epochs=args.defense_epochs, patience=args.defense_patience,
        hidden=args.defense_hidden, jaccard_args=args,
    )
    return {'status':'OK','method':'TS-SNIA','dataset':args.dataset,'victim':args.victim,
            'seed':args.seed,'artifact':str(args.artifact_input),'defense_results':rows,
            'original_nodes':clean_graph.num_nodes(),'split_verified':True}


def _accuracy(logits: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor) -> float:
    if idx.numel() == 0:
        return 0.0
    pred = logits[idx].argmax(dim=-1)
    return float((pred == labels[idx]).float().mean().item())


def _forward(model: torch.nn.Module, graph: dgl.DGLGraph, features: torch.Tensor, no_grad: bool = False):
    try:
        return model(graph, features, no_grad=no_grad)
    except TypeError:
        return model(graph, features)


def _needs_grad_forward(defense: str) -> bool:
    return defense.lower() in {"prognn", "gtrans"}


def _prepare_graph(
    graph: dgl.DGLGraph,
    defense: str,
    idx_train: torch.Tensor | None = None,
    idx_val: torch.Tensor | None = None,
    idx_test: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> dgl.DGLGraph:
    graph = dgl.remove_self_loop(graph)
    if defense in {"rgcn", "prognn", "gtrans"}:
        graph = dgl.add_self_loop(graph)
    if idx_train is not None:
        train_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=idx_train.device)
        val_mask = torch.zeros_like(train_mask)
        test_mask = torch.zeros_like(train_mask)
        train_mask[idx_train.long()] = True
        if idx_val is not None:
            val_mask[idx_val.long()] = True
        if idx_test is not None:
            test_mask[idx_test.long()] = True
        graph.ndata["train_mask"] = train_mask
        graph.ndata["val_mask"] = val_mask
        graph.ndata["test_mask"] = test_mask
    if labels is not None:
        graph.ndata["label"] = labels
    return graph


def _train_empirical_defense(
    defense: str,
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    labels: torch.Tensor,
    idx_train: torch.Tensor,
    idx_val: torch.Tensor,
    idx_test: torch.Tensor,
    epochs: int = 200,
    patience: int = 30,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    hidden: int = 64,
    dropout: float = 0.5,
    threshold: float = 0.1,
    tag: str = "poison",
) -> tuple[torch.nn.Module, DefenseTrainingResult]:
    device = features.device
    defense = defense.lower()
    graph = _prepare_graph(graph.to(device), defense, idx_train, idx_val, idx_test, labels)
    model = build_defense_model(
        name=defense,
        in_dim=int(features.size(1)),
        out_dim=int(labels.max().item() + 1),
        hidden=hidden,
        dropout=dropout,
        threshold=threshold,
        device=device,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state: Dict[str, torch.Tensor] | None = None
    best_val = -1.0
    bad = 0
    epochs_ran = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        opt.zero_grad()
        logits = _forward(model, graph, features, no_grad=False)
        loss = F.cross_entropy(logits[idx_train], labels[idx_train])
        loss.backward()
        opt.step()

        model.eval()
        if _needs_grad_forward(defense):
            logits_eval = _forward(model, graph, features, no_grad=False)
            val_acc = _accuracy(logits_eval.detach(), labels, idx_val)
        else:
            with torch.no_grad():
                logits_eval = _forward(model, graph, features, no_grad=True)
                val_acc = _accuracy(logits_eval, labels, idx_val)
        epochs_ran = epoch
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    if _needs_grad_forward(defense):
        logits = _forward(model, graph, features, no_grad=False)
        test_acc = _accuracy(logits.detach(), labels, idx_test)
    else:
        with torch.no_grad():
            logits = _forward(model, graph, features, no_grad=True)
            test_acc = _accuracy(logits, labels, idx_test)
    result = DefenseTrainingResult(
        clean_or_poison=tag,
        defense=defense,
        best_val_acc=float(best_val),
        test_acc=float(test_acc),
        epochs_ran=int(epochs_ran),
    )
    return model, result
