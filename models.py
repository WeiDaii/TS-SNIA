# -*- coding: utf-8 -*-
"""
Shared GNN wrapper: fixed attack surrogate and independently retrained evaluation victim.
"""
import numpy as np
import scipy.sparse as sp
import torch, torch.nn as nn, torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from utils import accuracy
from gnn_layers import dgl_gat, dgl_gcn, dgl_sage, dgl_appnp, dgl_gin, dgl_sgc, dgl_hang,dgl_evennet,dgl_simpgcn
# from GEN_utils  import load_data_citation, random_splits
from dgl.nn.pytorch.conv import GATConv, GraphConv, GINConv, APPNPConv, SAGEConv, SGConv

class dgl_gcn(nn.Module):
    def __init__(self, input_dim, nhidden, nclasses):
        super(dgl_gcn, self).__init__()
        self.layer1 = GraphConv(in_feats=input_dim, out_feats=nhidden, allow_zero_in_degree=True)
        self.layer2 = GraphConv(in_feats=nhidden, out_feats=nclasses, allow_zero_in_degree=True)

    def forward(self, g, features):
        x = self.layer1(g, features)
        x = self.layer2(g, x)

        return x

class dgl_gat(nn.Module):
    def __init__(self, input_dim, out_dim, num_heads, num_classes, dropout):
        super(dgl_gat, self).__init__()
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.dropout = dropout
        self.layer1 = GATConv(input_dim, out_dim, num_heads[0], feat_drop=dropout[0], attn_drop=dropout[1],
                              activation=None, allow_zero_in_degree=True, negative_slope=1.)
        self.layer2 = GATConv(num_heads[0] * out_dim, num_classes, num_heads[1], feat_drop=dropout[0], attn_drop=dropout[1],
                              activation=None, allow_zero_in_degree=True, negative_slope=1.)

    def forward(self, graph, feat):
        x1 = self.layer1(graph, feat)  # input_dim * num_heads[0] * out_dim
        x1 = x1.flatten(1)
        x1 = F.elu(x1)

        x1 = self.layer2(graph, x1)
        x1 = x1.squeeze(1)
        if self.num_heads[1] > 1:
            x1 = torch.mean(x1, dim=1)
        x1 = F.elu(x1)

        return x1
class dgl_sage(nn.Module):
    def __init__(self, input_dim, nhidden, aggregator_type, nclasses):
        super(dgl_sage, self).__init__()
        self.layer1 = SAGEConv(in_feats=input_dim, out_feats=nhidden, aggregator_type=aggregator_type)
        self.layer2 = SAGEConv(in_feats=nhidden, out_feats=nclasses, aggregator_type=aggregator_type)

    def forward(self, g, features):
        x = self.layer1(g, features)
        x = self.layer2(g, x)
        return x


class dgl_appnp(nn.Module):
    def __init__(self, input_dim, hidden, classes, k, alpha):
        super(dgl_appnp, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, classes)
        self.layer1 = APPNPConv(k=k, alpha=alpha, edge_drop=0.5)
        self.layer2 = APPNPConv(k=k, alpha=alpha, edge_drop=0.5)
        self.set_parameters()

    def set_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc1.weight, gain=gain)
        nn.init.xavier_normal_(self.fc2.weight, gain=gain)

    def forward(self, g, features):
        features = self.fc1(features)
        x = self.layer1(g, features)
        x = F.elu(self.fc2(x))
        x = self.layer2(g, x)
        x = F.elu(x)
        return x


class dgl_gin(nn.Module):
    def __init__(self, input_dim, hidden, classes, aggregator_type):
        super(dgl_gin, self).__init__()
        self.apply_func1 = nn.Linear(input_dim, hidden)
        self.apply_func2 = nn.Linear(hidden, classes)
        self.layer1 = GINConv(apply_func=self.apply_func1, aggregator_type=aggregator_type)
        self.layer2 = GINConv(apply_func=self.apply_func2, aggregator_type=aggregator_type)
        self.set_parameters()

    def set_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.apply_func1.weight, gain=gain)
        nn.init.xavier_normal_(self.apply_func2.weight, gain=gain)

    def forward(self, g_list, features):
            g = g_list
            x = self.layer1(g, features)
            x = F.elu(x)
            x = self.layer2(g, x)
            x = F.elu(x)
            return x


class dgl_sgc(nn.Module):
    def __init__(self, input_dim, hidden, classes):
        super(dgl_sgc, self).__init__()
        self.layer1 = SGConv(in_feats=input_dim, out_feats=hidden, cached=False, allow_zero_in_degree=True)  # k=1
        self.layer2 = SGConv(in_feats=hidden, out_feats=classes, cached=False, allow_zero_in_degree=True)

    def forward(self, g, features):
        x = self.layer1(g, features)
        x = F.elu(x)
        x = self.layer2(g, x)
        return x

def dgl_to_scipy_adj(g):
    src, dst = g.edges()
    src = src.detach().cpu().numpy()
    dst = dst.detach().cpu().numpy()

    n = g.num_nodes()
    data = np.ones(len(src), dtype=np.float32)
    adj = sp.csr_matrix((data, (src, dst)), shape=(n, n), dtype=np.float32)

    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()

    return adj


def dgl_to_torch_sparse_adj(g, device):
    src, dst = g.edges()
    idx = torch.stack([src.to(device), dst.to(device)], dim=0).long()
    val = torch.ones(idx.size(1), dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(
        idx,
        val,
        (int(g.num_nodes()), int(g.num_nodes())),
        device=device,
    ).coalesce()
    return adj

def tensor_to_numpy(x, dtype=None):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if dtype is not None:
        x = x.astype(dtype)
    return x


def features_to_scipy(x):
    x_np = tensor_to_numpy(x, np.float32)
    return sp.csr_matrix(x_np)


def indices_to_numpy(idx):
    return tensor_to_numpy(idx, np.int64)


class GNNModelWrapper:
    def __init__(self, model_name, in_dim, n_classes, lr, wd, args, device):
        self.device = device
        self.args = args
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.lr, self.wd = lr, wd

        a = args
        self.model_name = model_name.lower()
        self.model = None
        self.fit_num_nodes = None
        # Optional black-box query accounting.  Training calls the model
        # directly and is intentionally excluded; predict/predict_proba calls
        # are counted only when a stage name is explicitly enabled.
        self.query_counts = {}
        self.query_stage = None
        self.attack_output_mode = str(getattr(args, "attack_output_mode", "full_probability"))
        self.attack_output_bits = int(getattr(args, "attack_output_bits", 8))
        self.attack_output_topk = int(getattr(args, "attack_output_topk", 2))
        if model_name=='gat':
            self.model = dgl_gat(
                input_dim=in_dim, out_dim=a.hidden,
                num_heads=[a.num_heads1, a.num_heads2], num_classes=n_classes,
                dropout=[a.feat_dropout, a.attn_dropout]
            ).to(device)
        if model_name=='gcn':
            self.model = dgl_gcn(in_dim, nhidden=a.hidden, nclasses=n_classes).to(device)
        if model_name.lower() == 'sage':
            self.model = dgl_sage(input_dim=in_dim, nhidden=a.hidden, aggregator_type=a.sage_agg_type, nclasses=n_classes).to(device)
        if model_name.lower() == 'appnp':
            self.model = dgl_appnp(input_dim=in_dim, hidden=a.appnp_hidden, classes=n_classes, k=a.K, alpha=a.alpha).to(device)
        if model_name.lower() == 'gin':
            self.model = dgl_gin(input_dim=in_dim, hidden=a.gin_hidden, classes=n_classes, aggregator_type=a.gin_agg_type).to(device)
        if model_name.lower() == 'sgc':
            self.model = dgl_sgc(input_dim=in_dim, hidden=a.hidden, classes=n_classes).to(device)
        if model_name.lower() == 'simpgcn':
            self.model = dgl_simpgcn(
                input_dim=in_dim,
                hidden=a.hidden,
                classes=n_classes,
                k=getattr(args, "simpgcn_k", 20),
                gamma=getattr(args, "simpgcn_gamma", 0.1),
                lambda_self=getattr(args, "simpgcn_lambda_self", 1e-3),
                m_pairs=getattr(args, "simpgcn_m_pairs", 5),
                input_dropout=getattr(args, "simpgcn_input_dropout", 0.0),
                dropout=getattr(args, "dropout", 0.5),
                bn=getattr(args, "simpgcn_bn", False),
            ).to(device)
        if model_name.lower() == 'gcn_jaccard':
            from defense import GCNJaccard

            self.jaccard_threshold = getattr(args, "jaccard_threshold", 0.01)
            self.model = GCNJaccard(
                nfeat=in_dim,
                nhid=args.hidden,
                nclass=n_classes,
                dropout=getattr(args, "dropout", 0.5),
                lr=lr,
                weight_decay=wd,
                device=device
            ).to(device)

    #     if model_name.lower() == 'jaccard_gcn':
    #         self.model = dgl_jaccard_gcn(input_dim=in_dim, hidden=a.hidden, classes=n_classes).to(device)
    #     if model_name.lower() == 'hang':
    #         self.model = dgl_hang(input_dim=in_dim,
    #             hidden=a.hidden,              # = --hidden_dim
    #             classes=n_classes,
    #             time=3.0,                  # = --time
    #             step_size=1.0,        # = --step_size
    #             add_source=True,      # = --add_source
    #             input_dropout=0.4,# = --input_dropout
    #             dropout=0.0,            # = --dropout
    #             batch_norm=True,      # = --batch_norm
    #             energy_hidden=None,
    #             activation='tanh',            # 贴近 vanilla HANG
    #             p_init='copy',                # p0 = q0（源码常用）
    #             share_block=True              # 等价于 --block constant（参数跨步共享）
    # ).to(device)
    #     if model_name.lower() == 'evennet':
    #         self.model = dgl_evennet(
    #             input_dim=in_dim,
    #             hidden=a.hidden,  # --hidden
    #             classes=n_classes,
    #             K=10,  # --K  (默认 10)
    #             input_dropout=0.4,
    #             dropout=0.0,
    #             bn=False,
    #             mlp_layers=1
    #         ).to(device)
        self.lr, self.wd = lr, wd
        if self.model is None:
            raise RuntimeError("Victim.model was not constructed — check model_name and branches.")

    def _deeprobust_predict(self, G=None, feats=None):
        if self.model is None:
            raise RuntimeError(f"{self.model_name} has not been fitted yet.")
        if G is None or feats is None:
            return self.model.predict()
        if self.fit_num_nodes is not None and int(G.num_nodes()) != int(self.fit_num_nodes):
            raise RuntimeError(
                f"DeepRobust {self.model_name} was fitted with {self.fit_num_nodes} nodes, "
                f"but predict/logits got a graph with {G.num_nodes()} nodes. "
                "It cannot do DGL-style inductive forward on newly injected nodes; "
                "retrain it on the graph with the same number of nodes first."
            )
        return self.model.predict(features_to_scipy(feats), dgl_to_scipy_adj(G))
    # @torch.no_grad()
    # def predict(self, g, x):
    #     self.model.eval()
    #     return self.model(g, x)

    def fit_eval(self, g, x, y, tr, va, te, epochs=200, patience=50, verbose=False,
                 fixed_epochs=False):
        if self.model_name == 'gcn_jaccard':
            adj = dgl_to_scipy_adj(g)

            features = tensor_to_numpy(x, np.float32)
            labels = tensor_to_numpy(y, np.int64)

            idx_train = indices_to_numpy(tr)
            idx_val = indices_to_numpy(va)
            idx_test = indices_to_numpy(te)

            self.model.fit(
                features,
                adj,
                labels,
                idx_train,
                idx_val,
                threshold=self.jaccard_threshold,
                train_iters=epochs,
                verbose=verbose
            )

            self.model.eval()
            output = self.model.predict()  # log probabilities

            labels_t = torch.LongTensor(labels).to(output.device)
            idx_test_t = torch.LongTensor(idx_test).to(output.device)

            te_acc = accuracy(output[idx_test_t], labels_t[idx_test_t])
            return {'test_acc': float(te_acc), 'epochs_trained': int(epochs),
                    'fixed_epochs': bool(fixed_epochs)}

        needs_grad_forward = getattr(self.model, "requires_grad_forward", False)
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        best, best_state, wait = 0.0, None, 0

        for ep in range(1, epochs + 1):
            self.model.train()
            logits = self.model(g, x)
            loss = F.cross_entropy(logits[tr], y[tr])

            if self.model_name == 'simpgcn' and hasattr(self.model, "last_self_loss"):
                loss = loss + self.model.lambda_self * self.model.last_self_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            if needs_grad_forward:
                with torch.enable_grad():
                    logits = self.model(g, x)
            else:
                with torch.no_grad():
                    logits = self.model(g, x)

            va_acc = accuracy(logits[va], y[va])
            te_acc = accuracy(logits[te], y[te])

            if verbose and ep % 20 == 0:
                print(f"  [Victim] ep {ep:3d}/{epochs} | val {va_acc:.4f} | best test {best*100:.2f}%")

            if va_acc > best:
                best, best_state, wait = va_acc, {k: v.clone() for k, v in self.model.state_dict().items()}, 0
            else:
                wait += 1
                if not fixed_epochs and wait >= patience:
                    break

        # Normal experiments retain the validation-selected checkpoint.  The
        # duration-stability experiment deliberately evaluates the state after
        # exactly N optimizer updates, so restoring an earlier checkpoint would
        # erase the intended 100/200/300-epoch intervention.
        epochs_trained = int(ep)
        if best_state and not fixed_epochs:
            self.model.load_state_dict(best_state)

        with torch.no_grad():
            self.model.eval()
            logits = self.model(g, x)
            te_acc = accuracy(logits[te], y[te])

        return {'test_acc': float(te_acc), 'epochs_trained': epochs_trained,
                'fixed_epochs': bool(fixed_epochs)}

    @torch.no_grad()
    def logits(self, G, feats):
        if self.model_name == 'gcn_jaccard':
             return self.model.predict()
        """
        
        统一拿分类 logits（不做 softmax）。
        绝大多数 DGL 模型 forward(G, feats)；如果你的模型是其它签名，改这里即可。
        """
        self.model.eval()
       
        return self.model(G, feats)

    @torch.no_grad()
    def embeddings(self, G=None, feats=None, nodes=None):
        """
        返回节点嵌入。
        - SimPGCN: 使用本地 DGL SimPGCN 最近一次 forward 的第一层隐藏表示。
        - 其它 DGL 模型: 若没有专门嵌入接口，则回退为 logits。
        """
        if self.model_name == 'simpgcn':
            self.model.eval()
            _ = self.model(G, feats)
            emb = getattr(self.model, "last_embedding", None)
            if emb is None:
                emb = self.model(G, feats)
            if nodes is not None:
                emb = emb[nodes]
            return emb

        self.model.eval()
        try:
            out = self.model(G, feats, return_embedding=True)
            emb = out[1] if isinstance(out, tuple) else out
        except TypeError:
            emb = self.model(G, feats)
        if nodes is not None:
            emb = emb[nodes]
        return emb

    embedding = embeddings

    @torch.no_grad()
    def predict(self, G, feats, nodes=None):
        """
        返回预测标签（argmax）。
        nodes: Tensor[list[int]]，可选；若给定，只取这些节点的预测。
        """
        self._record_query(G, nodes)
        lg = self.logits(G, feats)
        if nodes is not None:
            lg = lg[nodes]
        return lg.argmax(dim=-1)

    @torch.no_grad()
    def predict_proba(self, G, feats, nodes=None, cls: int = None, reduction: str = "mean"):
        """
        返回置信度（概率）：
          - 如果 cls=None：返回各节点的 *最大类* 概率（模型信心）
          - 如果 cls=k  ：返回第 k 类的概率（可用于 targeted）
        nodes: 仅对这些节点取值（与 scorer 调用保持一致）
        reduction: "none" | "mean"。scorer 里用单节点，返回标量即可。
        """
        self._record_query(G, nodes)
        lg = self.logits(G, feats)
        if nodes is not None:
            lg = lg[nodes]
        # prob = F.softmax(lg, dim=-1)
        if self.model_name == 'gcn_jaccard':
            prob = torch.exp(lg)
        else:
            prob = F.softmax(lg, dim=-1)
        prob = self._restrict_attack_distribution(prob)
        if cls is None:
            conf, _ = prob.max(dim=-1)  # 每个节点的最大类概率
        else:
            conf = prob[:, cls]

        if reduction == "none":
            return conf
        if conf.numel() == 1:
            return conf.squeeze()
        return conf.mean()

    @torch.no_grad()
    def predict_distribution(self, G, feats, nodes=None):
        """Return the complete class distribution for explicitly queried nodes."""
        self._record_query(G, nodes)
        lg = self.logits(G, feats)
        if nodes is not None:
            lg = lg[nodes]
        prob = torch.exp(lg) if self.model_name == 'gcn_jaccard' else F.softmax(lg, dim=-1)
        return self._restrict_attack_distribution(prob)

    def _restrict_attack_distribution(self, prob):
        """Apply the configured score-based/decision-based attack interface."""
        mode = self.attack_output_mode
        if mode == "full_probability":
            return prob
        if mode == "quantized_probability":
            levels = max(1, (1 << max(1, self.attack_output_bits)) - 1)
            restricted = torch.round(prob * levels) / levels
            return restricted / restricted.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if mode == "topk_probability":
            count = max(1, min(self.attack_output_topk, int(prob.shape[-1])))
            values, indices = torch.topk(prob, k=count, dim=-1)
            restricted = torch.zeros_like(prob).scatter(-1, indices, values)
            return restricted / restricted.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if mode == "hard_label":
            labels = prob.argmax(dim=-1, keepdim=True)
            return torch.zeros_like(prob).scatter(-1, labels, 1.0)
        raise ValueError(f"Unknown attack_output_mode: {mode}")

    def _record_query(self, G, nodes=None):
        stage = self.query_stage
        if not stage:
            return
        entry = self.query_counts.setdefault(stage, {"model_calls": 0, "node_predictions": 0})
        entry["model_calls"] += 1
        if nodes is None:
            entry["node_predictions"] += int(G.num_nodes())
        elif torch.is_tensor(nodes):
            entry["node_predictions"] += int(nodes.numel())
        else:
            entry["node_predictions"] += len(nodes)

# Historical imports use this name for the same wrapper.
Victim = GNNModelWrapper
