# subgraph_sparse_attack.py
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import dgl
import torch
import torch.nn.functional as F
import time

@torch.no_grad()
def _remove_edges_uv(G, src_list, dst_list, undirected=True):
    if len(src_list) == 0: return G
    eids = []
    for u, v in zip(src_list, dst_list):
        try: eids.append(G.edge_ids(u, v, return_uv=False))
        except: pass
        if undirected:
            try: eids.append(G.edge_ids(v, u, return_uv=False))
            except: pass
    if len(eids) == 0: return G
    eids = torch.cat([e if torch.is_tensor(e) else torch.tensor([e], device=G.device) for e in eids])
    G2 = G.clone()
    G2.remove_edges(eids)
    return G2

# @torch.no_grad()
# def _edge_score_blackbox(victim, G, X, anchor, y_tgt, inj, v, mode="add"):
#     """
#     评分越大越好：
#     - mode='add': 加上 (inj->v) 后，目标损失下降多少（base_loss - new_loss）
#     - mode='rem': 删掉 (inj->v) 后，目标损失上升多少（new_loss - base_loss）
#     """
#     dev = G.device
#     # 轻量 margin-loss（和上面一致）
#     def loss_on(Gx):
#         out = victim.predict(Gx, X)
#         print("out是")
#         if out.dim()==2:
#             logit = out[anchor]
#             y_true = int(torch.argmax(out[anchor]).item())
#             margin = logit[y_tgt] - logit[y_true]
#             return (-margin).relu()
#         else:
#             pred = int(out[anchor].item())
#             return torch.tensor(0.0 if pred == int(y_tgt) else 1.0, device=dev)

#     base = loss_on(G)
#     if mode == "add":
#         if _has_uv(G, inj, v, undirected=True):   # 已有边则贡献≈0
#             return torch.tensor(0.0, device=dev)
#         Gp = G.clone(); Gp.add_edges(torch.tensor([inj],device=dev), torch.tensor([v],device=dev))
#         return (base - loss_on(Gp))
#     else:
#         if not _has_uv(G, inj, v, undirected=True):
#             return torch.tensor(0.0, device=dev)
#         Gm = _remove_edges_uv(G, [inj],[v], undirected=True)
#         return (loss_on(Gm) - base)

@torch.no_grad()
def _score_pseudo_label_probability_drop(surrogate, G, X, anchor, y_ref, inj, v, mode="add"):
    dev = G.device

    def ref_conf(Gx):
        return surrogate.predict_proba(
            Gx,
            X,
            nodes=torch.tensor([int(anchor)], device=dev),
            cls=int(y_ref),
            reduction="none"
        ).squeeze()

    base = ref_conf(G)

    if mode == "add":
        if _has_uv(G, inj, v, undirected=True):
            return torch.tensor(0.0, device=dev)
        G2 = G.clone()
        G2.add_edges(
            torch.tensor([int(inj)], device=dev),
            torch.tensor([int(v)], device=dev)
        )
        new = ref_conf(G2)
        return base - new

    elif mode == "rem":
        if not _has_uv(G, inj, v, undirected=True):
            return torch.tensor(0.0, device=dev)
        G2 = _remove_edges_uv(G, [int(inj)], [int(v)], undirected=True)
        new = ref_conf(G2)
        return base - new

    else:
        raise ValueError(f"Unknown mode: {mode}")

@torch.no_grad()
def _has_uv(G: dgl.DGLGraph, u: int, v: int, undirected: bool = False) -> bool:
    try:
        if not undirected:
            h = G.has_edges_between(u, v)
            return bool(h.item()) if torch.is_tensor(h) and h.numel() == 1 else bool(h)
        else:
            h1 = G.has_edges_between(u, v)
            h2 = G.has_edges_between(v, u)
            def to_bool(x): return bool(x.item()) if torch.is_tensor(x) and x.numel() == 1 else bool(x)
            return to_bool(h1) or to_bool(h2)
    except Exception:
        try:
            if not undirected:
                return bool((G.successors(u) == v).any().item())
            else:
                return bool((G.successors(u) == v).any().item() or (G.successors(v) == u).any().item())
        except Exception:
            return False

@torch.no_grad()
def _has_uv_batch(G: dgl.DGLGraph, u_scalar: int, v_list: List[int]) -> torch.Tensor:
    """
    批量判断 u_scalar -> v_i 是否已有边
    返回: [K] bool tensor (在 G.device 上)
    """
    dev = G.device
    if len(v_list) == 0:
        return torch.zeros(0, dtype=torch.bool, device=dev)
    v = torch.tensor(v_list, dtype=torch.long, device=dev)
    u = torch.full_like(v, fill_value=int(u_scalar))
    # dgl>=0.8 支持批量 has_edges_between
    h = G.has_edges_between(u, v)
    if torch.is_tensor(h):
        return h.to(device=dev)
    # 兜底（一般不会走到这）
    return torch.zeros_like(v, dtype=torch.bool, device=dev)

def _build_graph_with_edges(G: dgl.DGLGraph, add_src, add_dst):
    """在 GPU 上就地克隆 + add_edges（若 add_src 为空直接返回原图）"""
    if len(add_src) == 0:
        return G
    G2 = G.clone()
    G2.add_edges(torch.tensor(add_src, device=G.device),
                 torch.tensor(add_dst, device=G.device))
    return G2

def _degree_kl_loss(G_before: dgl.DGLGraph, G_after: dgl.DGLGraph) -> torch.Tensor:
    """全图入/出度分布 KL（softmax 归一）——默认关掉可省大量时间"""
    dev = G_before.device
    out_b = G_before.out_degrees().float().to(dev); in_b  = G_before.in_degrees().float().to(dev)
    out_a = G_after.out_degrees().float().to(dev);  in_a  = G_after.in_degrees().float().to(dev)
    kl = torch.nn.KLDivLoss(reduction="batchmean")
    p_out_b = F.softmax(out_b, dim=0); p_out_a = F.softmax(out_a, dim=0)
    p_in_b  = F.softmax(in_b,  dim=0); p_in_a  = F.softmax(in_a,  dim=0)
    return kl(p_out_b.log(), p_out_a) + kl(p_in_b.log(), p_in_a)

def _cos_sim(a: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a.view(1, -1), dim=-1)
    B = F.normalize(B, dim=-1)
    return (a * B).sum(dim=-1)

# ========== 单 anchor 损失 ==========
@torch.no_grad()
def _feature_refinement_loss(
    surrogate,
    G_base: dgl.DGLGraph,
    feats_base: torch.Tensor,
    anchor_id: int,
    anchor_pseudo_label: int,
    inj_id: int,
    cand_ids: List[int],
    # edge_mask_bin: torch.Tensor,        # [K] 0/1
    dx_inj: torch.Tensor,               # [d] (-1,1) 已限幅
    feature_l1_weight: float,
    T_homo: float,
    lam_homo: float,
    lam_degkl: float,
    class_proto: Optional[torch.Tensor] = None,
    prototype_alignment_weight: float = 0.0,
    lam_edgecls: float = 0.0,
    pseudo_labels: Optional[torch.Tensor] = None,
    use_amp: bool = False,
) -> torch.Tensor:
    dev = feats_base.device
    x0 = feats_base[inj_id]
    x_inj = (x0 + dx_inj).clamp(0, 1)

    # # 选中需要新增的边
    # add_idx = (edge_mask_bin > 0.5).nonzero(as_tuple=False).view(-1)
    # add_src = [inj_id] * int(add_idx.numel())
    # add_dst = [int(cand_ids[i]) for i in add_idx.tolist()]

    # # 构造 perturbed graph + feats
    # G_att = _build_graph_with_edges(G_base, add_src, add_dst)
    # feats_att = feats_base.clone()
    # feats_att[inj_id] = x_inj

    # # 只需要函数值（NES），不反传 victim，因此 no_grad + 可选 AMP
    # if use_amp and dev.type == 'cuda':
    #     from torch.cuda.amp import autocast
    #     with autocast(dtype=torch.float16):
    #         out = victim.predict(G_att, feats_att)
    # else:
    #     out = victim.predict(G_att, feats_att)
    # # print(out)
    # # 分类项：把 anchor 推向目标类
    # if out.dim() == 2 and out.dtype.is_floating_point:
    #     logit = out[anchor_id]  # [C]
    #     y_true = int(torch.argmax(out[anchor_id]).item())  # 没有GT时就用当前预测作“真类”
    #     tgt = int(y_target)
    #     margin = logit[tgt] - logit[y_true]  # 目标-真类
    #     ce = (-margin).unsqueeze(0).relu().mean()  # hinge 风格；也可 -margin 直接用
    # elif out.dim() == 1:
    #     pred = int(out[anchor_id].item())
    #     ce = torch.tensor(0.0 if pred == int(y_target) else 1.0, device=dev)
    # else:
    #     raise ValueError(f"predict() output not supported: shape={tuple(out.shape)}, dtype={out.dtype}")

    # 稀疏正则（边数量 + 特征扰动）
    l1 = feature_l1_weight * (dx_inj.abs().sum())

    # # 同质性（inj 与 cand 的余弦相似度 ≥ T_homo）
    # homo = torch.tensor(0.0, device=dev)
    # if len(add_dst) > 0 and lam_homo > 0:
    #     cosv = _cos_sim(x_inj, feats_att[torch.tensor(add_dst, device=dev)])
    #     homo = lam_homo * F.relu(T_homo - cosv).mean()

    # 注入特征对齐目标类原型
    class_align = torch.tensor(0.0, device=dev)
    if (class_proto is not None) and prototype_alignment_weight > 0:
        class_align = prototype_alignment_weight * (1.0 - _cos_sim(x_inj, class_proto.view(1, -1)).squeeze())

    # # 边的类别对齐（优先连向目标类 cand）
    # edge_cls = torch.tensor(0.0, device=dev)
    # if labels_for_sampler is not None and lam_edgecls > 0 and len(cand_ids) > 0:
    #     lab = labels_for_sampler[torch.tensor(cand_ids, device=dev)]
    #     mask_t = (lab == int(y_target)).float()
    #     p_in = edge_mask_bin.float()
    #     eps = 1e-8
    #     pos = (p_in * mask_t).sum() / (mask_t.sum() + eps)
    #     neg = (p_in * (1.0 - mask_t)).sum() / ((1.0 - mask_t).sum() + eps)
    #     edge_cls = lam_edgecls * (-pos + neg)

    # # 度分布 KL（默认关闭）
    # degkl = torch.tensor(0.0, device=dev)
    # if lam_degkl > 0:
    #     degkl = lam_degkl * _degree_kl_loss(G_base, G_att)
    # print(
    #     f"ce:{ce.detach().cpu().item():.6f}, "
    #     f"l1:{l1.detach().cpu().item():.6f}, "
    #     f"class_align:{class_align.detach().cpu().item():.6f}"
    # )
    # return ce + l1 + homo + class_align + edge_cls + degkl
    # return ce + l1 + class_align
    return l1+class_align
    # return ce
def _adam_step(theta: torch.Tensor, grad_est: torch.Tensor, adam_state: dict,
                lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
    t = adam_state.get("t", 0) + 1
    m = adam_state.get("m", torch.zeros_like(theta))
    v = adam_state.get("v", torch.zeros_like(theta))
    m = b1 * m + (1 - b1) * grad_est
    v = b2 * v + (1 - b2) * (grad_est * grad_est)
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    theta.data -= lr * m_hat / (v_hat.sqrt() + eps)
    adam_state["t"], adam_state["m"], adam_state["v"] = t, m, v

# ========== 主攻击 ==========
def run_sparse_refinement(
    G: dgl.DGLGraph,
    feats: torch.Tensor,
    surrogate,                                    # 需有 predict(G, feats) -> logits
    merged_by_anchor: Dict[int, Dict[str, List[int]]],
    pseudo_labels: torch.Tensor,
    class_protos: Optional[torch.Tensor],
    args,
    T_homo: float = 0.7,
    log_every:int = 20
):
    dev = feats.device
    G_work = G
    X_work = feats

    # === 读参数 ===
    nes_steps     = getattr(args, "nes_steps", 50)
    sigma     = getattr(args, "sigma", 0.25)
    queries   = getattr(args, "nes_directions", 32)
    lr        = getattr(args, "adam_lr", 0.05)
    feature_l1_weight    = getattr(args, "feature_l1_weight", 1e-3)
    lam_homo  = getattr(args, "lam_in", 0.1)
    lam_degkl = getattr(args, "degkl_w", 0.0)
    prototype_alignment_weight = getattr(args, "prototype_alignment_weight", 0.5)
    lam_edgec = getattr(args, "lam_edgecls", 0.5)

    # 新增：边预算与二值化策略
    # hard_topk = bool(getattr(args, "nes_hard_topk", False))
    # tau_e     = float(getattr(args, "nes_edge_tau", 1.0))
    # thr_e     = float(getattr(args, "nes_edge_thr", 0.5))
    edge_budget_default = getattr(args, "sparse_action_budget", None)
    optimize_features = bool(getattr(args, "optimize_nes_features", True))
    structure_mode = str(getattr(args, "structure_mode", "independent_ranking"))
    structure_mode = {"joint_static": "independent_ranking"}.get(structure_mode, structure_mode)
    stats = {"structure_added_edges": 0, "structure_removed_edges": 0,
             "structure_actions": [], "structure_candidate_actions": 0,
             "structure_selected_actions": 0, "structure_selection_seconds": 0.0}
    trace_path = str(getattr(args, "structure_trace_path", "") or "")
    trace = None
    if trace_path:
        trace = {
            "structure_mode": structure_mode,
            "scoring_function": "_edge_score_blackbox: anchor pseudo-label probability drop",
            "records": [],
        }
    # print(edge_budget_default,hard_topk)
    for a_idx, (anchor, pack) in enumerate(merged_by_anchor.items()):
        t_anchor0 = time.time()
        inj_id = pack.get("inj_id", None)
        cands: List[int] = pack.get("cands", [])
        if (inj_id is None) or (len(cands) == 0):
            continue
        # print(anchor)
        anchor_pseudo_label = int(pseudo_labels[int(anchor)].item())
        x0 = X_work[inj_id].detach()
        d  = int(x0.numel())
        K  = len(cands)

        # 每个锚点的边预算：默认取 --sparse_action_budget；未设置则用「Stage-3 同配比」的近似
        if edge_budget_default is None:
            # 若你在 main 里有 deg_per_node / intra_ratio，可通过 args 传进来
            deg_per_node = int(getattr(args, "deg_per_node_landing", 6))  # 没有就给个保守默认
            intra_ratio  = float(getattr(args, "intra_ratio_landing", 0.3))
            k_budget = max(0, min(K, int(round(deg_per_node * intra_ratio))))
        else:
            k_budget = max(0, min(K, int(edge_budget_default)))

        # theta_e = torch.zeros(K, device=dev)
        theta_x = torch.zeros(d, device=dev)
        # adam_e, adam_x = {}, {}
        adam_x = {}
        # 先验：已有边给正初值（利于少量“强”边获选）
        # with torch.no_grad():
        #     exist = torch.tensor([1.0 if _has_uv(G_work, inj_id, int(v)) else 0.0 for v in cands], device=dev)
        #     theta_e.data = torch.where(exist > 0.5, torch.tensor(2.0, device=dev), theta_e)

        for it in range(nes_steps if optimize_features else 0):
            # dir_e = torch.randn(queries, K, device=dev)
            dir_x = torch.randn(queries, d, device=dev)
            losses = []
            epoch=0
            for sgn in (+1.0, -1.0):
                for i in range(queries):
                    # ======= 边二值化：hard top-k 或「阈值+cap」 =======
                    # logit = (theta_e + sgn * sigma * dir_e[i]) / max(tau_e, 1e-6)

                    # if hard_topk:
                    #     if k_budget > 0:
                    #         idx = torch.topk(logit, k_budget).indices
                    #         e_bin = torch.zeros(K, device=dev)
                    #         e_bin[idx] = 1.0
                    #     else:
                    #         e_bin = torch.zeros(K, device=dev)
                    # else:
                    #     e_prob = torch.sigmoid(logit)
                    #     e_bin  = (e_prob > thr_e).float()
                    #     # cap 到预算（防止一次开太多）
                    #     nnz = int(e_bin.sum().item())
                    #     if nnz > k_budget:
                    #         keep = torch.topk(e_prob, k_budget).indices
                    #         e_bin.zero_(); e_bin[keep] = 1.0

                    # ======= 特征扰动 =======
                    dx = torch.tanh(theta_x + sgn * sigma * dir_x[i])  # (-1,1)

                    proto_t = class_protos[anchor_pseudo_label] if (class_protos is not None) else None
                    loss = _feature_refinement_loss(
                        surrogate=surrogate, G_base=G_work, feats_base=X_work,
                        anchor_id=int(anchor), anchor_pseudo_label=anchor_pseudo_label,
                        inj_id=int(inj_id), cand_ids=cands,
                        # edge_mask_bin=e_bin, 
                        dx_inj=dx,
                        feature_l1_weight=feature_l1_weight, T_homo=T_homo, lam_homo=lam_homo, lam_degkl=lam_degkl,
                        class_proto=proto_t, prototype_alignment_weight=prototype_alignment_weight,
                        lam_edgecls=lam_edgec, pseudo_labels=pseudo_labels
                    )
                    losses.append(loss)

            losses = torch.stack(losses)           # [2q]
            f_plus, f_minus = losses[:queries], losses[queries:]
            coef = (f_plus - f_minus).view(queries, 1)
            # g_e = (coef * dir_e).mean(dim=0) / (sigma + 1e-8)
            g_x = (coef * dir_x).mean(dim=0) / (sigma + 1e-8)

            # _nes_update(theta_e, g_e, adam_e, lr)
            _adam_step(theta_x, g_x, adam_x, lr)

            # if (it % log_every == 0):
            #     with torch.no_grad():
            #         e_cur = (torch.sigmoid(theta_e) > 0.5).float()
            #         add_idx = (e_cur > 0.5).nonzero(as_tuple=False).view(-1)
            #         add_src = [int(inj_id)] * int(add_idx.numel())
            #         add_dst = [int(cands[i]) for i in add_idx.tolist()]
            #         G_tmp = _build_graph_with_edges(G_work, add_src, add_dst)
            #         X_tmp = X_work.clone()
            #         X_tmp[int(inj_id)] = (X_tmp[int(inj_id)] + torch.tanh(theta_x)).clamp(0, 1)
            #         out = victim.predict(G_tmp, X_tmp)
            #         hit = (out[int(anchor)].argmax().item() == y_tgt) if out.dim() == 2 else (
            #                     int(out[int(anchor)].item()) == y_tgt)
                    # ✅ 正确显示锚点序号/ID、已用时
                    # print(f"[NES GPU] anchor_idx={a_idx:03d} anchor_id={int(anchor)} "
                    #       f"it={it:04d} | edges={int(e_cur.sum().item())} | hit={hit} "
                    #       f"| t={time.time() - t_anchor0:.1f}s")
            epoch+=1
        # ======= 落地：严格 top-k 写回（保证边数不超预算） =======
        with torch.no_grad():
            if hasattr(surrogate, 'query_stage'):
                surrogate.query_stage = 'structure'
            # 1) 拿到“候选新增”按概率排序，裁到一个上限 L（降低黑盒评估开销）
            # e_prob = torch.sigmoid(theta_e / max(tau_e, 1e-6))
            # # print(e_prob.shape)
            # L = int(min(16, K))  # 每个锚点评估 <=16 个新增候选
            # cand_idx = torch.topk(e_prob, L).indices.tolist()
            # L = int(min(16, K))
            query_fraction = float(getattr(args, "query_budget_fraction", 1.0))
            L = min(K, max(k_budget, int(round(K * query_fraction))))
            rand_score = torch.rand(K, device=dev)
            cand_idx = torch.topk(rand_score, L).indices.tolist()
            cand_add = [int(cands[i]) for i in cand_idx
                        if (int(inj_id) != int(cands[i])) and (not _has_uv(G_work, int(inj_id), int(cands[i]), True))]

            # 2) 当前 inj 的“恶性↔恶性”已存在边（如果你的恶性集合已知）
            inj_set = {int(p["inj_id"]) for p in merged_by_anchor.values() if "inj_id" in p}
            exist_m = [int(v) for v in G_work.successors(int(inj_id)).tolist() if v in inj_set and v != int(inj_id)]
            if query_fraction < 1.0 and exist_m:
                keep_existing = max(1, int(round(len(exist_m) * query_fraction)))
                exist_m = exist_m[:keep_existing]

            if structure_mode == "none":
                cand_add, exist_m = [], []
            elif structure_mode == "add_only":
                exist_m = []
            elif structure_mode == "remove_only":
                cand_add = []

            actions = [("add", int(v)) for v in cand_add] + [("rem", int(v)) for v in exist_m]
            stats["structure_candidate_actions"] += len(actions)
            selection_started = time.perf_counter()
            trace_record = None
            if trace is not None:
                trace_record = {
                    "anchor": int(anchor),
                    "injected_node": int(inj_id),
                    "action_budget": int(k_budget),
                    "candidate_actions": [f"{kind}:{int(inj_id)}:{node}" for kind, node in actions],
                    "candidate_count": len(actions),
                    "selected_actions": [],
                    "nes_steps": [],
                }

            if structure_mode in {"sequential", "beam"}:
                def apply_action(graph, action):
                    kind, node = action
                    if kind == "rem":
                        if not _has_uv(graph, int(inj_id), node, True):
                            return None
                        return _remove_edges_uv(graph, [int(inj_id)], [node], undirected=True)
                    if _has_uv(graph, int(inj_id), node, True):
                        return None
                    degree_limit = int(getattr(args, "deg_per_node_landing", 0))
                    if degree_limit > 0:
                        neighbours = (set(map(int, graph.successors(int(inj_id)).tolist())) |
                                      set(map(int, graph.predecessors(int(inj_id)).tolist()))) - {int(inj_id)}
                        if len(neighbours) >= degree_limit:
                            return None
                    return _build_graph_with_edges(graph, [int(inj_id)], [node])

                accepted = []
                if structure_mode == "sequential":
                    remaining = list(actions)
                    initial_scores = {}
                    initial_ranks = {}
                    for _ in range(k_budget):
                        scored_now = []
                        for action in remaining:
                            value = _score_pseudo_label_probability_drop(
                                surrogate, G_work, X_work, int(anchor), anchor_pseudo_label,
                                int(inj_id), action[1], mode=action[0]
                            )
                            scored_now.append((float(value.item()), action))
                        if not scored_now:
                            break
                        if not initial_scores:
                            initial_scores = {action: score for score, action in scored_now}
                            initial_ranks = {
                                action: 1 + sum(other_score > score for other_score, _ in scored_now)
                                for score, action in scored_now
                            }
                        _, chosen = max(scored_now, key=lambda item: item[0])
                        chosen_score = next(score for score, action in scored_now if action == chosen)
                        conditional_rank = 1 + sum(score > chosen_score for score, _ in scored_now)
                        if trace_record is not None:
                            trace_record["nes_steps"].append({
                                "step": len(trace_record["nes_steps"]) + 1,
                                "selected_action": f"{chosen[0]}:{int(inj_id)}:{chosen[1]}",
                                "conditional_score": float(chosen_score),
                                "conditional_rank": int(conditional_rank),
                                "independent_score": float(initial_scores.get(chosen, chosen_score)),
                                "independent_rank": int(initial_ranks.get(chosen, conditional_rank)),
                                "scores_at_current_graph": [
                                    {
                                        "action": f"{action[0]}:{int(inj_id)}:{action[1]}",
                                        "score": float(score),
                                        "conditional_rank": int(1 + sum(other_score > score for other_score, _ in scored_now)),
                                        "independent_score": float(initial_scores.get(action, score)),
                                        "independent_rank": int(initial_ranks.get(action, 1)),
                                    }
                                    for score, action in scored_now
                                ],
                            })
                        changed = apply_action(G_work, chosen)
                        remaining.remove(chosen)
                        if changed is None:
                            continue
                        G_work = changed
                        accepted.append(chosen)
                else:
                    # Small-candidate beam search is intentionally confined by
                    # the experiment driver to Cora/CiteSeer.
                    width = int(getattr(args, "beam_width", 3))
                    universe = actions[:int(getattr(args, "beam_candidate_limit", 8))]
                    states = [(G_work, tuple(), float('inf'))]
                    for _ in range(k_budget):
                        expanded = []
                        for graph_state, used, _ in states:
                            for action in universe:
                                if action in used:
                                    continue
                                changed = apply_action(graph_state, action)
                                if changed is None:
                                    continue
                                confidence = surrogate.predict_proba(
                                    changed, X_work,
                                    nodes=torch.tensor([int(anchor)], device=dev),
                                    cls=int(anchor_pseudo_label), reduction="none"
                                ).squeeze()
                                expanded.append((changed, used + (action,), float(confidence.item())))
                        if not expanded:
                            break
                        states = sorted(expanded, key=lambda item: item[2])[:max(1, width)]
                    if states and states[0][1]:
                        G_work, accepted, _ = states[0]
                        accepted = list(accepted)

                for kind, node in accepted:
                    stats["structure_added_edges" if kind == "add" else "structure_removed_edges"] += 1
                    stats["structure_actions"].append(f"{kind}:{int(inj_id)}:{node}")
                stats["structure_selected_actions"] += len(accepted)
                if trace_record is not None:
                    trace_record["selected_actions"] = [
                        f"{kind}:{int(inj_id)}:{node}" for kind, node in accepted
                    ]
                    trace["records"].append(trace_record)
                stats["structure_selection_seconds"] += time.perf_counter() - selection_started
                if optimize_features:
                    dx = torch.tanh(theta_x)
                    X_work = X_work.clone()
                    X_work[int(inj_id)] = (X_work[int(inj_id)] + dx).clamp(0, 1)
                continue

            # 3) 统一打分：新增边用 'add'，已有 m2m 边用 'rem'
            scored = []
            for v in cand_add:
                s = _score_pseudo_label_probability_drop(surrogate, G_work, X_work, int(anchor), anchor_pseudo_label, int(inj_id), v, mode="add")
                scored.append((float(s.item()), "add", v))
                # print("用了add")
                # print(scored)
            for v in exist_m:
                s = _score_pseudo_label_probability_drop(surrogate, G_work, X_work, int(anchor), anchor_pseudo_label, int(inj_id), v, mode="rem")
                scored.append((float(s.item()), "rem", v))
                # print("用了rem")
                # print(scored)
            scored.sort(reverse=True)  # 大到小

            # 4) 只保留“贡献最大的 k_budget 条动作”
            keep_actions = scored[:k_budget]
            actions_before = len(stats["structure_actions"])
            if trace_record is not None:
                rank_by_action = {
                    (kind, node): 1 + sum(other_score > score for other_score, _, _ in scored)
                    for score, kind, node in scored
                }
                trace_record["independent_scores"] = [
                    {
                        "action": f"{kind}:{int(inj_id)}:{node}",
                        "score": float(score),
                        "independent_rank": int(rank_by_action[(kind, node)]),
                        "selected_by_ranking": bool((score, kind, node) in keep_actions),
                    }
                    for score, kind, node in scored
                ]

            # 5) Apply removals first, then admit additions only while the
            # paper's total unique-neighbour degree budget is respected.
            to_add = [v for s, t, v in keep_actions if t == "add"]
            to_rem = [v for s, t, v in keep_actions if t == "rem"]

            if len(to_rem) > 0:
                G_work = _remove_edges_uv(G_work, [int(inj_id)] * len(to_rem), to_rem, undirected=True)
                stats["structure_removed_edges"] += len(to_rem)
                stats["structure_actions"].extend(f"rem:{int(inj_id)}:{int(v)}" for v in to_rem)

            total_degree_budget = int(getattr(args, "deg_per_node_landing", 0))
            if total_degree_budget > 0:
                succ = set(int(v) for v in G_work.successors(int(inj_id)).tolist())
                pred = set(int(v) for v in G_work.predecessors(int(inj_id)).tolist())
                current_degree = len((succ | pred) - {int(inj_id)})
                free_degree = max(0, total_degree_budget - current_degree)
                to_add = to_add[:free_degree]

            if len(to_add) > 0:
                add_src = torch.tensor([int(inj_id)] * len(to_add), device=dev)
                add_dst = torch.tensor(to_add, device=dev)
                # 过滤掉已经存在的（保险）
                mask = ~_has_uv_batch(G_work, int(inj_id), to_add)
                if mask.any():
                    G_work = _build_graph_with_edges(G_work, add_src[mask].tolist(), add_dst[mask].tolist())
                    stats["structure_added_edges"] += int(mask.sum().item())
                    stats["structure_actions"].extend(
                        f"add:{int(inj_id)}:{int(v)}" for v, keep in zip(to_add, mask.tolist()) if keep
                    )

            # 6) 写回特征（仍然夹紧）
            applied_actions = stats["structure_actions"][actions_before:]
            stats["structure_selected_actions"] += len(applied_actions)
            if trace_record is not None:
                trace_record["selected_actions"] = list(applied_actions)
                trace["records"].append(trace_record)
            stats["structure_selection_seconds"] += time.perf_counter() - selection_started
            if optimize_features:
                dx = torch.tanh(theta_x)
                X_work = X_work.clone()
                X_work[int(inj_id)] = (X_work[int(inj_id)] + dx).clamp(0, 1)

    args.attack_stats = stats
    if trace is not None:
        destination = Path(trace_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return G_work, X_work
