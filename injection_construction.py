# Anchor-guided neighbor sampling and injection structure construction.
# 通用增强型 Bernoulli 候选边采样（可替换 Stage-2 的候选池）
# 目标：从“基础池”（如 k-hop 或你已有的 pool）里，按先验融合打分→概率→Bernoulli 抽样，
#      每个锚点精确输出 K 个候选邻居；可选多视图，但为了和 NES 的固定 K 对齐，这里默认输出单视图。

import torch
import torch.nn.functional as F
import dgl
import numpy as np
from typing import List, Tuple, Optional, Dict


@torch.no_grad()
def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)

@torch.no_grad()
def _calibrate_expected_k(p: torch.Tensor, K: int, iters: int = 25) -> torch.Tensor:
    """
    给一组初始概率 p (0-1)，通过缩放系数 scale 的二分，校准期望选择数 sum(clamp(scale*p,0,1)) ≈ K。
    返回 q = clamp(scale*p, 0, 1)
    """
    if p.numel() == 0:
        return p
    lo, hi = 1e-3, 1e3
    for _ in range(iters):
        mid = 0.5*(lo+hi)
        q = (p * mid).clamp_(0, 1)
        s = q.sum().item()
        if s > K:  # 期望过大 -> 缩小 scale
            hi = mid
        else:
            lo = mid
    scale = 0.5*(lo+hi)
    return (p*scale).clamp_(0,1)

@torch.no_grad()
def sample_anchor_neighbors(
    G: dgl.DGLGraph,
    anchors: torch.Tensor,
    base_pool: Dict[int, List[int]],
    K: int,
    labels: Optional[torch.Tensor] = None,
    feats: Optional[torch.Tensor] = None,
    weights: Optional[Dict[str, float]] = None,
    temp: float = 1.0,
    exact_k: bool = True,
    pin_anchor: bool = True,   # <<< 新增：锚点固定进候选
) -> Dict[int, List[int]]:
    """
    改动点：
    - 始终把锚点 u 自身放进池；且若 pin_anchor=True，则最终结果中保证包含 u，
      其余 K-1 个按概率/权重筛选。
    """
    if weights is None:
        weights = dict(homo=1.0, degree=0.3, cos=0.5, khop=0.0, er=0.0)

    # 小工具：余弦相似（若你已有 _cos_sim，可删掉本函数）
    def _cos_sim(x: torch.Tensor, Y: torch.Tensor):
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-12)
        Y = Y / (Y.norm(dim=-1, keepdim=True) + 1e-12)
        return (x * Y).sum(dim=-1)

    device = G.device
    deg = G.out_degrees().float().to(device)
    deg_n = (deg + 1).log() / (deg.max() + 1).log()  # 稳定归一

    out: Dict[int, List[int]] = {}
    for u in anchors.tolist():
        u = int(u)
        # 1) 取基础池，强制把锚点 u 放进去（放在首位以便稳定）
        pool = base_pool.get(u, [])
        pool = [u] + [int(v) for v in pool]  # 先加上u
        # 去重但保序（保持 u 在最前）
        seen = set()
        pool_uni = []
        for v in pool:
            if v not in seen:
                seen.add(v)
                pool_uni.append(v)
        pool = pool_uni

        # 不允许只有 u 且 K>1 时的极端情况：从全图补点
        if len(pool) < max(1, K):
            all_nodes = torch.arange(G.num_nodes(), device=device).tolist()
            all_nodes = [x for x in all_nodes if x != u]
            need = max(1, K) - len(pool)
            pool += all_nodes[:need]

        V = torch.tensor(pool, device=device, dtype=torch.long)

        # 2) 拆分：锚点与其余候选
        anchor_mask = (V == u)
        assert anchor_mask.any(), "pool 中必须包含锚点"
        V_anchor_idx = torch.nonzero(anchor_mask, as_tuple=True)[0][0].item()
        V_oth = V[~anchor_mask]

        # 3) 计算其余候选的融合打分 s（不对锚点本身打分）
        s = torch.zeros(V_oth.shape[0], device=device)

        if labels is not None and 'homo' in weights and weights['homo'] != 0:
            same = (labels[V_oth] == labels[u]).float()
            s = s + weights['homo'] * same

        if 'degree' in weights and weights['degree'] != 0:
            s = s + weights['degree'] * deg_n[V_oth]

        if feats is not None and 'cos' in weights and weights['cos'] != 0:
            sim = _cos_sim(feats[u].unsqueeze(0), feats[V_oth]).clamp_min(0)
            s = s + weights['cos'] * sim

        # 4) 分数 -> 概率
        p = torch.sigmoid(s / max(1e-6, float(temp)))  # (0,1)

        # 5) 期望校准到 K_other（其余名额）
        K_anchor = 1 if (pin_anchor and K > 0) else 0
        K_other = max(K - K_anchor, 0)
        if exact_k:
            q = _calibrate_expected_k(p, K=K_other)
        else:
            q = p

        # 6) 采样其余候选
        if K_other > 0 and V_oth.numel() > 0:
            chosen = (torch.rand_like(q) < q).nonzero(as_tuple=False).view(-1)
            if chosen.numel() < K_other:
                topk = torch.topk(q, k=min(K_other, q.numel()), largest=True).indices
                need = K_other - chosen.numel()
                # 还未被选中的 topk 候选里补齐
                mask_left = torch.ones_like(q, dtype=torch.bool)
                mask_left[chosen] = False
                cands_left = topk[mask_left[topk]]
                add = cands_left[:need]
                chosen = torch.unique(torch.cat([chosen, add]))
            elif chosen.numel() > K_other:
                chosen = chosen[torch.topk(q[chosen], k=K_other, largest=True).indices]
            sel_oth = V_oth[chosen[:K_other]].tolist()
        else:
            sel_oth = []

        # 7) 组装最终候选：锚点 + 其余
        if pin_anchor and K > 0:
            final = [u] + [int(x) for x in sel_oth]
        else:
            # 不固定锚点的情况：回退到“其余”或仅锚点
            final = [int(x) for x in sel_oth] if K_other > 0 else [u]

        # 截断/补齐到恰好 K 个（极罕见边界）
        if len(final) < K:
            # 用 V_oth 中 q 最大的补
            if V_oth.numel() > 0:
                order = torch.topk(q, k=min(K - len(final), q.numel())).indices
                extra = [int(V_oth[i]) for i in order.tolist() if int(V_oth[i]) not in final]
                final += extra[:(K - len(final))]
        elif len(final) > K:
            # 保留：锚点 + 其余里 q 最大的
            if pin_anchor and u in final:
                keep = [u]
                rest = [x for x in final if x != u]
                if len(rest) > 0 and V_oth.numel() > 0:
                    # 计算 rest 在 V_oth 中的 q 排序
                    idx_map = {int(V_oth[i]): i for i in range(V_oth.numel())}
                    rest_sorted = sorted(rest, key=lambda x: q[idx_map.get(x, 0)].item() if x in idx_map else -1, reverse=True)
                    keep += rest_sorted[:(K-1)]
                final = keep
            else:
                final = final[:K]

        out[u] = [int(x) for x in final]

    return out
def _safe_add_node_with_feat(G, feat_row: torch.Tensor) -> int:
    """只为新加的1个节点写入特征，不重写整个 feat 矩阵。"""
    # 保证形状、设备、dtype 一致
    feat_row = feat_row.reshape(1, -1).to(G.device).to(G.ndata['feat'].dtype)
    G.add_nodes(1, data={'feat': feat_row})
    new_id = G.num_nodes() - 1
    # 保险校验
    assert G.ndata['feat'].shape[0] == G.num_nodes(),\
        f"feat rows ({G.ndata['feat'].shape[0]}) != num_nodes ({G.num_nodes()})"
    return new_id
def construct_cross_block_injection(
    G: dgl.DGLGraph,
    anchors: torch.Tensor,
    candidate: dict,
    deg_per_node: int,
    feats: torch.Tensor = None,
    undirected: bool = True,
    verbose: bool = True,
):
    """
    对每个锚点子图注入 1 个节点并连 K 条边（不扰动特征），打印增量。
    - G: 现有 DGL 图（就地修改）
    - anchors: 形如 (M,) 的长整型 Tensor / list
    - candidate: {anchor_id: [neighbor_ids...]} 来自增强 Bernoulli 的候选
    - deg_per_node: 每个新节点要连的边数 K
    - feats: (N, d) 原始特征（可选；若 G.ndata['feat'] 已存在可不传）
    - undirected: True 时为无向图，自动加反向边
    返回：G, inj_ids(list), anchor_to_injid(dict)
    """
    # --- 设备 & 特征对齐 ---
    base_nodes = G.num_nodes()
    base_edges = G.num_edges()
    print(f"  Original graph size N={base_nodes}, E={base_edges}")
    dev = G.device if hasattr(G, "device") else (feats.device if feats is not None else torch.device("cpu"))
    if 'feat' not in G.ndata:
        if feats is None:
            raise ValueError("G.ndata['feat'] 不存在且未提供 feats。请传入 feats 或先给 G.ndata['feat'] 赋值。")
        G.ndata['feat'] = feats.to(dev)

    # --- 统计 & 初始化 ---
    if torch.is_tensor(anchors):
        anchors_list = anchors.detach().cpu().tolist()
    else:
        anchors_list = list(map(int, anchors))
    inj_ids = []
    anchor_to_injid = {}

    if verbose:
        print("\n[Stage 2b] Injecting 1 node per subgraph using Enhanced-Bernoulli neighbors…")

    # --- 逐子图注入 ---
    for idx, a in enumerate(anchors_list):
        a = int(a)
        before_n, before_e = G.num_nodes(), G.num_edges()

        # 1) 加节点
        anchor_feat = G.ndata['feat'][a]
        new_id = _safe_add_node_with_feat(G, anchor_feat)

        # 特征 = 锚点特征拷贝（不扰动）
        # anchor_feat = G.ndata['feat'][a].unsqueeze(0).to(dev)
        # G.ndata['feat'] = torch.cat([G.ndata['feat'], anchor_feat], dim=0)
        # assert G.ndata['feat'].shape[0] == G.num_nodes(), \
        #     f"feat rows ({G.ndata['feat'].shape[0]}) != num_nodes ({G.num_nodes()})"
        # # feats 外部引用可能已过期，这里返回时你可以用 G.ndata['feat'] 覆盖外部 feats
        inj_ids.append(new_id)
        anchor_to_injid[a] = new_id

        # 2) 选 K 个邻居并连边
        cand_list = candidate.get(a, [])
        cand_list = [int(x) for x in cand_list if int(x) < base_nodes]  # 只允许原始节点，避免连到此前注入的节点

        K = max(0, int(deg_per_node))
        if len(cand_list) >= K:
            neighs = cand_list[:K]
        else:
            need = K - len(cand_list)
            # 从原始节点集合兜底补齐（不含新节点）
            pool = list(range(base_nodes))
            # 去掉已选
            exist = set(cand_list)
            fallback = []
            # 简单无放回补齐
            for v in pool:
                if v not in exist:
                    fallback.append(v)
                    exist.add(v)
                    if len(fallback) >= need:
                        break
            neighs = cand_list + fallback

        if len(neighs) > 0:
            src = torch.full((len(neighs),), new_id, dtype=torch.long, device=dev)
            dst = torch.tensor(neighs, dtype=torch.long, device=dev)
            G.add_edges(src, dst)
            # if undirected:
            #     G.add_edges(dst, src)

        after_n, after_e = G.num_nodes(), G.num_edges()
        # if verbose:
        #     print(f"  [2b] Subgraph {idx:03d} (anchor={a}): +nodes={after_n-before_n}, "
        #           f"+edges={after_e-before_e} -> N={after_n}, E={after_e}")

    if verbose:
        print(f"  [Stage-2b summary] +nodes={G.num_nodes()-base_nodes}, "
              f"+edges={G.num_edges()-base_edges} (now N={G.num_nodes()}, E={G.num_edges()})")

    return G, inj_ids, anchor_to_injid


def _to_bool_s3(x) -> bool:
    """将 DGL 返回的 has_edges_between 结果安全转成 bool。"""
    try:
        if isinstance(x, bool):
            return x
        if torch.is_tensor(x):
            return bool(x.item())
        return bool(x)
    except Exception:
        return False

def _fit_powerlaw_alpha_from_degrees_s3(deg: np.ndarray, kmin: int = 1) -> float:
    """
    用 Clauset 等的近似 MLE（离散幂律）估计 alpha：
    alpha ≈ 1 + n / sum(log(k_i/(kmin-0.5))), 其中 k_i >= kmin
    """
    k = deg[deg >= kmin].astype(np.float64)
    if k.size == 0:
        return 2.0  # 退化兜底
    denom = np.log(k / max(kmin - 0.5, 0.5)).sum()
    if denom <= 1e-12:
        return 2.0
    alpha = 1.0 + k.size / denom
    return float(np.clip(alpha, 1.2, 5.0))

def _sample_trunc_powerlaw_s3(n: int, alpha: float, kmin: int, kmax: int, rs: np.random.RandomState) -> np.ndarray:
    """
    在 [kmin, kmax] 上按离散截断幂律 k^{-alpha} 采样 n 个样本。
    若区间非法或 alpha 异常，退化为均匀采样。
    """
    kmin = int(kmin); kmax = int(kmax)
    if kmax < kmin or kmax <= 0:
        return np.zeros(n, dtype=np.int64)
    ks = np.arange(kmin, kmax + 1, dtype=np.float64)
    if alpha is None or not np.isfinite(alpha) or alpha <= 0:
        p = np.ones_like(ks)
    else:
        p = ks ** (-float(alpha))
    p = p / p.sum()
    return rs.choice(ks.astype(np.int64), size=int(n), replace=True, p=p)

# =========================
# 主函数：Stage-3 预算严格控制的“恶性↔恶性”有向连边
# =========================

@torch.no_grad()
def construct_internal_injected_edges(
        G: dgl.DGLGraph,
        inj_ids: torch.LongTensor,             # 恶性节点ID（len=M）
        deg_per_node: int,                    # 每个恶性节点“总目标度”预算
        intra_ratio: float = 0.3,             # 恶性→恶性外向度占比（相对 deg_per_node）
        per_node_inner_used: Optional[Dict[int, int]] = None,  # 已用于“恶性→正常”内边的度
        undirected: bool = False,             # 已弃用：按有向图处理
        seed: int = 0,
        verbose: bool = True,
        alpha_intra: float = 0.0,             # 0=自动用全图出度拟合；>0=固定指数
        tgt_pref: str = "indegree",           # 'uniform' 或 'indegree'
        tgt_gamma: float = 1.0,               # 仅当 tgt_pref='indegree' 时生效
        outer_mode: str = "per_node",         # 'plaw' | 'per_node' | 'global'
) -> Tuple[dgl.DGLGraph, List[Tuple[int, int]]]:
    """
    有向连边（只加 u->v，不加反向），三种 d_out 分配模式：
    - 'plaw'：按截断幂律逐点采样（原逻辑，sum(d_out) 可能 << 预算）
    - 'per_node'：每个恶性点用满 cap_i => sum(d_out) ≈ 预算（推荐）
    - 'global'：只保证全局之和=预算，各点不超过各自 cap_i

    备注：
    - cap_i = floor(intra_ratio * (deg_per_node - inner_used_i))，并且 cap_i ≤ (M-1)
    - 目标节点默认按入度^gamma 作为权重（tgt_pref='indegree'）；否则均匀
    - 满足去重、自环剔除、已存在边剔除，不做“强制补齐”，但会尽力二次补齐
    """
    # 规范 inj_ids
    if isinstance(inj_ids, list):
        inj_ids = torch.tensor(inj_ids, dtype=torch.long, device=G.device)
    inj_ids = inj_ids.to(G.device)
    M = int(inj_ids.numel())
    if M <= 1:
        if verbose:
            print("[Stage 3] Only one injected node; no directed inter-injected edges.")
        return G, []

    rs = np.random.RandomState(seed)
    dev = G.device

    # 1) 计算每个恶性点的“外向容量上限” cap_i
    base_d = int(round(intra_ratio * float(deg_per_node)))
    caps = np.full(M, base_d, dtype=np.int64)
    if per_node_inner_used is not None:
        caps = np.zeros(M, dtype=np.int64)
        for i, nid in enumerate(inj_ids.tolist()):
            used = int(per_node_inner_used.get(int(nid), 0))
            remain_total = max(0, int(deg_per_node) - used)
            caps[i] = int(np.floor(intra_ratio * remain_total))
    # 不能超过可选目标数
    caps = np.minimum(caps, np.maximum(M - 1, 0))
    total_cap = int(caps.sum())

    if total_cap == 0:
        if verbose:
            print("[Stage 3] No remaining directed out-degree; skip.")
        return G, []

    # 2) 幂律指数 alpha（仅在需要时使用）
    if alpha_intra and alpha_intra > 0:
        alpha = float(alpha_intra)
        src_alpha_mode = "fixed"
    else:
        deg_out = G.out_degrees().to('cpu').numpy().astype(np.int64)
        alpha = _fit_powerlaw_alpha_from_degrees_s3(deg_out, kmin=1)
        src_alpha_mode = "auto"

    if verbose:
        print(f"[Stage 3|Directed-CCDF] Using alpha_out = {alpha:.3f} ({src_alpha_mode})")

    # 3) 分配每个点的外向度 d_out
    if outer_mode == "per_node":
        # 精确用满各自上限 => sum(d_out) == sum(caps)
        d_out = caps.copy()

    elif outer_mode == "global":
        # 只保证全局之和=预算；各点不超过 cap
        if caps.max() == 0:
            d_out = caps.copy()
        else:
            # 用幂律样本作为“分配权重”
            w = _sample_trunc_powerlaw_s3(M, alpha, 1, max(1, int(caps.max())), rs).astype(np.float64)
            w = np.maximum(w, 1e-12)
            w = w / w.sum()
            raw = np.floor(w * total_cap).astype(np.int64)
            d_out = np.minimum(raw, caps)
            # 补齐剩余
            deficit = total_cap - int(d_out.sum())
            if deficit > 0:
                # 优先把余量（cap - d_out）大的点补满
                order = np.argsort(-(caps - d_out + 1e-9 * w))
                for idx in order:
                    if deficit == 0:
                        break
                    give = min(deficit, int(caps[idx] - d_out[idx]))
                    if give > 0:
                        d_out[idx] += give
                        deficit -= give

    elif outer_mode == "plaw":
        # 原逻辑：每点独立 1..cap_i 采样 => 总量可能明显低于预算
        d_raw = np.zeros(M, dtype=np.int64)
        for i in range(M):
            d_raw[i] = int(_sample_trunc_powerlaw_s3(1, alpha, 1, int(caps[i]), rs)[0]) if caps[i] > 0 else 0
        # 超了才压缩（通常不会超）
        if d_raw.sum() > total_cap and d_raw.sum() > 0:
            scale = total_cap / float(d_raw.sum())
            d_raw = np.floor(d_raw * scale).astype(np.int64)
        d_out = np.minimum(d_raw, caps)

    else:
        raise ValueError(f"outer_mode must be one of ['plaw','per_node','global'], got {outer_mode}")

    if d_out.sum() == 0:
        if verbose:
            print("[Stage 3] Zero out-degree after allocation; skip.")
        return G, []

    # 4) 目标偏好权重
    inj_list = inj_ids.tolist()
    indeg_all = G.in_degrees().to(dev).float()
    eps = 1e-12

    def has_uv(u: int, v: int) -> bool:
        if u == v:
            return True
        try:
            # The graph is represented as directed, but the paper's degree
            # budget is an undirected/unique-neighbour budget.  A reciprocal
            # arc therefore must not be counted as a second usable pair.
            h_uv = G.has_edges_between(u, v)
            h_vu = G.has_edges_between(v, u)
            return _to_bool_s3(h_uv) or _to_bool_s3(h_vu)
        except Exception:
            try:
                return bool((G.successors(u) == v).any().item() or
                            (G.successors(v) == u).any().item())
            except Exception:
                return False

    # 5) 逐源点分配目标并加边（尽量满足 d_out[i]）
    src_new: List[int] = []
    dst_new: List[int] = []

    inj_set = set(inj_list)
    inj_pos = {int(nid): i for i, nid in enumerate(inj_list)}
    # caps are incident-degree capacities, not independent out-degree caps.
    # Every new injected-to-injected pair consumes one unit at both endpoints.
    remaining = caps.copy()
    pending_pairs = set()
    for i_pos, u in enumerate(inj_list):
        need = min(int(d_out[i_pos]), int(remaining[i_pos]))
        if need <= 0:
            continue

        # 候选目标（同为恶性且不等于自己）
        tgt_cands = [x for x in inj_list
                     if x != u and remaining[inj_pos[int(x)]] > 0]
        if len(tgt_cands) == 0:
            continue

        # 权重
        if tgt_pref == "indegree":
            w = (indeg_all[torch.tensor(tgt_cands, device=dev)] + eps) ** float(tgt_gamma)
            w = (w / (w.sum() + eps)).detach().cpu().numpy()
        else:
            w = None  # 均匀

        # 第一次尝试：按（入度或均匀）权重，尽量不放回采样
        if w is None:
            order = np.arange(len(tgt_cands))
            rs.shuffle(order)
        else:
            # numpy.choice 在 replace=False + p 时是允许的，但这里我们手动“排序取前 need”
            # 采用“重要性分数 + 极小抖动”的方式近似无放回抽样
            jitter = rs.rand(len(tgt_cands)) * 1e-6
            scores = -np.log(w + eps) + jitter  # 等价于按 w 降序
            order = np.argsort(scores)

        chosen = 0
        for idx in order:
            if chosen >= need:
                break
            v = int(tgt_cands[int(idx)])
            pair_key = (min(int(u), v), max(int(u), v))
            if remaining[inj_pos[v]] <= 0 or has_uv(u, v) or pair_key in pending_pairs:
                continue
            src_new.append(u); dst_new.append(v)
            pending_pairs.add(pair_key)
            remaining[i_pos] -= 1
            remaining[inj_pos[v]] -= 1
            chosen += 1

        # 如果因为“已有边/冲突”没凑够，再做一次“均匀”补齐尝试
        if chosen < need:
            remain = [v for v in tgt_cands
                      if remaining[inj_pos[int(v)]] > 0
                      and not has_uv(u, v)
                      and (min(int(u), int(v)), max(int(u), int(v))) not in pending_pairs]
            if len(remain) > 0:
                rs.shuffle(remain)
                for v in remain:
                    if chosen >= need:
                        break
                    v = int(v)
                    pair_key = (min(int(u), v), max(int(u), v))
                    src_new.append(u); dst_new.append(v)
                    pending_pairs.add(pair_key)
                    remaining[i_pos] -= 1
                    remaining[inj_pos[v]] -= 1
                    chosen += 1
        # 仍不足则接受欠配（不会越过 cap，也不会死循环）

    # 6) 写回（只加 u->v）
    if len(src_new) == 0:
        if verbose:
            print("[Stage 3] No valid directed pairs found under constraints.")
        return G, []

    src_t = torch.tensor(src_new, dtype=torch.long, device=dev)
    dst_t = torch.tensor(dst_new, dtype=torch.long, device=dev)
    G.add_edges(src_t, dst_t)

    if verbose:
        # One directed arc consumes two incident-degree units.
        expect_budget = int((total_cap if outer_mode != "plaw" else d_out.sum()) // 2)
        print(f"[Stage 3|Directed-CCDF] |inj|={M}, sum(d_out)={int(d_out.sum())}, "
              f"expected by budget={expect_budget}, added directed edges={len(src_new)}.")

    return G, list(zip(src_new, dst_new))
