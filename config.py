"""TS-SNIA parameters, experiment selections, and backward-compatible CLI aliases."""
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / 'data'
RESULTS_ROOT = PROJECT_ROOT / 'results'
DATASETS = ('cora', 'citeseer', 'cora_ml', 'pubmed', 'amazon_computers')
VICTIMS = ('gcn', 'sgc', 'gin', 'sage')
# Public-release defaults; optional packages must always be selected explicitly.
DEFENSES = ('gnnguard', 'rgcn', 'prognn', 'gtrans', 'gcn_jaccard')
SMOOTHING_DEFENSES = ('node_aware', 'collective_lp')
DEFENSE_NAMES = DEFENSES + SMOOTHING_DEFENSES
SEEDS = (0, 1, 2)
LCC_DATASETS = frozenset({'cora', 'cora_ml'})
TRAINING = {'epochs': 200, 'patience': 30, 'lr': 0.01, 'weight_decay': 5e-4}
ATTACK = {'injection_ratio': 0.05, 'external_budget_ratio': 0.7,
          'candidate_hops': 1, 'sparse_action_budget': 2, 'nes_steps': 30,
          'nes_directions': 2, 'sigma': 0.25, 'adam_lr': 0.05,
          'feature_l1_weight': 1e-3, 'prototype_alignment_weight': 1.2,
          'anchor_probe_edges': 5}
DEFENSE_TRAINING = {'epochs': 200, 'patience': 30, 'hidden': 64,
                    'dropout': 0.5, 'lr': 0.01, 'weight_decay': 5e-4, 'threshold': 0.1}
DATA_SPLITS = {'cora_citeseer_pubmed': 'fixed stratified 10/10/80, split seed 42',
               'cora_ml': 'stored split if supplied, otherwise stratified 10/10/80 using the run seed',
               'amazon_computers': 'class-stratified 10/10/80, run-level seed'}

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

def build_args(argv=None):
    p = argparse.ArgumentParser(description='TS-SNIA: topology-sensitive anchors, anchor-guided injection, and sparse refinement.')
    p.set_defaults(data_root=str(DATA_ROOT))
    # 数据/受害模型
    p.add_argument(
        '--dataset',
        type=normalize_dataset_name,
        default='citeseer',
        choices=['cora', 'citeseer', 'cora_ml', 'pubmed', 'amazon_computers', 'amazon_photo'],
    )
    p.add_argument(
        '--victim',
        type=normalize_victim_name,
        default='gcn',
        choices=['gcn', 'sgc', 'gcn_jaccard', 'gin', 'sage'],
        help='Victim model. Aliases: SAC=SGC, jaccardgcn=GCN-Jaccard, GraphSAGE=sage.',
    )
    p.add_argument('--epochs', type=int, default=TRAINING['epochs'])
    p.add_argument('--patience', type=int, default=TRAINING['patience'])
    p.add_argument(
        '--fixed_surrogate_epochs', action='store_true',
        help='Train the clean surrogate for exactly --epochs updates and use the final state. '
             'Intended only for training-duration robustness analysis.',
    )
    p.add_argument('--lr', type=float, default=TRAINING['lr'])
    p.add_argument('--wd', type=float, default=TRAINING['weight_decay'])
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--weight_decay', type=float, default=5e-4, help=argparse.SUPPRESS)
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
                   choices=['untargeted','targeted'], help=argparse.SUPPRESS)
    p.add_argument('--target_ids', type=str, default='', help=argparse.SUPPRESS)  # 逗号分隔

    # 预算与稀疏
    p.add_argument('--injected_degree_budget', '--deg_per_node', dest='deg_per_node', type=str, default='auto')  # 'auto' 或者整数
    p.add_argument('--max_rounds', type=int, default=1, help=argparse.SUPPRESS)
    # 超参可调
    p.add_argument('--opt_steps', type=int, default=200, help=argparse.SUPPRESS)
    p.add_argument('--opt_lr', type=float, default=0.1, help=argparse.SUPPRESS)
    p.add_argument('--hid_sur', type=int, default=64, help=argparse.SUPPRESS)

    p.add_argument('--feature_l1_weight', '--lam_l1', type=float, default=ATTACK['feature_l1_weight'])
  
    p.add_argument('--sparse_action_budget', '--nes_edge_budget', type=int, default=ATTACK['sparse_action_budget'],
                   help='Maximum number of edge-addition and internal-edge-removal actions per anchor (Delta L).')
    p.add_argument('--nes_hard_topk', action='store_true',default=True,
                   help=argparse.SUPPRESS)
    p.add_argument('--nes_edge_tau', type=float, default=1.0,
                   help=argparse.SUPPRESS)
    p.add_argument('--nes_edge_thr', type=float, default=0.5,
                   help=argparse.SUPPRESS)
    p.add_argument('--nes_directions', '--bb_queries', type=int, default=ATTACK['nes_directions'],
                   help='Paired NES random directions Q; each direction evaluates the feature objective twice.')
    p.add_argument('--nes_steps', '--steps', type=int, default=ATTACK['nes_steps'])
    p.add_argument('--sigma', type=float, default=ATTACK['sigma'])
    p.add_argument('--adam_lr', type=float, default=ATTACK['adam_lr'])
    p.add_argument('--tau', type=float, default=0.5, help=argparse.SUPPRESS)     # 对比温度
    p.add_argument('--mmd_w', type=float, default=1.0, help=argparse.SUPPRESS)
    p.add_argument('--degkl_w', type=float, default=0.0)
    p.add_argument('--causal_w', type=float, default=1.0, help=argparse.SUPPRESS)
    p.add_argument('--l0_w', type=float, default=1e-3, help=argparse.SUPPRESS)
    p.add_argument('--candidate_hops', '--khop_edge', type=int, default=ATTACK['candidate_hops'],
                        help='围绕锚点进行子图采样的 hop 数；0 表示不用采样，靠 1/2-hop 近邻集')
    p.add_argument('--p_stru', type=float, default=0.8,
                   help=argparse.SUPPRESS)
    p.add_argument('--use_pseudo', action='store_true',default=True,
                   help='Use clean-surrogate pseudo labels for neighbor sampling and prototypes.')
    # 剪边（隐蔽性）
    p.add_argument('--prune_ratio', type=float, default=0.3, help=argparse.SUPPRESS)

    # 训练/日志
    p.add_argument('--rounds', type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument(
        '--model_seed', type=int, default=None,
        help='Optional model-initialization seed; dataset split and attack randomness still use --seed.',
    )

    p.add_argument("--lam_in", default=0.08, type=float)
    p.add_argument("--lam_out", default=0.1, type=float, help=argparse.SUPPRESS)
    p.add_argument("--lam_global", default=0.1, type=float, help=argparse.SUPPRESS)
    p.add_argument("--alpha_intra", default=0.0, type=float, help="0 表示自动用 CCDF 拟合；>0 则固定幂律指数")
    p.add_argument('--lam_edgecls', type=float, default=0.8,
                   help='子图稀疏优化中：优先连向目标类的边约束权重')

    p.add_argument('--log_every', type=int, default=10)

    # ===== 损失形态 =====
    p.add_argument('--loss_type', type=str, default='margin', choices=['margin', 'ce'],
                   help=argparse.SUPPRESS)
    p.add_argument('--prototype_alignment_weight', '--lam_class', type=float, default=ATTACK['prototype_alignment_weight'],
                   help='注入特征与目标类原型对齐的权重')
   
    p.add_argument('--external_budget_ratio', type=float, default=ATTACK['external_budget_ratio'],
                   help='External share of the injected-node degree budget; the internal share is 1-r.')
    p.add_argument('--injection_ratio', '--inj_node_budget', type=float, default=ATTACK['injection_ratio'],
                   help='Injected-node ratio rho, relative to the original node count.')
    
     # Anchor scoring strategy copied from TCNIA.
    p.add_argument('--anchor_score_method', default='topology_sensitivity',
                   type=lambda value: {'causal': 'topology_sensitivity'}.get(value, value),
                   choices=['topology_sensitivity', 'ppr', 'bridge_community'],
                   help='Anchor selection strategy: topology sensitivity, PPR, or bridge/community score.')
    p.add_argument('--anchor_metric', type=str, default='confidence_drop',
                   choices=['confidence_drop', 'entropy_increase', 'margin_drop', 'js_divergence', 'random'])
    p.add_argument('--anchor_probe_cache', type=str, default='',
                   help='Optional shared counterfactual-probability cache for anchor-metric comparisons.')
    p.add_argument('--anchor_drop_top', type=int, default=ATTACK['anchor_probe_edges'])
    p.add_argument('--anchor_drop_strategy', default='highest_degree',
                   choices=['highest_degree', 'lowest_degree', 'random', 'feature_dissimilar'])
    p.add_argument('--anchor_selection', default='top_m',
                   choices=['top_m', 'distance_greedy', 'mmr_local', 'community_stratified'])
    p.add_argument('--anchor_diversity_lambda', type=float, default=0.5)
    p.add_argument('--anchor_output_path', type=str, default='')
    p.add_argument(
        '--anchor_only', action='store_true',
        help='Stop after anchor scoring and write the anchor/scoring outputs.',
    )
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Keep only the largest connected component before training and attack.'
    )
    p.add_argument('--result_json', type=str, default='',
                   help='Optional path for machine-readable experiment metrics.')
    p.add_argument('--artifact_path', type=str, default='',
                   help='Optional DGL .bin path for the final poisoned graph and tensors.')
    p.add_argument(
        '--attack_memory_only', action='store_true',
        help='Measure attack-generation peak CUDA allocation and stop before poison retraining.',
    )
    p.add_argument(
        '--experiment_variant',
        default='full',
        choices=['full', 'random_anchors', 'no_anchor_guidance', 'no_internal_edges',
                 'no_nes_features', 'no_structure_optimization'],
        help='Component-ablation switch; full preserves the paper configuration.',
    )
    p.add_argument(
        '--structure_mode',
        default='independent_ranking',
        type=lambda value: {'joint_static': 'independent_ranking'}.get(value, value),
        choices=['independent_ranking', 'none', 'add_only', 'remove_only', 'sequential', 'beam'],
        help='Allowed sparse structural actions.',
    )
    p.add_argument(
        '--query_budget_fraction', type=float, default=1.0,
        help='Fraction of the full anchor/structure query workload to expose (0, 1].',
    )
    p.add_argument('--beam_width', type=int, default=3)
    p.add_argument('--beam_candidate_limit', type=int, default=8)
    p.add_argument('--structure_trace_path', type=str, default='',
                   help='Optional JSON path for per-action sparse-refinement diagnostics.')
    p.add_argument('--attack_output_mode', default='full_probability',
                   choices=['full_probability', 'quantized_probability', 'topk_probability', 'hard_label'])
    p.add_argument('--attack_output_bits', type=int, default=8)
    p.add_argument('--attack_output_topk', type=int, default=2)
    p.add_argument('--cue_homo_weight', type=float, default=1.0)
    p.add_argument('--cue_degree_weight', type=float, default=1.0)
    p.add_argument('--cue_cos_weight', type=float, default=1.0)
    p.add_argument('--sampling_temperature', type=float, default=0.7)
    p.add_argument('--mode', choices=['train', 'defense'], default='train')
    p.add_argument('--datasets', nargs='+', type=normalize_dataset_name, choices=list(DATASETS))
    p.add_argument('--models', nargs='+', type=normalize_victim_name, choices=list(VICTIMS))
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--defense', choices=list(DEFENSE_NAMES), default=None,
                   help='One defense; collective_lp and node_aware require their optional directories.')
    p.add_argument('--defenses', nargs='+', choices=list(DEFENSE_NAMES), default=None,
                   help='Defenses to evaluate. Defaults to the five empirical defenses only.')
    p.add_argument('--defense_epochs', type=int, default=DEFENSE_TRAINING['epochs'])
    p.add_argument('--defense_patience', type=int, default=DEFENSE_TRAINING['patience'])
    p.add_argument('--defense_hidden', type=int, default=DEFENSE_TRAINING['hidden'])
    p.add_argument('--results_dir', type=Path, default=RESULTS_ROOT)
    p.add_argument('--overwrite', action='store_true', help='Recompute completed cases.')
    p.add_argument('--artifact_input', type=Path, default=None,
                   help='Evaluate defenses on a saved poisoned graph without regenerating the attack.')
    return p.parse_args(argv)
