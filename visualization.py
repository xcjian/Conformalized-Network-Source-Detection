#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-sample source detection visualization (components with sources only) + optional clipping.

- Only displays connected components that contain ≥1 true source.
- Optional rectangular clipping window: keep nodes with x_min ≤ x ≤ x_max and y_min ≤ y ≤ y_max.
- By default, layout coordinates are normalized to [0,1]x[0,1] (toggle with --normalize_layout).

Colors:
  gray  : all nodes in shown subgraph
  green : correctly identified sources (true ∩ predicted)
  orange: falsely identified sources (predicted \ true) within shown subgraph
  red   : sources not identified (true \ predicted) within shown subgraph
"""

import os
import pickle
import argparse
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from utils.score_convert import (
    set_truncate,
    recall_score, recall_score_gtunknown,
    avg_score, avg_score_gtunknown,
    min_score, min_score_gtunknown,
    cpquantile,
)
from DSI.src.diffusion_source.infection_model import FixedTSI
from DSI.src.diffusion_source.discrepancies import ADiT_h

np.random.seed(41)


def build_paths(graph_name, train_exp_name, test_exp_name, pow_expected):
    graph_extract_path = f'SD-STGCN/dataset/{graph_name}/data/graph/{graph_name}.edgelist'
    graph_path = f'data/{graph_name}/graph/{graph_name}.edgelist'
    test_data_file = f'SD-STGCN/dataset/{graph_name}/data/SIR/{test_exp_name}_entire.pickle'
    train_model_file = f'SD-STGCN/output/models/{graph_name}/{train_exp_name}'
    test_res_path = f'SD-STGCN/output/test_res/{graph_name}/{test_exp_name}'
    save_root = f'results/{graph_name}/{test_exp_name}/pow_expected{pow_expected}'
    vis_dir = os.path.join(save_root, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    return graph_extract_path, graph_path, test_data_file, train_model_file, test_res_path, save_root, vis_dir


def ensure_graph(graph_extract_path, graph_path):
    if not os.path.exists(graph_path):
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        os.system(f'cp {graph_extract_path} {graph_path}')
    graph = nx.read_edgelist(graph_path, nodetype=int)
    G = nx.Graph(graph)
    # IMPORTANT: do NOT set G.graph = G
    return G


def ensure_test_results(graph_name, test_exp_name):
    data_extract_path = f'SD-STGCN/output/test_res/{graph_name}/{test_exp_name}/res.pickle'
    data_path = f'data/{graph_name}/test_res/{test_exp_name}/res.pickle'
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        os.system(f'cp {data_extract_path} {data_path}')
    return data_path


def load_and_flatten(data_path, calib_ratio):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    inputs_raw = data['inputs']         # n_batch x n_sample x n_nodes
    pred_scores_raw = data['predictions']
    ground_truths_raw = data['ground_truth']
    logits_raw = data['logits']

    inputs, pred_scores, ground_truths, logits = [], [], [], []
    for i in range(len(pred_scores_raw)):
        for j in range(len(pred_scores_raw[i])):
            inputs.append(inputs_raw[i][j])
            pred_scores.append(pred_scores_raw[i][j])
            ground_truths.append(ground_truths_raw[i][j])
            logits.append(logits_raw[i][j])

    n_samples = len(pred_scores)
    n_calibration = int(n_samples * calib_ratio)
    return (inputs, pred_scores, ground_truths, logits,
            n_samples, n_calibration)


def split_indices(n_samples, n_calibration, save_root, mc_idx=0):
    idx_file = os.path.join(save_root, f'calib_index_repeat{mc_idx}.npy')
    if os.path.exists(idx_file):
        calib_index = np.load(idx_file)
    else:
        calib_index = np.random.choice(n_samples, n_calibration, replace=False)
        np.save(idx_file, calib_index)
    test_index = np.setdiff1d(np.arange(n_samples), calib_index)
    return calib_index, test_index


def compute_threshold(cfscore_calib, alpha, n_calib, tail_sign=+1):
    tail_prop = (1 - alpha) * (1 + 1 / n_calib)
    if tail_sign == +1:
        thresh = cpquantile(cfscore_calib, tail_prop)
    else:
        thresh = -cpquantile(-cfscore_calib, tail_prop)
    return thresh


def predict_set_single_sample(
    method, alpha, sample_idx,
    inputs, pred_scores, ground_truths, logits,
    calib_index, n_nodes, prop_model, pow_expected,
    G=None, m_l=20, m_p=20
):
    inputs_calib = [inputs[i] for i in calib_index]
    pred_scores_calib = [pred_scores[i] for i in calib_index]
    ground_truths_calib = [ground_truths[i] for i in calib_index]
    logits_calib = [logits[i] for i in calib_index]
    n_calib = len(calib_index)

    y_true = ground_truths[sample_idx]
    true_sources = set(np.nonzero(y_true)[0])

    if method in ('set_recall', 'set_prec', 'set_min'):
        cfscore_calib = []
        for i in range(n_calib):
            infected_nodes_ = np.nonzero(inputs_calib[i])[0]
            pred_prob_ = pred_scores_calib[i][:, 1]
            gt_one_hot = ground_truths_calib[i]
            gt_part = set_truncate(gt_one_hot, pred_prob_, pow_expected)

            if method == 'set_recall':
                s = recall_score(pred_prob_, gt_part, prop_model, infected_nodes_)
            elif method == 'set_prec':
                s = avg_score(pred_prob_, gt_part, prop_model, infected_nodes_)
            else:
                s = min_score(pred_prob_, gt_part, prop_model, infected_nodes_)
            cfscore_calib.append(s)
        cfscore_calib = np.array(cfscore_calib)

        infected_nodes_sample = np.nonzero(inputs[sample_idx])[0]
        pred_prob_sample = pred_scores[sample_idx][:, 1]
        if method == 'set_recall':
            cfscore_test = recall_score_gtunknown(pred_prob_sample, prop_model, infected_nodes_sample)
            tail_sign = +1
        elif method == 'set_prec':
            cfscore_test = avg_score_gtunknown(pred_prob_sample, prop_model, infected_nodes_sample)
            tail_sign = +1
        else:
            cfscore_test = min_score_gtunknown(pred_prob_sample, prop_model, infected_nodes_sample)
            tail_sign = +1

        threshold = compute_threshold(cfscore_calib, alpha, n_calib, tail_sign=tail_sign)
        pred_set = set(j for j in range(n_nodes) if cfscore_test[j] <= threshold)
        return pred_set, true_sources

    elif method == 'ADiT_DSI':
        if prop_model != 'SI':
            raise ValueError("ADiT-DSI only supports SI in this script.")
        if G is None:
            raise ValueError("Graph G is required for ADiT-DSI.")
        infected_nodes_sample = np.nonzero(inputs[sample_idx])[0]
        model = FixedTSI(G, [ADiT_h], canonical=True, expectation_after=False,
                         m_l=m_l, m_p=m_p, T=max(len(infected_nodes_sample) - 1, 0))
        confidence_sets = model.confidence_set_mp(infected_nodes_sample, [alpha], new_run=True)
        confidence_sets = confidence_sets['ADiT_h']
        pred_set = set(confidence_sets[str(alpha)])
        return pred_set, true_sources

    else:
        raise ValueError(f"Unknown method: {method}")


def make_layout(G, layout='spring', seed=41):
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=seed)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G, scale=1.0)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed)
    return pos


def normalize_positions(pos):
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()
    W = max(maxx - minx, 1e-12)
    H = max(maxy - miny, 1e-12)
    S = 1.0 / max(W, H)  # isotropic
    return {n: ((x - minx) * S, (y - miny) * S) for n, (x, y) in pos.items()}


def clip_subgraph_by_bbox(G, pos, x_min, x_max, y_min, y_max):
    """Keep only nodes with x_min ≤ x ≤ x_max and y_min ≤ y ≤ y_max; return subgraph + clipped pos."""
    if x_min >= x_max or y_min >= y_max:
        raise ValueError(f"Invalid clip box: [{x_min}, {x_max}] × [{y_min}, {y_max}]")
    keep_nodes = [n for n, (x, y) in pos.items() if (x_min <= x <= x_max and y_min <= y <= y_max)]
    if len(keep_nodes) == 0:
        raise RuntimeError("Clipping removed all nodes; relax the bounds.")
    H = G.subgraph(keep_nodes).copy()
    pos_clip = {n: pos[n] for n in H.nodes()}
    return H, pos_clip


def visualize(G, pred_set, true_sources, pos, out_path,
              base_node_size=80, highlight_size=140, linewidths=0.2):
    nodes = list(G.nodes())
    index_of = {u: i for i, u in enumerate(nodes)}
    color_map = ['lightgray'] * len(nodes)
    sizes = [base_node_size] * len(nodes)

    true_and_pred = true_sources.intersection(pred_set)
    false_pos = pred_set.difference(true_sources)
    missed = true_sources.difference(pred_set)

    for u in true_and_pred:
        i = index_of[u]; color_map[i] = 'tab:green'; sizes[i] = highlight_size
    for u in false_pos:
        i = index_of[u]; color_map[i] = 'tab:orange'; sizes[i] = highlight_size
    for u in missed:
        i = index_of[u]; color_map[i] = 'tab:red'; sizes[i] = highlight_size

    plt.figure(figsize=(8, 7))
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=sizes,
                           linewidths=linewidths, edgecolors='k')
    plt.axis('off')

    legend_elements = [
        # Patch(facecolor='lightgray', edgecolor='k', label='All nodes'),
        # Patch(facecolor='lightgray', edgecolor='k'),
        Patch(facecolor='tab:green', edgecolor='k', label='Correctly identified'),
        Patch(facecolor='tab:orange', edgecolor='k', label='Falsely identified'),
        Patch(facecolor='tab:red', edgecolor='k', label='Sources not identified'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def nodes_in_source_components(G, true_sources):
    if not true_sources:
        return set()
    keep = set()
    for comp_nodes in nx.connected_components(G):
        comp_nodes = set(comp_nodes)
        if comp_nodes & true_sources:
            keep |= comp_nodes
    return keep


def main():
    parser = argparse.ArgumentParser()

    # --- original-ish args ---
    parser.add_argument('--graph', type=str, default='highSchool')
    parser.add_argument('--train_exp_name', type=str, default='SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16')
    parser.add_argument('--test_exp_name', type=str,  default='SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16')
    parser.add_argument('--pow_expected', type=float, default=0.7)
    parser.add_argument('--calib_ratio', type=float, default=0.95)
    parser.add_argument('--prop_model', type=str, default='SI')
    parser.add_argument('--confi_levels', nargs='*', type=float, default=[0.05, 0.07, 0.10, 0.15, 0.20])
    parser.add_argument('--mc_runs', type=int, default=1)

    parser.add_argument('--set_recall', type=int, default=1)
    parser.add_argument('--set_prec', type=int, default=0)
    parser.add_argument('--set_min', type=int, default=0)
    parser.add_argument('--ADiT_DSI', type=int, default=0)
    parser.add_argument('--PGM_CQC', type=int, default=0)
    parser.add_argument('--ArbiTree_CQC', type=int, default=0)

    parser.add_argument('--m_l', type=int, default=20)
    parser.add_argument('--m_p', type=int, default=20)

    parser.add_argument('--sample_index', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.10)
    parser.add_argument('--method', type=str, default='set_recall', choices=['set_recall', 'set_prec', 'set_min'])
    parser.add_argument('--layout', type=str, default='spring', choices=['spring', 'kamada_kawai', 'spectral'])

    # --- NEW: layout normalization & clipping controls ---
    parser.add_argument('--normalize_layout', type=int, default=1,
                        help='1: rescale layout to [0,1]x[0,1] before clipping, 0: raw layout coords')
    parser.add_argument('--x_min', type=float, default=0.5)
    parser.add_argument('--x_max', type=float, default=0.8)
    parser.add_argument('--y_min', type=float, default=0.2)
    parser.add_argument('--y_max', type=float, default=0.6)

    args = parser.parse_args()

    # paths & graph
    gext, gpath, test_data_file, train_model_file, test_res_path, save_root, vis_dir = build_paths(
        args.graph, args.train_exp_name, args.test_exp_name, args.pow_expected
    )
    G = ensure_graph(gext, gpath)

    data_path = ensure_test_results(args.graph, args.test_exp_name)
    (inputs, pred_scores, ground_truths, logits,
     n_samples, n_calib) = load_and_flatten(data_path, args.calib_ratio)

    if args.sample_index < 0 or args.sample_index >= n_samples:
        raise IndexError(f"--sample_index={args.sample_index} out of range [0, {n_samples-1}]")

    calib_index, _ = split_indices(n_samples, n_calib, os.path.dirname(vis_dir), mc_idx=0)

    pred_set, true_sources = predict_set_single_sample(
        method=args.method,
        alpha=args.alpha,
        sample_idx=args.sample_index,
        inputs=inputs,
        pred_scores=pred_scores,
        ground_truths=ground_truths,
        logits=logits,
        calib_index=calib_index,
        n_nodes=G.number_of_nodes(),
        prop_model=args.prop_model,
        pow_expected=args.pow_expected,
        G=G,
        m_l=args.m_l,
        m_p=args.m_p
    )

    print(f"Sample {args.sample_index}: |predicted|={len(pred_set)}, |true|={len(true_sources)}")
    print(f"Breakdown -> correct:{len(true_sources & pred_set)}  false+:{len(pred_set - true_sources)}  missed:{len(true_sources - pred_set)}")

    # keep only components containing true sources
    keep_nodes = nodes_in_source_components(G, true_sources)
    if not keep_nodes:
        raise RuntimeError("No true sources found in this sample; nothing to display.")
    H = G.subgraph(keep_nodes).copy()

    # Restrict sets to H
    pred_set_H = pred_set & set(H.nodes())
    true_sources_H = true_sources & set(H.nodes())

    # Layout, optional normalization
    pos = make_layout(H, layout=args.layout, seed=41)
    if args.normalize_layout:
        pos = normalize_positions(pos)

    # Optional clipping
    do_clip = all(v is not None for v in (args.x_min, args.x_max, args.y_min, args.y_max))
    if do_clip:
        H, pos = clip_subgraph_by_bbox(H, pos, args.x_min, args.x_max, args.y_min, args.y_max)
        pred_set_H = pred_set_H & set(H.nodes())
        true_sources_H = true_sources_H & set(H.nodes())

    # Output
    suffix = "_norm" if args.normalize_layout else "_raw"
    if do_clip:
        suffix += f"_clip[{args.x_min}-{args.x_max}]x[{args.y_min}-{args.y_max}]"
    out_png = os.path.join(
        vis_dir,
        f'sample{args.sample_index}_{args.method}_alpha{args.alpha:.2f}_sources.png'
    )
    visualize(H, pred_set_H, true_sources_H, pos, out_png)
    print(f"Saved figure to: {out_png}")


if __name__ == '__main__':
    main()
