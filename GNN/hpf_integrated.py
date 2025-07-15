# python filter_utils\hpf_integrated.py

import torch
import numpy as np
import os
import sys
import networkx as nx
from torch_sparse import coalesce
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
sys.path.append('SD-STGCN/utils')  # Adjust to actual path
from math_graph import weight_matrix # type: ignore

def alpha_relu(x, alpha=1.):
    """
    alpha ∈ [0,1]
    Compute threshold f(alpha) = alpha - 1 and using x -> max(x, f(alpha)).
    default alpha = 1 is just relu, for alpha = 0 it is close to original.
    """
    threshold = alpha - 1
    return torch.maximum(x, torch.tensor(threshold, device=x.device, dtype=x.dtype))

def HPF(edge_filepath, probs=None, mode='prob', cutoff_point=700, alpha = 1., n_nodes = 774):
    """
    High-Pass Graph Filtering (HPF) for node-level signals on a fixed-size graph.

    This function performs a spectral high-pass filter on a batch of node-wise signals
    (either class probabilities or raw logits) based on the eigen-decomposition of the
    unnormalized graph Laplacian. It selectively retains the high-frequency components
    above a given cutoff, which can emphasize local structural changes in the graph.

    The input graph is specified via an edge list and is assumed to contain a fixed number
    of nodes (default: 774). Input signals are of shape (batch_size, n_nodes, 2), where
    the last dimension represents either class probabilities (p0, p1) summing to 1, or
    raw logits (l0, l1) without such constraint.

    Parameters
    ----------
    edge_filepath : str
        Path to the edge list file. Each line should contain a pair of node IDs.

    probs : torch.Tensor or None, optional
        A tensor of shape (batch_size, n_nodes, 2). If mode='prob', the last dimension
        should contain class probabilities summing to 1; the second channel (probs[:,:,1])
        is treated as the signal to be filtered. If mode='logits', both channels are
        independently filtered as raw signals.

    mode : str, optional
        'prob' or 'logits'. If 'prob', high-pass filtering is applied to the class-1 
        probability followed by alpha-thresholding and normalization to [0,1]; the output
        retains the probabilistic structure by enforcing p0 + p1 = 1. If 'logits', both
        channels are independently filtered and returned without further processing.

    cutoff_point : int, optional
        Frequency index from which high-pass components are retained. Lower frequencies
        (below cutoff) are suppressed.

    alpha : float, optional
        Regularization parameter for alpha_relu. Only used if mode='prob'. Controls
        thresholding after filtering. alpha=1 is standard ReLU; alpha=0 yields identity.

    n_nodes : int, optional
        Number of nodes expected in the graph. Default is 774.

    Returns
    -------
    output_probs : torch.Tensor
        A tensor of shape (batch_size, n_nodes, 2). If mode='prob', it represents a valid
        class probability distribution after filtering and normalization. If mode='logits',
        it contains the high-pass filtered logits for both classes, preserving original scale.
    """
    g = nx.read_edgelist(edge_filepath)
    node_mapping = {node: idx for idx, node in enumerate(g.nodes())}
    num_nodes = len(node_mapping) 
    assert num_nodes == 774, f"Expected 774 nodes, got {num_nodes}"
    edges = [(node_mapping[src], node_mapping[dst]) for src, dst in g.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t()  # Shape: (2, num_edges)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Add (dst, src)
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=774)
    L = csgraph.laplacian(adj, normed=False)
    _ , eigvec = eigh(L.toarray())
    # we find N=700 is good to split.
    

    if probs is None:
        probs = np.random.rand(8, n_nodes, 2)
        if mode=='prob':
            probs = probs / probs.sum(axis=2, keepdims=True)
        probs = torch.tensor(probs, dtype=torch.float32)
        print(probs[:2,:2,:])

    U = torch.tensor(eigvec, dtype=probs.dtype, device=probs.device, requires_grad=False)  # (n_nodes, n_nodes)
    U_T = U.transpose(0, 1)

    band_filter = np.zeros(U.shape[0], dtype=np.float32)
    band_filter[cutoff_point:] = 1.0
    G = torch.diag(torch.tensor(band_filter, dtype=probs.dtype, device=probs.device, requires_grad=False))  # (n_nodes, n_nodes)

    H = U @ G @ U_T  # (n_nodes, n_nodes)

    if mode == 'prob':
        signal = probs[:, :, 1]  # shape: (batch_size, n_nodes)
        signal_filtered = signal @ H.T  # Apply high-pass filter
        signal_filtered = alpha_relu(signal_filtered, alpha=alpha)

        # Normalize to [0,1]
        min_val = signal_filtered.min(dim=1, keepdim=True)[0]
        max_val = signal_filtered.max(dim=1, keepdim=True)[0]
        eps = 1e-8
        signal_norm = (signal_filtered - min_val) / (max_val - min_val + eps)

        p1 = signal_norm
        p0 = 1.0 - p1
        output_probs = torch.stack([p0, p1], dim=2)  # shape: (batch_size, n_nodes, 2)

    elif mode == 'logits':
        # Separate l0 and l1
        l0 = probs[:, :, 0]  # shape: (batch_size, n_nodes)
        l1 = probs[:, :, 1]

        # Each passes through high-pass filter independently
        l0_filtered = l0 @ H.T
        l1_filtered = l1 @ H.T

        # No alpha_relu or normalization
        output_probs = torch.stack([l0_filtered, l1_filtered], dim=2)

    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'prob' or 'logits'.")

    return output_probs

if __name__=="__main__":
    graph_path = 'SD-STGCN/dataset/highSchool/data/graph/'
    graph_file = 'highSchool.edgelist'
    edge_filepath = graph_path + graph_file
    print(HPF(edge_filepath, mode='prob')[:2,:2,:])