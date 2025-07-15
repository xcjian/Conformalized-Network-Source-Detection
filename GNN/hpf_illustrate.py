# python filter_utils\hpf_illustrate.py

"""
This script demonstrates spectral high-pass filtering of node-level signals on a fixed-size graph.

Key functionalities:
- Loads a graph from 'highSchool.edgelist' with 774 nodes.
- Computes the unnormalized graph Laplacian and its full eigendecomposition.
- Visualizes the Laplacian spectrum and its second-order difference (Δ²λ) to identify meaningful cutoff points.
- Generates synthetic node signals from an exponential distribution with shape (n_nodes, 2).
- Selects the second channel (e.g., l1) as the input signal for filtering.
- Applies an ideal high-pass graph filter by truncating the spectrum below the cutoff index (cutoff=700).
- Does not apply any post-filtering operations such as alpha-ReLU or normalization (commented out).
- Visualizes both the original and filtered signals over the node index domain.

Use this script to analyze how high-frequency components in graph signals behave, assess filter design, 
and understand spectral-domain signal manipulation on graphs.
"""

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

graph_path = 'SD-STGCN/dataset/highSchool/data/graph/'
graph_file = 'highSchool.edgelist'

g = nx.read_edgelist(graph_path + graph_file)
# Map node IDs to consecutive integers (0 to 773)
node_mapping = {node: idx for idx, node in enumerate(g.nodes())}
num_nodes = len(node_mapping)  # Should be 774
assert num_nodes == 774, f"Expected 774 nodes, got {num_nodes}"
# Extract edges and weights
edges = [(node_mapping[src], node_mapping[dst]) for src, dst in g.edges()]
edge_index = torch.tensor(edges, dtype=torch.long).t()  # Shape: (2, num_edges)
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Add (dst, src)
print(edge_index)

adj = to_scipy_sparse_matrix(edge_index, num_nodes=774)
L = csgraph.laplacian(adj, normed=False)

eigval, eigvec = eigh(L.toarray())
print(eigval)

import matplotlib.pyplot as plt

plt.plot(eigval)  
plt.title("Laplacian Spectrum")
plt.xlabel("Index")
plt.ylabel("Eigenvalue")
plt.grid(True)
plt.show()

sorted_eigenvalues = np.sort(eigval)
eigvals = np.array(sorted_eigenvalues)  # shape: (n,)

sliding_window = False
if sliding_window:
    window_size = 3
    window = np.ones(window_size) / window_size
    eigval_post = np.convolve(window,eigvals, mode='valid')
else:
    eigval_post = eigvals

delta_1 = np.diff(eigval_post)  # shape: (n-1,)
delta_2 = np.diff(delta_1)  # shape: (n-2,)
plt.plot(delta_2)
plt.title("Second Derivative of Laplacian Spectrum")
plt.xlabel("Index")
plt.ylabel("Curvature (Δ²λ)")
plt.grid(True)
plt.show()

# we find N=700 is good to split according to the spectrum.
mode = 'logits'
n_nodes = 774
# probs = np.random.rand(n_nodes, 2)
probs = np.random.standard_exponential(size=(n_nodes, 2))

if mode =='prob':
    probs = probs / probs.sum(axis=1, keepdims=True) # block it if you want to see logits

signal = probs[:, 1]

U = eigvec
U_T = U.T
x_hat = U_T @ signal

# CUTOFF

cutoff_point = 700

band_filter = np.zeros(n_nodes)
band_filter[cutoff_point:] = 1.0  # ideal high-pass filter
G = np.diag(band_filter)  


# alpha = -np.inf #settings for alpha-relu

x_hat_filtered = G @ x_hat
signal_filtered = U @ x_hat_filtered
signal_normalized = signal_filtered

# signal_filtered = np.maximum(alpha-1, signal_filtered)

# def normalize_minmax(x):
#     x_min = x.min()
#     x_max = x.max()
#     return (x - x_min) / (x_max - x_min + 1e-8)  
# signal_normalized = normalize_minmax(signal_filtered)

plt.figure(figsize=(10, 4))
plt.plot(signal, label='Original signal')
plt.plot(signal_normalized, label='Filtered signal')
plt.title("Signal Filtering (Band-Pass on Graph)")
plt.xlabel("Node Index")
plt.ylabel("Signal Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
