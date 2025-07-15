import networkx as nx
import numpy as np
import itertools
import pickle
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
import tensorflow as tf
from os.path import join as pjoin

import sys
sys.path.append('./SD-STGCN/data_loader')
from data_utils import *
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler


from collections import defaultdict

def MPmaxscore(Y_hat, edges, alpha, beta):
    """
    Efficiently computes maximum scores using two-phase message passing.
    
    Arguments:
        Y_hat: (K,) array with {-1, 1} or real-valued elements
        edges: List of tuples representing undirected edges in the tree
        alpha: Dictionary of node potentials (alpha[k] for node k)
        beta: Dictionary of edge potentials (beta[(k,l)] for edge (k,l))
        
    Returns:
        maxscores: (K,) array of maximum scores for each y_k=1
    """
    K = len(Y_hat)
    
    # 1. Build tree structure and select root
    neighbors = defaultdict(list)
    for k, l in edges:
        neighbors[k].append(l)
        neighbors[l].append(k)
    root = 0  # Arbitrarily choose node 0 as root
    
    # 2. Compute node potentials
    phi_vals = {}
    for k in range(K):
        phi_vals[k] = alpha[k] * Y_hat[k] * np.array([-1, 1])  # [score for yk=-1, yk=1]
    
    # 3. Compute edge potentials
    psi_vals = {}
    for edge in edges:
        k, l = edge
        psi_vals[edge] = beta[edge] * np.array([[1, -1], [-1, 1]])
        psi_vals[(l, k)] = psi_vals[edge].T  # Add reverse direction
    
    # 4. Upward pass (leaves to root)
    up_msgs = {}
    
    # Identify leaves (nodes with only one connection that aren't root)
    leaves = [k for k in range(K) if len(neighbors[k]) == 1 and k != root]
    
    # Initialize queue with leaves
    queue = leaves.copy()
    visited = set(queue)
    
    while queue:
        node = queue.pop(0)

        for parent in neighbors[node]:
            if (parent, node) not in up_msgs:
                break
        
        # Compute upward message from node to parent
        msg = np.zeros(2)
        for y_parent in [0, 1]:  # y_tilde in message formula
            max_val = -np.inf
            for y_node in [0, 1]:  # y_l in message formula
                term = phi_vals[node][y_node]
                term += psi_vals[(node, parent)][y_node, y_parent]
                # Sum messages from children (in upward direction)
                for child in neighbors[node]:
                    if child != parent and (child, node) in up_msgs:
                        term += up_msgs[(child, node)][y_node]
                max_val = max(max_val, term)
            msg[y_parent] = max_val
        up_msgs[(node, parent)] = msg
        
        # Add parent to queue if all its children have sent messages
        n_remained_neighbor = len(neighbors[parent])
        for child in neighbors[parent]:
            if (child, parent) in up_msgs:
                n_remained_neighbor = n_remained_neighbor - 1
        if n_remained_neighbor == 1 or (n_remained_neighbor == 0 and parent == root):
            parent_ready = True
        else:
            parent_ready = False
        
        if parent_ready and parent != root and parent not in visited:
            queue.append(parent)
            visited.add(parent)
    
    # 5. Downward pass (root to leaves)
    down_msgs = {}
    queue = [root]
    
    while queue:
        node = queue.pop(0)
        
        for child in neighbors[node]:

            if (child, node) not in down_msgs:
                # Compute downward message from node to child
                msg = np.zeros(2)
                for y_child in [0, 1]:  # y_tilde in message formula
                    max_val = -np.inf
                    for y_node in [0, 1]:  # y_l in message formula
                        term = phi_vals[node][y_node]
                        term += psi_vals[(node, child)][y_node, y_child]
                        # Sum all incoming messages except from child
                        for neighbor in neighbors[node]:
                            if neighbor != child:
                                # Use upward message if coming from below
                                if (neighbor, node) in up_msgs:
                                    term += up_msgs[(neighbor, node)][y_node]
                                # Use downward message if coming from above
                                if (neighbor, node) in down_msgs:
                                    term += down_msgs[(neighbor, node)][y_node]
                        max_val = max(max_val, term)
                    msg[y_child] = max_val
                down_msgs[(node, child)] = msg
                queue.append(child)
    
    # print messages
    # print("up messages:", up_msgs)
    # print("down messages:", down_msgs)


    # 6. Compute max scores for each y_k=1
    maxscores = np.zeros(K)
    for k in range(K):
        # Start with node potential for y_k=1
        total = phi_vals[k][1]
        
        # Sum all incoming messages
        for neighbor in neighbors[k]:
            if (neighbor, k) in up_msgs:
                total += up_msgs[(neighbor, k)][1]
            if (neighbor, k) in down_msgs:
                total += down_msgs[(neighbor, k)][1]
        
        maxscores[k] = total
    
    return maxscores

def ArbiTree(Y, Y_hat):
    """
    This function calculates the tree and parameters needed for computing the ArbiTree score function.

    Arguments:
    Y: (n_samples x n_labels) array, with {-1, 1} elements.
    Y_hat: (n_samples x n_labels) array, with {-1, 1} or real-valued elements.

    Returns:
    The maximum spanning tree and the corresponding weights.
    edges: the edge list of MST.
    alpha, beta: the parameters on nodes and edges.
    """

    n_sample, K = Y.shape

    # learn the maximum spanning tree

    ## compute the edge weights
    edge_weights = (Y.T @ Y) ** 2 / 4 # convex conjugate
    np.fill_diagonal(edge_weights, 0)  # Nullify diagonal

    ## create a graph
    G = nx.Graph()
    for i in range(K):
        for j in range(i + 1, K):  # Upper triangle to avoid duplicates
            if edge_weights[i, j] > 0:  # Add only positive edges
                G.add_edge(i, j, weight=edge_weights[i, j])
    
    ## Compute MaxST using Kruskal's algorithm
    edges = list(nx.maximum_spanning_edges(G, algorithm="kruskal", data=True))
    edges = [(edge_[0], edge_[1]) for edge_ in edges]
    # maxst = nx.Graph()
    # maxst.add_edges_from(maxst_edges)

    # learn the parameters for scoring function

    ## node parameters:
    alpha_vec = np.diag(Y.T @ Y_hat) / 2
    alpha = {}
    for k in range(K):
        alpha[k] = alpha_vec[k]

    ## edge parameters:
    beta_mat = Y.T @ Y / 2
    beta = {}
    for edge_ in edges:
        beta[edge_] = beta_mat[edge_[0], edge_[1]]

    return edges, alpha, beta

def PGMTree(Y, S, from_graph = True, G = {}):
    """
    This function calculates the tree and parameters needed for computing the PGM score function.

    Arguments:
    Y: (n_samples x n_labels) array, with {-1, 1} elements.
    S: (n_samples x n_labels) array. The (i, k)-th entry is the score s_k over the i-th sample. 
    from_graph: if true, then the MST will be directly learned from a provided graph.

    Returns: 
    The maximum spanning tree and the corresponding weights.
    edges: the edge list of MST.
    alpha, beta: the parameters on nodes and edges.
    """

    n_samples, K = Y.shape

    if not from_graph:

        # compute the mutual information matrix

        G = nx.Graph()
        for i in range(K - 1):
            for j in range(i + 1, K):
                
                Y_ = Y[:, [i, j]]
                S_ = S[:, [i, j]]
                edge_ = [(0, 1)]

                # fit alpha, beta on this single-edge graph
                alpha_, beta_ = fit_model(Y_, S_, edge_)

                # estimate edge empirical mutual information
                p_joint = np.zeros(n_samples)
                p_k = np.zeros(n_samples)
                p_l = np.zeros(n_samples)
                for n in range(n_samples):
                    score_ = S_[n, :]
                    y_ = Y_[n, :]
                    p_single_, p_pair_ = compute_model_marginals(alpha_, beta_, score_, edge_)
                    p_joint[n] = p_pair_[(0, 1)][0 if y_[0] == -1 else 1][0 if y_[1] == -1 else 1]
                    p_k[n] = p_single_[0][0 if y_[0] == -1 else 1]
                    p_l[n] = p_single_[1][0 if y_[1] == -1 else 1]
                I_e_hat = np.sum(np.log(p_joint / (p_k * p_l)))

                G.add_edge(i, j, weight=I_e_hat)
    
    # learn the maximal spanning tree and the corresponding parameters

    ## NetworkX computes MaxST by negating weights
    maxst_edges = nx.maximum_spanning_edges(G, algorithm="kruskal", data=True)
    edges = [(e[0], e[1]) for e in list(maxst_edges)]

    ## compute the alpha and beta parameters
    alpha, beta = fit_model(Y, S, edges, num_iters=1000, lr=10 ** (-3))

    return edges, alpha, beta

def fit_model(Y, S, edges, num_iters=300, lr=10 ** (-1)):
    """Gradient descent solver.
    
    Arguements:
    Y: (n_samples x n_labels) array, with {-1, 1} elements.
    S: (n_samples x n_labels) array. The (i, k)-th entry is the score s_k over the i-th sample.
    edges: a list of tuples.
    """
    n_samples, K = Y.shape
    n_edge = len(edges)
    C = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, -1, -1],
    ])

    alpha = {k: np.zeros(2) for k in range(K)}
    beta = {(k, l): np.zeros(4) for (k, l) in edges}
    beta_tilde = {(k, l): np.zeros(3) for (k, l) in edges}

    alpha_vec = np.concatenate([alpha[k] for k in range(K)])
    beta_vec = np.concatenate([beta[edge] for edge in edges])
    beta_tilde_vec = np.concatenate([beta_tilde[edge] for edge in edges])
    
    for it in range(num_iters):
        grad_alpha_vec, grad_beta_tilde_vec = compute_gradient(alpha, beta, Y, S, edges)
        
        # Update parameters
        alpha_vec = alpha_vec + lr * grad_alpha_vec
        beta_tilde_vec = beta_tilde_vec + lr * grad_beta_tilde_vec

        # print the gradient norm
        grad_norm = (np.linalg.norm(grad_alpha_vec) ** 2 + np.linalg.norm(grad_beta_tilde_vec) ** 2) ** (1/2)
        print("gradient norm:", grad_norm)
        
        # evaluate the objective function
        beta_vec = np.reshape(np.reshape(beta_tilde_vec, (n_edge, 3)) @ C.T, n_edge * 4)
        alpha, beta = unflatten_params(alpha_vec, beta_vec, K, edges)
        # obj = compute_function_value(alpha, beta, Y, S, edges)
        # print(f"Iter {it}: Objective = {obj:.4f}")
    
    return alpha, beta

def compute_gradient(alpha, beta, Y, S, edges):
    """Compute gradient with expectation subtracted.
    
    In this function, the constraint sum(beta) = 0 is imposed.
    In other words, assume beta = C @ beta_tilde. then beta_tilde is 3-dim.
    We only compute the gradient w.r.t. beta_tilde.
    """
    n_samples, K = Y.shape
    grad_alpha = {k: np.zeros(2) for k in range(K)}
    grad_beta = {(k, l): np.zeros(4) for (k, l) in edges}
    grad_beta_tilde = {(k, l): np.zeros(3) for (k, l) in edges}
    
    for i in range(n_samples):
        # Get model expectations
        E_phi, E_psi = compute_model_expectations(alpha, beta, S[i], edges)
        
        # Observed terms
        for k in range(K):
            grad_alpha[k] += phi_func(Y[i, k], S[i, k])
        for (k, l) in edges:
            grad_beta[(k, l)] += psi_func(Y[i, k], Y[i, l])

        # Subtract expected terms
        for k in range(K):
            grad_alpha[k] -= E_phi[k]
        for (k, l) in edges:
            grad_beta[(k, l)] -= E_psi[(k,l)]
    
    # compute the gradient w.r.t. beta_tilde
    C = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, -1, -1],
    ])
    for (k, l) in edges:
        grad_beta_tilde[(k, l)] = C.T @ grad_beta[(k, l)]

    # Concatenate to obtain the gradient vectors
    grad_alpha_vec = np.concatenate([grad_alpha[k] for k in range(K)])
    grad_beta_tilde_vec = np.concatenate([grad_beta_tilde[edge] for edge in edges])

    return grad_alpha_vec, grad_beta_tilde_vec

def compute_model_expectations(alpha, beta, s, edges):
    """Compute E[phi(Y_k)] and E[psi(Y_k, Y_l)] from marginals."""
    p_single, p_pair = compute_model_marginals(alpha, beta, s, edges)
    
    K = len(alpha)
    E_phi = {k: np.zeros(2) for k in range(K)}
    E_psi = {(k, l): np.zeros(4) for (k, l) in edges}
    
    for k in range(K):
        probs = p_single[k]  # [p(Y_k=-1), p(Y_k=1)]
        s_k = s[k]
        # phi: dimension 2
        E_phi[k][0] = (-s_k) * probs[0]  # For Y_k = -1
        E_phi[k][1] = (s_k) * probs[1]   # For Y_k = +1
        
    for (k, l) in edges:
        probs = p_pair[(k, l)]  # shape (2,2), p(Y_k, Y_l)
        # psi: dimension 4
        # map (Y_k, Y_l): (-1,-1), (1,-1), (-1,1), (1,1) → index 0,1,2,3
        E_psi[(k,l)][0] = probs[0,0]  # Y_k = -1, Y_l = -1
        E_psi[(k,l)][1] = probs[1,0]  # Y_k = 1,  Y_l = -1
        E_psi[(k,l)][2] = probs[0,1]  # Y_k = -1, Y_l = 1
        E_psi[(k,l)][3] = probs[1,1]  # Y_k = 1,  Y_l = 1
        
    return E_phi, E_psi


def compute_function_value(alpha, beta, Y, S, edges):
    """Compute objective correctly according to formula (22) using marginals."""
    n_samples, K = Y.shape
    total = 0
    for i in range(n_samples):
        y = Y[i]
        s = S[i]
        
        # Get marginals
        p_single, p_pair = compute_model_marginals(alpha, beta, s, edges)
        
        local_score = 0
        
        # First sum over nodes
        for k in range(K):
            yk = y[k]
            if yk == -1:
                prob = p_single[k][0]
            else:
                prob = p_single[k][1]
            local_score += np.log(prob + 1e-10)  # add small constant for stability

        # Then sum over edges
        for (k, l) in edges:
            yk = y[k]
            yl = y[l]
            # Get probabilities
            idx_yk = 0 if yk == -1 else 1
            idx_yl = 0 if yl == -1 else 1
            
            p_joint = p_pair[(k,l)][idx_yk, idx_yl]
            p_k = p_single[k][idx_yk]
            p_l = p_single[l][idx_yl]
            
            ratio = (p_joint + 1e-10) / (p_k * p_l + 1e-10)
            local_score += np.log(ratio)

        total += local_score

    return total


def compute_model_marginals(alpha, beta, s, edges):
    """Compute singleton and pairwise marginals by summing over neighbor configurations."""
    
    K = len(alpha)
    G = nx.Graph()
    G.add_nodes_from(range(K))
    G.add_edges_from(edges)

    # Precompute unary potentials (already exponentiated)
    unary = {}
    for k in range(K):
        unary[k] = np.zeros(2)
        for idx, yk in enumerate([-1, 1]):
            unary[k][idx] = np.exp(alpha[k] @ phi_func(yk, s[k]) / G.degree(k))
    
    # Precompute pairwise potentials (already exponentiated)
    pairwise = {}
    for (k, l) in edges:
        pot = np.zeros((2,2))
        for j, yk in enumerate([-1,1]):
            for i, yl in enumerate([-1,1]):
                pot[j,i] = np.exp(beta[(k,l)] @ psi_func(yk, yl)) * unary[k][0 if yk == -1 else 1] * unary[l][0 if yl == -1 else 1]
        pairwise[(k,l)] = pot
    
    # precompute messages over edges
    messages = {}
    for (k, l) in edges: 
        msg_k2l = np.zeros(2) 
        msg_l2k = np.zeros(2) 
        for j, yk in enumerate([-1, 1]): # compute m_{k\to l}
            msg_k2l[j] = np.sum(pairwise[(k,l)][:, j]) # sum over yk
        
        for i, yl in enumerate([-1, 1]): # compute m_{l\to k}
            msg_l2k[i] = np.sum(pairwise[(k,l)][i, :]) # sum over yl
        
        messages[(k,l)] = msg_k2l
        messages[(l,k)] = msg_l2k

    # ---- Pairwise Marginals ----

    ## Initialize pairwise marginals
    p_pair = {}
    for (u,v) in edges:
        p_pair[(u,v)] = pairwise[(u,v)]  # (yu, yv)

        neighbors_u = list(G.neighbors(u))
        neighbors_v = list(G.neighbors(v))
        neighbors_u = [n for n in neighbors_u if n != v]
        neighbors_v = [n for n in neighbors_v if n != u]

        for u_neighbor in neighbors_u:
            for idx_ in [0, 1]:
                p_pair[(u,v)][:, idx_] = p_pair[(u,v)][:, idx_] * messages[(u_neighbor, u)]
        
        for v_neighbor in neighbors_v:
            for idx_ in [0, 1]:
                p_pair[(u,v)][idx_, :] = p_pair[(u,v)][idx_, :] * messages[(v_neighbor, v)]
    
        p_pair[(u,v)] = p_pair[(u,v)] / np.sum(p_pair[(u,v)])
    
    # ---- Singleton Marginals----
    p_single = {}
    for (u,v) in edges:

        if u not in p_single:
            p_single[u] = np.sum(p_pair[(u,v)], axis=1)
            p_single[u] = p_single[u] / np.sum(p_single[u])
        
        if v not in p_single:
            p_single[v] = np.sum(p_pair[(u,v)], axis=0)
            p_single[v] = p_single[v] / np.sum(p_single[v])

    return p_single, p_pair

def compute_model_marginals_bf(alpha, beta, s, edges):
    """Compute singleton and pairwise marginals by summing over neighbor configurations."""
    
    K = len(alpha)
    G = nx.Graph()
    G.add_nodes_from(range(K))
    G.add_edges_from(edges)

    # Precompute unary potentials (already exponentiated)
    unary = {}
    for k in range(K):
        unary[k] = np.zeros(2)
        for idx, yk in enumerate([-1, 1]):
            unary[k][idx] = np.exp(alpha[k] @ phi_func(yk, s[k]))
    
    # Precompute pairwise potentials (already exponentiated)
    pairwise = {}
    for (k, l) in edges:
        pot = np.zeros((2,2))
        for i, yl in enumerate([-1,1]):
            for j, yk in enumerate([-1,1]):
                pot[i,j] = np.exp(beta[(k,l)] @ psi_func(yl, yk))
        pairwise[(k,l)] = pot
        pairwise[(l,k)] = pot.T  # Symmetric

    # ---- Singleton Marginals ----
    p_single = {}
    for v in range(K):
        probs = np.zeros(2)  # (-1, +1)

        neighbors = list(G.neighbors(v))
        num_neighbors = len(neighbors)
        
        # Enumerate over all neighbors' label settings
        for y_neighbors in itertools.product([-1, 1], repeat=num_neighbors):
            for idx, yv in enumerate([-1,1]):
                val = unary[v][idx]
                for l_idx, l in enumerate(neighbors):
                    yl = y_neighbors[l_idx]

                    # try:
                    #     val *= unary[l][0 if yl==-1 else 1] * np.exp(beta[(v,l)] @ psi_func(yv, yl))
                    # except:
                    #     val *= unary[l][0 if yl==-1 else 1] * np.exp(beta[(l,v)] @ psi_func(yv, yl))

                    if (v,l) in edges:
                        val *= unary[l][0 if yl==-1 else 1] * np.exp(beta[(v,l)] @ psi_func(yv, yl))
                    else:
                        val *= unary[l][0 if yl==-1 else 1] * np.exp(beta[(l,v)] @ psi_func(yl, yv))
                    
                probs[idx] += val
        
        # Normalize
        probs /= np.sum(probs)
        p_single[v] = probs

    # ---- Pairwise Marginals ----
    p_pair = {}
    for (u,v) in edges:
        probs = np.zeros((2,2))  # (yl, yk)

        neighbors_u = list(G.neighbors(u))
        neighbors_v = list(G.neighbors(v))
        neighbors_u = [n for n in neighbors_u if n != v]
        neighbors_v = [n for n in neighbors_v if n != u]
        
        total_neighbors = list(set(neighbors_u + neighbors_v))
        num_total_neighbors = len(total_neighbors)

        for y_neighbors in itertools.product([-1, 1], repeat=num_total_neighbors):
            neighbor_assignment = {total_neighbors[i]: y_neighbors[i] for i in range(num_total_neighbors)}

            for i, yu in enumerate([-1,1]):
                for j, yv in enumerate([-1,1]):

                    # try:
                    #     val = unary[u][i] * unary[v][j] * np.exp(beta[(u,v)] @ psi_func(yu, yv))
                    # except:
                    #     val = unary[u][i] * unary[v][j] * np.exp(beta[(v,u)] @ psi_func(yu, yv))
                    val = unary[u][i] * unary[v][j] * np.exp(beta[(u,v)] @ psi_func(yu, yv))                        
                    
                    for n in neighbors_u:
                        yn = neighbor_assignment[n]
                        # edge = (u,n) if (u,n) in pairwise else (n,u)
                        # try:
                        #     val *= np.exp(beta[u,n] @ psi_func(yn, yu))
                        # except:
                        #     val *= np.exp(beta[n,u] @ psi_func(yn, yu))
                        if (u,n) in edges:
                            val *= np.exp(beta[u,n] @ psi_func(yu, yn))
                        else:
                            val *= np.exp(beta[n,u] @ psi_func(yn, yu))

                    for n in neighbors_v:
                        yn = neighbor_assignment[n]
                        # edge = (v,n) if (v,n) in pairwise else (n,v)
                        # try:
                        #     val *= np.exp(beta[v,n] @ psi_func(yn, yv))
                        # except:
                        #     val *= np.exp(beta[n,v] @ psi_func(yn, yv))
                        if (v,n) in edges:
                            val *= np.exp(beta[v,n] @ psi_func(yv, yn))
                        else:
                            val *= np.exp(beta[n,v] @ psi_func(yn, yv))
                    
                    probs[i,j] += val

        # Normalize
        probs /= np.sum(probs)
        p_pair[(u,v)] = probs

    return p_single, p_pair

def unflatten_params(alpha_vec, beta_vec, K, edges):
    """ Unflatten parameter vector into alpha, beta dictionaries. """
    alpha = {}
    beta = {}
    idx = 0
    for k in range(K):
        alpha[k] = alpha_vec[idx:idx+2]
        idx += 2
    idx = 0
    for (k, l) in edges:
        beta[(k, l)] = beta_vec[idx:idx+4]
        idx += 4
    return alpha, beta

def vectorize_params(alpha_dict, beta_dict, K, edges):
    "concatenate parameter in dictionaries to alpha, beta vectors."

    alpha_vec = np.concatenate([alpha_dict[k] for k in range(K)])
    beta_vec = np.concatenate([beta_dict[edge] for edge in edges])

    return alpha_vec, beta_vec

def psi_func(y_l, y_k):
    """Corrected: map (y_l, y_k) into 4-dim one-hot vector, matching order."""
    psi = np.zeros(4)
    if (y_l, y_k) == (-1, -1):
        psi[0] = 1
    elif (y_l, y_k) == (1, -1):
        psi[1] = 1
    elif (y_l, y_k) == (-1, 1):
        psi[2] = 1
    elif (y_l, y_k) == (1, 1):
        psi[3] = 1

    return psi

def phi_func(y_k, s_k):
    # s_k is scalar, output 2-dimensional vector
    phi = np.zeros(2)
    if y_k == -1:
        phi[0] = -s_k
    else:  # y_k == +1
        phi[1] = s_k
    return phi


def get_opfeatures(data_file, model_file, train_pct, n_node, n_frame=16, val_pct=0, end=-1):
    """
    This function gets the output feature from the GNN.
    Args:
    data_file: str. path to the dataset.
    model_file: str. path to the pre-trained model.
    train_pct: float. training percentage.

    Returns:
    features: (n_feature x n_samples) array.
    ground_truths: (n_nodes x n_samples) 0-1 array.
    """

    # load data
    inputs = data_gen(data_file, n_node, n_frame, train_pct, val_pct)

    # load model
    model_path = tf.train.get_checkpoint_state(model_file).model_checkpoint_path

    test_graph = tf.Graph()

    with test_graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    with tf.compat.v1.Session(graph=test_graph) as test_sess:
        saver.restore(test_sess, tf.train.latest_checkpoint(model_file))
        print(f'>> Loading saved model from {model_path} ...')

        features = test_graph.get_collection('out_feature')[0]

        batch_size = 16 # This is not important.
        n_channel = 3 # SIR.

        features_all = []
        ground_truths = []
        for (x_batch, y_batch, meta_batch) in gen_xy_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=False):
            x_batch_ = onehot(iteration2snapshot(x_batch, n_frame, start=meta_batch, end=end, random=0),n_channel)
            y_batch_ = snapshot_to_labels(y_batch, n_node)
            features_batch_ = test_sess.run(features, feed_dict={'data_input:0': x_batch_, 'data_label:0': y_batch_, 'keep_prob:0': 1.0})

            features_all.append(features_batch_)
            ground_truths.append(y_batch_)
        
        # Concatenate along the batch dimension (axis=0)
        features_all = np.concatenate(features_all, axis=0)
        features_all = features_all[:, 0, :, :]
        ground_truths = np.concatenate(ground_truths, axis=0)

    return features_all, ground_truths

def logistic_regression_nodewise(features, labels):
    """
    Apply logistic regression to each vertex to predict 0-1 labels.

    Args:
        features: np.ndarray, shape (n_sample, n_node, n_dim), float array of features.
        labels: np.ndarray, shape (n_sample, n_node), 0-1 array of labels.

    Returns:
        coefficients: np.ndarray, shape (n_node, n_dim), logistic regression coefficients for each vertex.
    """
    # Validate input shapes
    if features.shape[0] != labels.shape[0] or features.shape[1] != labels.shape[1]:
        raise ValueError(f"Shape mismatch: features {features.shape}, labels {labels.shape}")

    n_sample, n_node, n_dim = features.shape

    # Initialize coefficients array
    coefficients = np.zeros((n_node, n_dim))

    # Apply logistic regression to each vertex
    for i in range(n_node):
        # Extract features and labels for vertex i
        X = features[:, i, :]  # Shape: (n_sample, n_dim)
        y = labels[:, i]       # Shape: (n_sample,)

        # Check for single-label case
        unique_labels = np.unique(y)
        if len(unique_labels) < 2: # Note: this is almost impossible under a large training set, e.g., 20000
            # Single label (all 0s or all 1s): assign zero coefficients
            coefficients[i, :] = 0.0
            print(f"Node {i}: Only one label ({unique_labels[0]}), assigning zero coefficients")
        else:
            # Fit logistic regression
            model = LogisticRegression(solver='lbfgs', max_iter=1000)
            model.fit(X, y)
            coefficients[i, :] = model.coef_.flatten()  # Shape: (n_dim,)

    return coefficients

def logistic_regression_nodewise_online(data_file, model_file, save_file, train_pct, n_node, batch_size = 200, n_frame=16, val_pct=0, end=-1):
    """
    Performs online logistic regression node-wise on features extracted from a GNN model.
    Args:
        data_file: str. Path to the dataset.
        model_file: str. Path to the pre-trained model.
        save_file: str. Path to the file storing the results.
        train_pct: float. Training percentage.
        n_node: int. Number of nodes.
        n_frame: int. Number of frames (default: 16).
        val_pct: float. Validation percentage (default: 0).
        end: int. End index for snapshot (default: -1).
    
    Returns:
        coefficients: np.ndarray, shape (n_node, n_dim). Logistic regression coefficients for each node.
        intercepts: np.ndarray, shape (n_node,). Logistic regression intercepts for each node.
    """

    try:
        with open(save_file, 'rb') as f:
            res = pickle.load(f)
        coefficients = res['coefficients']
        intercepts = res['intercepts']
    except:

        # Load data
        inputs = data_gen(data_file, n_node, n_frame, train_pct, val_pct)

        # Load model
        model_path = tf.train.get_checkpoint_state(model_file).model_checkpoint_path

        test_graph = tf.Graph()

        with test_graph.as_default():
            saver = tf.compat.v1.train.import_meta_graph(pjoin(f'{model_path}.meta'))

        with tf.compat.v1.Session(graph=test_graph) as test_sess:
            saver.restore(test_sess, tf.train.latest_checkpoint(model_file))
            print(f'>> Loading saved model from {model_path} ...')

            features = test_graph.get_collection('out_feature')[0]

            n_channel = 3  # SIR.

            # Initialize one SGDClassifier and StandardScaler per node
            # classifiers = [SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01, warm_start=True) 
            #                for _ in range(n_node)]
            classifiers = [SGDClassifier(loss='log', learning_rate='optimal', eta0=0.01, warm_start=True) 
                            for _ in range(n_node)]
            scalers = [StandardScaler() for _ in range(n_node)]

            # Process batches
            batch_idx = 0
            for (x_batch, y_batch, meta_batch) in gen_xy_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=False):

                start_time = time.time()

                x_batch_ = onehot(iteration2snapshot(x_batch, n_frame, start=meta_batch, end=end, random=0), n_channel)
                y_batch_ = snapshot_to_labels(y_batch, n_node)
                features_batch_ = test_sess.run(features, feed_dict={'data_input:0': x_batch_, 
                                                                    'data_label:0': y_batch_, 
                                                                    'keep_prob:0': 1.0})

                features_batch_ = features_batch_[:, 0, :, :]
                # features_batch_ shape: (batch_size, n_node, n_dim)
                # y_batch_ shape: (batch_size, n_node)
                for node_idx in range(n_node):
                    # Extract features and labels for this node
                    X_node = features_batch_[:, node_idx, :]  # Shape: (batch_size, n_dim)
                    y_node = y_batch_[:, node_idx]  # Shape: (batch_size,)

                    # Standardize features for this node
                    X_node_scaled = scalers[node_idx].partial_fit(X_node).transform(X_node)

                    # Update logistic regression for this node
                    classifiers[node_idx].partial_fit(X_node_scaled, y_node, classes=[0, 1])

                print(str(batch_idx) + '-th batch finished, time cost:', time.time() - start_time)
                batch_idx = batch_idx + 1

            # Extract coefficients for each node
            n_dim = features_batch_.shape[2]  # Feature dimension
            coefficients = np.zeros((n_node, n_dim))
            intercepts = np.zeros(n_node)
            for node_idx in range(n_node):
                coefficients[node_idx, :] = classifiers[node_idx].coef_
                intercepts[node_idx] = classifiers[node_idx].intercept_[0]
            
            res = {'coefficients': coefficients, 'intercepts': intercepts}
            if save_file:
                with open(save_file + '/log_reg_res.pickle', 'wb') as f:
                    pickle.dump(res, f)

    return coefficients, intercepts