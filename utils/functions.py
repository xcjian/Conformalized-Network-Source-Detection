import networkx as nx
import numpy as np
import itertools

from collections import defaultdict

def MPmaxscore(Y_hat, edges, alpha, beta):
    """
    This function efficiently computes the set S_+ in eq.(18) in the paper
    Cauchois, M., Gupta, S., & Duchi, J. C. (2021). Knowing what you know: valid and validated confidence sets in multiclass and multilabel prediction. Journal of machine learning research, 22(81), 1-42.
    i.e., for each k in [K], compute max{s(y): y_k = 1}.

    Arguments:
    edges, alpha, beta: Tree structure and parameters learned by ArbiTree.
    Y_hat: (n_labels) array, with {-1, 1} or real-valued elements.

    Returns:
    maxscores: (n_labels) array.
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

    # 4. Downward pass (root to leaves)
    down_msgs = {}
    stack = [(root, None)]  # (node, parent)
    visited = set()

    while stack:
        node, parent = stack.pop(0)
        visited.add(node)

        # Compute message from parent to node (downward)
        if parent is not None:
            msg = np.zeros(2)
            for y_tilde in [0, 1]:  # y_tilde: 0=-1, 1=1
                max_val = -np.inf
                for y_parent in [0, 1]:
                    term = phi_vals[parent][y_parent]
                    term += psi_vals[(parent, node)][y_parent, y_tilde]
                    for grand_parent in neighbors[parent]:
                        if (grand_parent, parent) in down_msgs:
                            term += down_msgs[(grand_parent, parent)][y_parent]
                    max_val = max(max_val, term)
                msg[y_tilde] = max_val
            down_msgs[(parent, node)] = msg
        
        # Push children to stack
        for child in neighbors[node]:
            if child != parent and child not in visited:
                stack.append((child, node))

    # 5. Upward pass (leaves to root)
    up_msgs = {}
    # Initialize with leaves (nodes with only one neighbor except root)
    queue = []
    for k in range(K):
        if len(neighbors[k]) == 1 and k != root:
            queue.append(k)
    
    while queue:
        node = queue.pop(0)
        parent = neighbors[node][0]  # Only one neighbor for leaves
        
        # Compute message from node to parent (upward)
        msg = np.zeros(2)
        for y_parent in [0, 1]:
            max_val = -np.inf
            for y_node in [0, 1]:
                term = phi_vals[node][y_node]
                term += psi_vals[(node, parent)][y_node, y_parent]
                # Add messages from node's children
                for child in neighbors[node]:
                    if child != parent and (child, node) in up_msgs:
                        term += up_msgs[(child, node)][y_node]
                max_val = max(max_val, term)
            msg[y_parent] = max_val
        up_msgs[(node, parent)] = msg
        
        # Add parent to queue if all children have sent messages
        all_children_done = True
        for child in neighbors[parent]:
            if child != node and (child, parent) not in up_msgs:
                all_children_done = False
                break
        if all_children_done and parent != root:
            queue.append(parent)

    # 6. Compute max scores for each y_k=1
    maxscores = np.zeros(K)
    for k in range(K):
        # Start with node potential for y_k=1
        total = phi_vals[k][1]
        
        # Add all incoming messages
        for neighbor in neighbors[k]:
            if (neighbor, k) in down_msgs:
                total += down_msgs[(neighbor, k)][1]
            if (neighbor, k) in up_msgs:
                total += up_msgs[(neighbor, k)][1]
        
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