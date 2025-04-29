import networkx as nx
import numpy as np
import itertools

from collections import defaultdict

def fit_model(Y, S, edges, num_iters=10, lr=1.0):
    """Newton solver with exact gradient and Hessian."""
    n_samples, K = Y.shape
    alpha = {k: np.zeros(2) for k in range(K)}
    beta = {(k, l): np.zeros(4) for (k, l) in edges}
    
    for it in range(num_iters):
        grad_alpha, grad_beta = compute_gradient(alpha, beta, Y, S, edges)
        H = compute_hessian(alpha, beta, Y, S, edges)
        
        # Flatten gradient
        grad = []
        for k in sorted(alpha.keys()):
            grad.append(grad_alpha[k])
        for (k, l) in sorted(edges):
            grad.append(grad_beta[(k, l)])
        grad = np.concatenate(grad)
        
        # Newton step
        delta = np.linalg.solve(H + 1e-6*np.eye(H.shape[0]), -grad)
        
        # Update parameters
        param_vec = []
        for k in sorted(alpha.keys()):
            param_vec.append(alpha[k])
        for (k,l) in sorted(edges):
            param_vec.append(beta[(k,l)])
        param_vec = np.concatenate(param_vec)
        
        new_param_vec = param_vec + lr * delta
        alpha, beta = unflatten_params(new_param_vec, K, edges)
        
        obj = compute_function_value(alpha, beta, Y, S, edges)
        print(f"Iter {it}: Objective = {obj:.4f}")
    
    return alpha, beta

def compute_gradient(alpha, beta, Y, S, edges):
    """Compute gradient with expectation subtracted."""
    n_samples, K = Y.shape
    grad_alpha = {k: np.zeros(2) for k in range(K)}
    grad_beta = {(k, l): np.zeros(4) for (k, l) in edges}
    
    for i in range(n_samples):
        # Get model expectations
        E_phi, E_psi = compute_model_expectations(alpha, beta, S[i], edges)
        
        # Observed terms
        for k in range(K):
            grad_alpha[k] += phi_func(Y[i, k], S[i, k])
        for (k, l) in edges:
            grad_beta[(k, l)] += psi_func(Y[i, l], Y[i, k])

        # Subtract expected terms
        for k in range(K):
            grad_alpha[k] -= E_phi[k]
        for (k, l) in edges:
            grad_beta[(k, l)] -= E_psi[(k,l)]
    
    return grad_alpha, grad_beta

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
        # map (Y_l, Y_k): (-1,-1), (1,-1), (-1,1), (1,1) → index 0,1,2,3
        E_psi[(k,l)][0] = probs[0,0]  # Y_l = -1, Y_k = -1
        E_psi[(k,l)][1] = probs[1,0]  # Y_l = 1,  Y_k = -1
        E_psi[(k,l)][2] = probs[0,1]  # Y_l = -1, Y_k = 1
        E_psi[(k,l)][3] = probs[1,1]  # Y_l = 1,  Y_k = 1
        
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

def unflatten_params(vec, K, edges):
    """ Unflatten parameter vector into alpha, beta dictionaries. """
    alpha = {}
    beta = {}
    idx = 0
    for k in range(K):
        alpha[k] = vec[idx:idx+2]
        idx += 2
    for (k, l) in edges:
        beta[(k, l)] = vec[idx:idx+4]
        idx += 4
    return alpha, beta


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