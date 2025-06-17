import numpy as np
from utils.functions import *

# alpha = {
#     0: np.array([1.0, 2.0]),   # Root node
#     1: np.array([0.5, 1.5]),    # Left child of 0
#     2: np.array([2.0, 0.5]),    # Right child of 0
#     3: np.array([1.0, 1.0]),    # Left child of 1
#     4: np.array([0.0, 2.0])     # Right child of 1
# }
# beta = {
#     (0, 1): np.array([1.0, 1.0, 0.5, 0.5]),  # Edge 0-1
#     (0, 2): np.array([0.8, 0.2, 0.3, 0.7]),  # Edge 0-2
#     (1, 3): np.array([1.5, 0.5, 0.5, 1.5]),  # Edge 1-3
#     (1, 4): np.array([0.5, 1.5, 1.5, 0.5])   # Edge 1-4
# }
# s = np.array([0.5, 0.25, 0.2, 0.3, 0.1])
# edges = [(0, 1), (0, 2), (1, 3), (1, 4)]

# alpha = {
#     0: np.array([1.0, 2.0]),   # Root node
#     1: np.array([0.5, 1.5]),    # Left child of 0
#     2: np.array([2.0, 0.5]),    # Right child of 0
#     3: np.array([1.0, 1.0])    # Left child of 1
# }
# beta = {
#     (0, 1): np.array([1.0, 1.0, 0.5, 0.5]),  # Edge 0-1
#     (0, 2): np.array([0.8, 0.2, 0.3, 0.7]),  # Edge 0-2
#     (1, 3): np.array([1.5, 0.5, 0.5, 1.5])  # Edge 1-3
# }
# s = np.array([0.5, 0.25, 0.2, 0.3])
# edges = [(0, 1), (0, 2), (1, 3)]

# alpha = {
#     0: np.array([-0.5, 0.5]),   
#     1: np.array([0.25, -0.25]),    
#     2: np.array([-1, 1]),    
# }
# beta = {
#     (0, 1): np.array([1.0, 1.0, 2, 0.5]),  # Edge 0-1
#     (1, 2): np.array([2, 1, -1, 1]),  # Edge 1-2
# }
# s = np.array([0.5, 1, 2])
# edges = [(0, 1), (1, 2)]


# test the optimizer over a single-edge tree.

alpha = {
    0: np.array([-0.5, 0.5]),   
    1: np.array([0.25, -0.25]),    
}
beta = {
    (0, 1): np.array([1.0, 0.5, -2, 0.5])  # Edge 0-1
}
s = np.array([0.5, 1])
edges = [(0, 1)]
n_labels = len(s)

p_single, p_pair = compute_model_marginals(alpha, beta, s, edges)

# generate Y:
n_samples = 1000
S = np.zeros((n_samples, n_labels))
for k in range(n_labels):
    S[:, k] = s[k]


"""
Generate samples from a 2x2 probability matrix for binary variables in {-1, 1}^2.

Args:
    p_pair: 2x2 matrix where p_pair[i,j] = P(Y1=val_i, Y2=val_j) 
            with val_i, val_j in {-1, 1}.
    n_samples: Number of samples to generate.

Returns:
    Y: Array of shape (n_samples, 2) with values in {-1, 1}.
"""
# Flatten the probability matrix and define possible outcomes
outcomes = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])  # All possible (Y1, Y2) combinations
p_flat = p_pair[(0,1)].flatten()  # Reshape to 1D probability vector

# Randomly choose outcomes based on probabilities
indices = np.random.choice(len(outcomes), size=n_samples, p=p_flat)
Y = outcomes[indices]

alpha_hat, beta_hat = fit_model(Y, S, edges, num_iters=1000, lr=10 ** (-3))
print("alpha:", alpha, "alpha_hat:", alpha_hat)
print("beta:", beta, "beta_hat:", beta_hat)

p_single_hat, p_pair_hat = compute_model_marginals(alpha_hat, beta_hat, s, edges)

print("p_single:", p_single, "p_pair:", p_pair)
print("p_single_hat:", p_single_hat, "p_pair_hat:", p_pair_hat)

print('ok')