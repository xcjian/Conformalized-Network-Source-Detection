import numpy as np
from itertools import product
import random

from utils.score_convert import *
from utils.functions import *

# Set random seeds for reproducibility
RANDOM_SEED = 101  # You can use any integer value
random.seed(RANDOM_SEED)  # For Python's random module
np.random.seed(RANDOM_SEED)  # For NumPy's random number generator

# define a tree with parameters

# edges = [(0, 1), (0, 2), (1, 5), (1, 6), (2, 3), (2, 4)]

# alpha = {
#     0: 1,
#     1: 0.5,
#     2: 2,
#     3: 0.25,
#     4: 1,
#     5: 3,
#     6: 1
# }

# beta = {
#     (0, 1): 1,
#     (0, 2): 0.5,
#     (1, 5): 0.5,
#     (1, 6): -1,
#     (2, 3): -1,
#     (2, 4): 1
# }

# n_node = len(alpha)

# # assumed estimations
# Y_hat = np.array([1, -0.5, 0.5, 2, 0.25, -1, -0.5])



def generate_random_tree(n_node):
    """Generate a random connected tree with n_node nodes."""
    if n_node == 1:
        return []
    
    edges = []
    nodes = list(range(n_node))
    random.shuffle(nodes)
    
    for i in range(1, n_node):
        parent = random.choice(nodes[:i])
        edges.append((parent, nodes[i]))
    
    return edges


# Generate random connected tree
n_node = 7  # You can change this to any number of nodes
edges = generate_random_tree(n_node)

# Generate random alpha parameters
alpha = {node: random.uniform(0.1, 3) for node in range(n_node)}

# Generate random beta parameters for each edge
beta = {}
for u, v in edges:
    beta[(u, v)] = random.uniform(-1, 1)
    beta[(v, u)] = beta[(u, v)]  # Assuming undirected edges

# Generate random assumed estimations
Y_hat = np.array([random.uniform(-2, 2) for _ in range(n_node)])


# 1. Generate all possible combinations of {-1, 1}^n_node
power_set = list(product([-1, 1], repeat=n_node))

# 2. Compute scores for all combinations
score_all = []
for combo in power_set:
    Y_ = np.array(combo)
    score_ = ArbiTreescore(Y_, Y_hat, edges, alpha, beta)
    score_all.append(score_)

# 3. Find maximum scores and corresponding combinations for each label (brute-force method)
maxscore_gt = np.zeros(n_node)
max_combo_gt = [None] * n_node  # To store the combinations that attain max scores

for k in range(n_node):
    max_score = -np.inf
    best_combo = None
    for i, combo in enumerate(power_set):
        if combo[k] == 1:  # Only consider sets where y_k = 1
            if score_all[i] > max_score:
                max_score = score_all[i]
                best_combo = combo
    maxscore_gt[k] = max_score
    max_combo_gt[k] = best_combo

# 4. Print results
print("Brute-force results:")
for k in range(n_node):
    print(f"Node {k}: Max score = {maxscore_gt[k]:.2f}, Attained by {max_combo_gt[k]}")

# compute via the mp algorithm alternatively

maxscore_mp = MPmaxscore(Y_hat, edges, alpha, beta)

# 6. Compare results
print("Brute-force maximum scores:", maxscore_gt)
print("Message passing maximum scores:", maxscore_mp)
print("Difference:", maxscore_gt - maxscore_mp)