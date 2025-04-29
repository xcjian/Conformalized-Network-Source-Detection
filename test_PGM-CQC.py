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

alpha = {
    0: np.array([-0.5, 0.5]),   
    1: np.array([0.25, -0.25]),    
    2: np.array([-1, 1]),    
}
beta = {
    (0, 1): np.array([1.0, 1.0, 2, 0.5]),  # Edge 0-1
    (1, 2): np.array([2, 1, -1, 1]),  # Edge 1-2
}
s = np.array([0.5, 1, 2])
edges = [(0, 1), (1, 2)]


p_single, p_pair = compute_model_marginals(alpha, beta, s, edges)



print('ok')