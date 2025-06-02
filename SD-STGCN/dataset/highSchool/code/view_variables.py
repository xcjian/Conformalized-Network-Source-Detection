import pickle
import os

gpath = os.path.dirname(os.getcwd())

# with open(gpath + '\data\SIR\split\SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_entire_1.pickle', 'rb') as f:
#     data = pickle.load(f)

with open(r'C:\py-codes\NTU-programs-2\source-detection\Conformalized-Network-Source-Detection\SD-STGCN\dataset\highSchool\data\SIR\split\SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_entire_1.pickle', 'rb') as f:
    data = pickle.load(f)


def explore_structure(obj, indent=0, is_root=True):
    """Explore the structure of a nested Python object with special handling for nested dicts."""
    prefix = '  ' * indent
    if isinstance(obj, dict):
        print(f"{prefix}dict with {len(obj)} keys")
        for k, v in obj.items():
            print(f"{prefix}- key: {repr(k)}")
            if isinstance(v, dict) and not is_root:
                # Special case: dict inside dict
                keys = sorted(v.keys())
                values = set(v.values())
                print(f"{prefix}  dict (summarized):")
                print(f"{prefix}    sorted keys: {keys}")
                print(f"{prefix}    unique values: {values}")
            else:
                # Recurse normally
                explore_structure(v, indent + 1, is_root=False)
    elif isinstance(obj, list):
        print(f"{prefix}list of length {len(obj)}")
        if len(obj) > 0:
            print(f"{prefix}- first element:")
            explore_structure(obj[0], indent + 1, is_root=False)
    elif isinstance(obj, tuple):
        print(f"{prefix}tuple of length {len(obj)}")
        for i, v in enumerate(obj):
            print(f"{prefix}- index {i}:")
            explore_structure(v, indent + 1, is_root=False)
    elif hasattr(obj, 'shape'):
        print(f"{prefix}{type(obj).__name__} with shape {obj.shape}")
    else:
        print(f"{prefix}{type(obj).__name__}: {repr(obj)[:80]}")


explore_structure(data)


# for i in range(6):
#     print(data[0][0][i]['node_count'])
# print(data[0][0])

# print('ok')

"""
The data loaded above takes the following form:
X: a list which contains observed propogation data for each iteration.
each element include the following elements:
'iteration': the current iteration number.
'status': indices of nodes in each status.
'node_count': the number of nodes in each status.
'status_delta': the change in the number of nodes in each status.
y: a list which contains the source node index for each iteration.


X_trainable: 21200 x [30 x N x 3]
y: 21200 x [list of sources]
"""

