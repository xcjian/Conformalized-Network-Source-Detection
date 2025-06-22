import pickle
import os

gpath = os.path.dirname(os.getcwd())

# with open(gpath + '\data\SIR\split\SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_entire_1.pickle', 'rb') as f:
#     data = pickle.load(f)

with open(gpath + r'\data\MixedSIR\MixedSIR_nsrc1-ratio0.3_nsrc7-ratio0.4_nsrc10-ratio-1_Rzero2.5_beta0.3_gamma0_T30_ls400_nf16_entire.pickle', 'rb') as f:
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



def explore_structure_with_mixed_info(obj, indent=0, is_root=True, mixed_info=None):
    """
    Explore the structure of a nested Python object with intelligent dict compression 
    and summary of MixedSIR source counts.
    
    Parameters:
    - obj: the object to explore
    - indent: current indentation level
    - is_root: whether this is the root level
    - mixed_info: dictionary to track MixedSIR source count summary
    """
    prefix = '  ' * indent
    if mixed_info is None:
        mixed_info = {'source_count': {}}

    if isinstance(obj, dict):
        key_count = len(obj)
        print(f"{prefix}dict with {key_count} keys")
        if key_count == 0:
            return

        if key_count > 10:
            # Compress output for large dicts
            sample_keys = list(obj.keys())
            sample_values = list(obj.values())
            key_types = set(type(k).__name__ for k in sample_keys)
            value_types = set(type(v).__name__ for v in sample_values)
            print(f"{prefix}  key types: {key_types}")
            print(f"{prefix}  value types: {value_types}")
            
            # Attempt to show min/max of keys if they are numeric
            if all(isinstance(k, (int, float)) for k in sample_keys):
                try:
                    print(f"{prefix}  key min: {min(sample_keys)}, key max: {max(sample_keys)}")
                except:
                    pass

            # Collect a sample of unique values
            unique_values = set()
            for v in sample_values:
                try:
                    unique_values.add(v)
                except:
                    unique_values.add(str(v))
            uv_list = list(unique_values)
            print(f"{prefix}  unique values (sample): {uv_list[:5]}{'...' if len(uv_list) > 5 else ''}")

            # Explore the first key's value as an example
            first_key = sample_keys[0]
            print(f"{prefix}  first key-value sample:")
            explore_structure_with_mixed_info(obj[first_key], indent + 1, is_root=False, mixed_info=mixed_info)

        else:
            # Print each key-value pair for small dicts
            for k, v in obj.items():
                print(f"{prefix}- key: {repr(k)}")
                explore_structure_with_mixed_info(v, indent + 1, is_root=False, mixed_info=mixed_info)

    elif isinstance(obj, list):
        print(f"{prefix}list of length {len(obj)}")
        if len(obj) > 0:
            print(f"{prefix}- first element:")
            explore_structure_with_mixed_info(obj[0], indent + 1, is_root=False, mixed_info=mixed_info)

    elif isinstance(obj, tuple):
        print(f"{prefix}tuple of length {len(obj)}")
        for i, v in enumerate(obj):
            print(f"{prefix}- index {i}:")
            explore_structure_with_mixed_info(v, indent + 1, is_root=False, mixed_info=mixed_info)

        # If this is a (X, y) tuple, summarize source count in y
        if len(obj) == 2:
            y_obj = obj[1]
            if isinstance(y_obj, list):
                for item in y_obj:
                    if isinstance(item, list):
                        src_count = len(item)
                    else:
                        src_count = 1
                    mixed_info['source_count'][src_count] = mixed_info['source_count'].get(src_count, 0) + 1

                print(f"{prefix}MixedSIR source count summary:")
                for src_count, count in sorted(mixed_info['source_count'].items()):
                    print(f"{prefix}- source count {src_count}: {count} samples")

    elif hasattr(obj, 'shape'):
        # Handle numpy arrays or similar objects
        print(f"{prefix}{type(obj).__name__} with shape {obj.shape}")

    else:
        # Handle primitive or other types
        print(f"{prefix}{type(obj).__name__}: {repr(obj)[:80]}")



# explore_structure(data)
explore_structure_with_mixed_info(data)  #use for mixedSIR

# Extract y from data
y = data[1]

# Take the first 20 samples
first_20_y = y[:20]

# Count source count for each sample
for i, item in enumerate(first_20_y):
    if isinstance(item, list):
        src_count = len(item)
    else:
        src_count = 1  # single int source
    print(f"Sample {i}: source count = {src_count}")


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

