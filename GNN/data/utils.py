import os, pickle
import torch


def convert_timeseries_to_features(data, nf=16, normalize=False):
    """
    Convert the time series data from the 'data' tuple into feature vectors for all nodes,
    ensuring the output format is [21200, 3, 774].

    Args:
        data (tuple): The dataset tuple containing:
                      - data[0]: Time series (shape: [21200, 30, 774]).
                      - data[2]: Skip values (shape: [21200, 1]).
        nf (int): The number of snapshots to consider (default is 16).
        normalize (bool): Whether to normalize the output values by dividing by nf.

    Returns:
        torch.Tensor: A PyTorch tensor of shape [21200, 3, 774], where:
                      - 21200: Number of simulations.
                      - 3: Feature vector dimensions [t0, t1, t2].
                      - 774: Number of nodes per simulation.
    """
    time_series = data[0]  # Shape: [21200, 30, 774]
    skip_values = data[2]  # Shape: [21200, 1]

    # Initialize an empty tensor to store features
    num_simulations, num_snapshots, num_nodes = time_series.shape
    node_features = torch.zeros((num_simulations, 3, num_nodes), dtype=torch.float32)

    for sim_idx in range(num_simulations):
        skip = skip_values[sim_idx][0]  # Get the skip value for this simulation
        sliced_series = time_series[sim_idx, skip:skip + nf, :]  # Slice the time series based on skip and nf

        # Count occurrences of 0, 1, and 2 across the sliced series
        t0 = (sliced_series == 0).sum(dim=0)  # Count 0s for each node
        t1 = (sliced_series == 1).sum(dim=0)  # Count 1s for each node
        t2 = (sliced_series == 2).sum(dim=0)  # Count 2s for each node

        # Normalize if required
        if normalize:
            t0 = t0 / nf
            t1 = t1 / nf
            t2 = t2 / nf

        # Store the feature vectors in the output tensor
        node_features[sim_idx, 0, :] = t0
        node_features[sim_idx, 1, :] = t1
        node_features[sim_idx, 2, :] = t2

    return node_features

def create_and_save_new_data_with_node_features():
    
    data_folder = '/home/twp/lantian/Conformalized-Network-Source-Detection/SD-STGCN/dataset/highSchool/data/SIR'
    data_file = 'SIR_nsrc1_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16_torchentire.pickle'

    data_path = os.path.join(data_folder, data_file)
    # Load the data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)


    node_features = convert_timeseries_to_features(data, normalize=True)
    # Replace data[0] with node_features and save as a new pickle file
    new_data = (node_features, data[1], data[2])

    # Define new file name
    new_data_file = data_file.replace('.pickle', '_with_node_features.pickle')
    new_data_path = os.path.join(data_folder, new_data_file)

    # Save the new data tuple
    with open(new_data_path, 'wb') as f_out:
        pickle.dump(new_data, f_out)

    print(f"Saved new data with node features to {new_data_path}")