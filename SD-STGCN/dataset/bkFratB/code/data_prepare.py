import os
import sys
import pickle
import re
import glob
import math
import numpy as np
import torch

# Add the data_loader directory to sys.path
data_loader_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data_loader'))
sys.path.append(data_loader_path)

from data_utils import iteration2snapshot, onehot, snapshot_to_labels

add_skip = False
# Add the number of skip for all dataset in SIR/split, because previous simulator does not have such information.
# This step is not needed.

merge_and_split = True
# merge and split data for all datasets.

convert_to_torch = False
# convert the data to the torch-accessible format.
# This step should only be executed if you have already done merge_and_split.

if add_skip:
    folder_path = '../data/SIR/split'

    # Function to extract nsrc from file name
    def get_nsrc(filename):
        # Use regex to find 'nsrc' followed by digits
        match = re.search(r'nsrc(\d+)', filename)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract nsrc from filename: {filename}")

    # Get all .pickle files in the folder
    pickle_files = glob.glob(os.path.join(folder_path, '*.pickle'))

    # Process each pickle file
    for file_path in pickle_files:
        # Extract nsrc from the file name
        try:
            nsrc = get_nsrc(os.path.basename(file_path))
        except:
            continue
        
        # Read the pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Validate data structure
        if not isinstance(data, (list, tuple)) or len(data) != 2:
            print(f"Warning: Invalid data structure in {file_path}, skipping...")
            continue
        
        # Get the length of the 1-th element
        try:
            len_1th = len(data[1])
        except TypeError as e:
            print(f"Error: 1-th element in {file_path} does not support len(), skipping...")
            continue
        
        # Create the 2-th element based on nsrc
        if nsrc == 1:
            new_element = [[2]] * len_1th
        elif nsrc in {7, 10, 14}:
            new_element = [[1]] * len_1th
        else:
            print(f"Warning: nsrc={nsrc} in {file_path} is not supported, skipping...")
            continue
        
        # Create new data structure with the 2-th element
        # Preserve original type (list or tuple)
        if isinstance(data, tuple):
            new_data = (*data, new_element)
        else:  # Assume list
            new_data = data + [new_element]
        
        # Save the modified data back to the file
        with open(file_path, 'wb') as f:
            pickle.dump(new_data, f)
        
        print(f"Updated {file_path} with new 2-th element of length {len_1th}")

    print("Processing complete.")


if merge_and_split:
    # Define the folders
    folders = ['../data/SIR/split']

    # Function to extract base name and chunk number from filename
    def get_base_name_and_chunk(filename):
        # Match base name and chunk number (e.g., _5.pickle)
        match = re.match(r'(.+)_(\d+)\.pickle$', filename)
        if match:
            return match.group(1), int(match.group(2))
        return filename.replace('.pickle', ''), None

    # Function to merge data from split files
    def merge_split_files(split_folder, parent_folder):
        # Ensure parent folder exists
        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder)
        
        # Get all .pickle files in split folder
        split_files = glob.glob(os.path.join(split_folder, '*.pickle'))
        if not split_files:
            print(f"No pickle files found in {split_folder}")
            return
        
        # Group files by base name
        file_groups = {}
        for file_path in split_files:
            filename = os.path.basename(file_path)
            base_name, chunk_num = get_base_name_and_chunk(filename)
            if chunk_num is not None:
                if base_name not in file_groups:
                    file_groups[base_name] = []
                file_groups[base_name].append((file_path, chunk_num))
        
        # Process each group
        for base_name, files in file_groups.items():
            # Check if merged file already exists
            merged_filename = f"{base_name}.pickle"
            merged_path = os.path.join(parent_folder, merged_filename)
            if os.path.exists(merged_path):
                print(f"Skipping merge for {base_name}: {merged_path} already exists")
                continue
            
            # Sort files by chunk number
            files.sort(key=lambda x: x[1])
            
            # Load and merge data
            data_0, data_1, data_2 = [], [], []
            for file_path, chunk_num in files:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                if not isinstance(data, (list, tuple)) or len(data) != 3:
                    print(f"Warning: Invalid data structure in {file_path}, skipping...")
                    continue
                data_0.append(data[0])
                data_1.append(data[1])
                data_2.append(data[2])
            
            if not data_0:
                print(f"No valid data to merge for base name {base_name}")
                continue
            
            # Concatenate data (handle lists or NumPy arrays)
            try:
                merged_data_0 = np.concatenate(data_0) if isinstance(data_0[0], np.ndarray) else sum(data_0, [])
                merged_data_1 = np.concatenate(data_1) if isinstance(data_1[0], np.ndarray) else sum(data_1, [])
                merged_data_2 = np.concatenate(data_2) if isinstance(data_2[0], np.ndarray) else sum(data_2, [])
            except ValueError as e:
                print(f"Error merging data for {base_name}: {e}")
                continue
        
            # Create merged data structure (preserve original type)
            merged_data = (merged_data_0, merged_data_1, merged_data_2) if isinstance(data, tuple) else [merged_data_0, merged_data_1, merged_data_2]
            
            # Save merged file
            with open(merged_path, 'wb') as f:
                pickle.dump(merged_data, f)
            print(f"Merged {len(files)} chunks into {merged_path}")

    # Function to split data into chunks
    def split_files(parent_folder, split_folder):
        # Ensure split folder exists
        if not os.path.exists(split_folder):
            os.makedirs(split_folder)
        
        # Get all .pickle files in parent folder
        parent_files = glob.glob(os.path.join(parent_folder, '*.pickle'))
        if not parent_files:
            print(f"No pickle files found in {parent_folder}")
            return
        
        chunk_size = 5000
        
        for file_path in parent_files:
            filename = os.path.basename(file_path)
            if filename.endswith('_entire.pickle'):  # Only process entire files
                # Check if any split files already exist
                base_filename = filename.replace('.pickle', '')
                existing_splits = glob.glob(os.path.join(split_folder, f"{base_filename}_[0-9]*.pickle"))
                if existing_splits:
                    print(f"Skipping split for {filename}: Split files already exist (e.g., {existing_splits[0]})")
                    continue
                
                # Load data
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                if not isinstance(data, (list, tuple)) or len(data) != 3:
                    print(f"Warning: Invalid data structure in {file_path}, skipping...")
                    continue
                
                # Get total data points
                try:
                    total_data_points = len(data[0])
                except TypeError as e:
                    print(f"Error: Cannot determine number of data points for {file_path}")
                    continue
                
                # Calculate number of chunks
                num_chunks = math.ceil(total_data_points / chunk_size)
                
                # Split data into chunks
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, total_data_points)
                    
                    # Create chunk data
                    chunk_data = (
                        data[0][start_idx:end_idx],
                        data[1][start_idx:end_idx],
                        data[2][start_idx:end_idx]
                    ) if isinstance(data, tuple) else [
                        data[0][start_idx:end_idx],
                        data[1][start_idx:end_idx],
                        data[2][start_idx:end_idx]
                    ]
                    
                    # Save chunk
                    chunk_filename = f"{base_filename}_{i+1}.pickle"
                    chunk_path = os.path.join(split_folder, chunk_filename)
                    with open(chunk_path, 'wb') as f:
                        pickle.dump(chunk_data, f)
                    print(f'Saved chunk {i+1}/{num_chunks} with {end_idx-start_idx} data points to {chunk_path}')

    # Main processing
    for split_folder in folders:
        # Get parent folder (e.g., ../data/SIR/ from ../data/SIR/split)
        parent_folder = os.path.dirname(split_folder)
        
        # Merge split files into parent folder
        print(f"\nMerging files in {split_folder} to {parent_folder}")
        merge_split_files(split_folder, parent_folder)
        
        # Split parent files into split folder
        print(f"\nSplitting files in {parent_folder} to {split_folder}")
        split_files(parent_folder, split_folder)

    print("\nMerge and split processing completed successfully!")


if convert_to_torch:

    n_channel = 3
    n_frame = 30

    # Define the folders
    folders = ['../data/SIR']

    # Function to process a single pickle file
    def process_pickle_file(file_path, save_path):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            

            ## maintain metadata:
            meta = data[2]
    
            ## convert x:
            x = data[0]
            n_sample = len(x)
    
            ## an example of converting one single sample (0-th sample)
            x_padded = []
            for x_tmp in x:
                pseudo_snapshot_ = x_tmp[0].copy() # duplicate the 0-th iteration to end up with a complete list of snapshots.
                x_tmp_ = x_tmp.copy() 
                x_tmp_.insert(0, pseudo_snapshot_)
                x_padded.append(x_tmp_)
            # x_onehot = onehot(iteration2snapshot(x_padded, n_frame, start=[[0]] * n_sample, end=-1, random=False), n_channel)
            x_onehot = iteration2snapshot(x_padded, n_frame, start=[[0]] * n_sample, end=-1, random=False)
            x_onehot = torch.from_numpy(x_onehot)
    
            ## convert y:
            y = data[1]
            # n_vertex = np.shape(x_onehot, 2)
            n_vertex = 774
            y_onehot = snapshot_to_labels(y, n_vertex)
            y_onehot = torch.from_numpy(y_onehot)

            # Save the processed data
            data_to_save = (x_onehot, y_onehot, meta)
            with open(save_path, 'wb') as f:
                pickle.dump(data_to_save, f)
            print(f"Successfully processed and saved: {save_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    # Iterate through folders and process files
    for folder in folders:
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue
        
        # List all files in the folder (exclude subdirectories)
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            
            # Check if it's a file and ends with '_entire.pickle'
            if os.path.isfile(file_path) and filename.endswith('_entire.pickle'):
                # Generate save path by replacing '_entire.pickle' with '_torchentire.pickle'
                save_filename = filename.replace('_entire.pickle', '_torchentire.pickle')
                save_path = os.path.join(folder, save_filename)
                
                # Check if save file already exists
                if os.path.exists(save_path):
                    print(f"Skipping: {file_path} (output {save_path} already exists)")
                else:
                    print(f"Processing: {file_path}")
                    process_pickle_file(file_path, save_path)
    
    