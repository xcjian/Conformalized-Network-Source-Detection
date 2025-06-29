import pickle
import os
from pathlib import Path

split = True
# For all experimental results in the folder, if they exceed 90MB, then split them into chunks. each chunk include 400 samples.
# If the chunks already exist then skip.

merge = True
# Merge the result chunk into res.pickle.
# If the merged file already exist then skip.

# Create directory if it doesn't exist
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Get all folders in current working directory
exp_folders = [f for f in os.listdir() if os.path.isdir(f) and not f.startswith('.')]

for exp_name in exp_folders:
    if split:
        res_file = f'{exp_name}/res.pickle'
        split_dir = f'{exp_name}/split'
        
        # Check if file exists and get its size in MB
        if os.path.exists(res_file):
            file_size_mb = os.path.getsize(res_file) / (1024 * 1024)
            
            # Only split if file size > 90MB
            if file_size_mb > 90:
                # Check if split files already exist
                if not os.path.exists(split_dir) or not any('res_' in f for f in os.listdir(split_dir)):
                    try:
                        with open(res_file, 'rb') as f:
                            data = pickle.load(f)

                        data_pred = data['predictions']
                        data_ground_truth = data['ground_truth']
                        data_inputs = data['inputs']
                        
                        # Ensure split directory exists
                        ensure_dir(split_dir)
                        
                        # Calculate number of chunks (400 samples per chunk)
                        chunk_size = 400
                        num_chunks = (len(data_pred) + chunk_size - 1) // chunk_size
                        
                        # Split data into chunks
                        for i in range(num_chunks):
                            start_idx = i * chunk_size
                            end_idx = min((i + 1) * chunk_size, len(data_pred))
                            
                            # Create chunk dictionary
                            chunk_data = {
                                'predictions': data_pred[start_idx:end_idx],
                                'ground_truth': data_ground_truth[start_idx:end_idx],
                                'inputs': data_inputs[start_idx:end_idx]
                            }
                            
                            # Save chunk
                            chunk_file = f'{split_dir}/res_{i+1}.pickle'
                            with open(chunk_file, 'wb') as f:
                                pickle.dump(chunk_data, f)
                            print(f'Created chunk for {exp_name}: {chunk_file}')
                    except Exception as e:
                        print(f'Error processing split for {exp_name}: {str(e)}')

    if merge:
        merged_file = f'{exp_name}/res.pickle'
        split_dir = f'{exp_name}/split'
        
        # Only merge if merged file doesn't exist
        if not os.path.exists(merged_file) and os.path.exists(split_dir):
            try:
                merged_data = {
                    'predictions': [],
                    'ground_truth': [],
                    'inputs': []
                }
                
                # Get all chunk files
                chunk_files = sorted([f for f in os.listdir(split_dir) if f.startswith('res_') and f.endswith('.pickle')],
                                  key=lambda x: int(x.split('_')[1].split('.')[0]))
                
                # Merge chunks
                for chunk_file in chunk_files:
                    with open(f'{split_dir}/{chunk_file}', 'rb') as f:
                        chunk_data = pickle.load(f)
                        merged_data['predictions'].extend(chunk_data['predictions'])
                        merged_data['ground_truth'].extend(chunk_data['ground_truth'])
                        merged_data['inputs'].extend(chunk_data['inputs'])
                
                # Save merged data
                with open(merged_file, 'wb') as f:
                    pickle.dump(merged_data, f)
                print(f'Created merged file for {exp_name}: {merged_file}')
            except Exception as e:
                print(f'Error processing merge for {exp_name}: {str(e)}')

print('ok')