import pickle
import os
import math

# Configuration: point to your mixed data file
prop_model = 'MixedSIR'
# Example file name - you can modify as needed
file_name = 'MixedSIR_nsrc1-ratio0.5_nsrc7-ratio0.4_nsrc10-ratio-1_Rzero2.5_beta0.3_gamma0_T30_ls400_nf16_entire.pickle'

data_path = f"../data/{prop_model}/"
full_path = os.path.join(data_path, file_name)

# Load data
with open(full_path, 'rb') as f:
    data = pickle.load(f)

# Determine splitting parameters
chunk_size = 50  # max data points per split
total_data_points = len(data[0])  # data[0] holds X
num_chunks = math.ceil(total_data_points / chunk_size)

# Prepare split output folder
split_folder = os.path.join(data_path, 'split')
if not os.path.exists(split_folder):
    os.makedirs(split_folder)

# Generate base filename without extension
base_filename = file_name.replace('.pickle', '')

# Split and save chunks
for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, total_data_points)

    chunk_data = (
        data[0][start_idx:end_idx],
        data[1][start_idx:end_idx]
    )

    chunk_filename = f"{base_filename}_{i+1}.pickle"
    chunk_path = os.path.join(split_folder, chunk_filename)

    with open(chunk_path, 'wb') as f:
        pickle.dump(chunk_data, f)

    print(f"Saved chunk {i+1}/{num_chunks} with {end_idx - start_idx} data points")

print("Splitting completed successfully! Please go to file_rename.py to shorten the file name.")
