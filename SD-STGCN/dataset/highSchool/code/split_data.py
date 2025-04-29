import pickle
import os
import math

# This code is used to split the data such that it will not exceed the size of git commit.
prop_model = 'SIR'
Rzero = 43.44  # simulation R0
nsrc = 14
beta = 0.25   # beta
gamma = 0.15    # simulation gamma
ns = 21200    # num of sequences
nf = 16      # num of frames
gt = "highSchool"  # graph type
T = 30       # simulation time steps

data_path = f"../data/{prop_model}/"
file_name = f"{prop_model}_nsrc{nsrc}_Rzero{Rzero}_beta{beta}_gamma{gamma}_T{T}_ls{ns}_nf{nf}_entire.pickle"

with open(data_path + file_name, 'rb') as f:
    data = pickle.load(f)

# Calculate number of chunks needed (5000 data points per file)
chunk_size = 5000
total_data_points = len(data[0])  # Assuming data[0] contains the sequences
num_chunks = math.ceil(total_data_points / chunk_size)

# split the data into multiple files and store them
split_folder = f"../data/{prop_model}/split/"
if not os.path.exists(split_folder):
    os.makedirs(split_folder)

base_filename = file_name.replace('.pickle', '')

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, total_data_points)
    
    # Create chunk data
    chunk_data = (
        data[0][start_idx:end_idx],
        data[1][start_idx:end_idx]
    )
    
    # Save chunk
    chunk_filename = f"{base_filename}_{i+1}.pickle"
    with open(os.path.join(split_folder, chunk_filename), 'wb') as f:
        pickle.dump(chunk_data, f)
    print(f'Saved chunk {i+1}/{num_chunks} with {end_idx-start_idx} data points')

print('Splitting completed successfully!')