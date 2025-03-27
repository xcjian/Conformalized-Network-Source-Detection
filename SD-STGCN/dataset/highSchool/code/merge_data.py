import pickle
import os
import glob

# Configuration (should match your splitting script)
prop_model = 'SIR'
Rzero = 2.5    # simulation R0
beta = 0.3     # beta
gamma = 0      # simulation gamma
ns = 21200     # num of sequences
nf = 16        # num of frames
T = 30         # simulation time steps

# Path configuration
data_path = f"../data/{prop_model}/"
split_folder = f"../data/{prop_model}/split/"
base_filename = f"{prop_model}_Rzero{Rzero}_beta{beta}_gamma{gamma}_T{T}_ls{ns}_nf{nf}_entire"

# Find all split files matching the pattern
split_files = sorted(glob.glob(os.path.join(split_folder, f"{base_filename}_*.pickle")))

if not split_files:
    print(f"No split files found matching pattern: {split_folder}{base_filename}_*.pickle")
    exit()

# Initialize empty lists to hold merged data
merged_data_0 = []
merged_data_1 = []

# Process each split file
for file_path in split_files:
    with open(file_path, 'rb') as f:
        chunk_data = pickle.load(f)
        merged_data_0.extend(chunk_data[0])
        merged_data_1.extend(chunk_data[1])
    print(f"Processed {os.path.basename(file_path)}")

# Combine into original format
merged_data = (merged_data_0, merged_data_1)

# Save the merged file
output_path = os.path.join(data_path, f"{base_filename}.pickle")
with open(output_path, 'wb') as f:
    pickle.dump(merged_data, f)

print(f"\nSuccessfully merged {len(split_files)} files into:")
print(output_path)
print(f"Total sequences: {len(merged_data[0])}")
print(f"Total labels: {len(merged_data[1])}")

# Verification
if len(merged_data[0]) == ns and len(merged_data[1]) == ns:
    print("Verification: Data count matches original specification")
else:
    print(f"Warning: Data count ({len(merged_data[0])}) doesn't match expected ({ns})")