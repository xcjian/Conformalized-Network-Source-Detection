import pickle
import os
import glob

# Path configuration
prop_model = 'MixedSIR'
data_path = f"../data/{prop_model}/"
split_folder = os.path.join(data_path, 'split')

# Example base filename -- you can customize as needed
# This should match your split files (without the _1.pickle, _2.pickle etc.)
base_filename = 'MixedSIR_nsrc1-ratio0.3_nsrc7-ratio0.4_nsrc10-ratio-1_Rzero2.5_beta0.3_gamma0_T30_ls400_nf16_entire'

# Find all split files matching the pattern
split_files = sorted(glob.glob(os.path.join(split_folder, f"{base_filename}_*.pickle")))

if not split_files:
    print(f"No split files found matching pattern: {split_folder}{base_filename}_*.pickle")
    exit()

# Initialize empty lists to hold merged data
merged_X = []
merged_y = []

# Process each split file
for file_path in split_files:
    with open(file_path, 'rb') as f:
        chunk_data = pickle.load(f)
        merged_X.extend(chunk_data[0])
        merged_y.extend(chunk_data[1])
    print(f"Processed {os.path.basename(file_path)}")

# Combine into original format
merged_data = (merged_X, merged_y)

# Save the merged file
output_path = os.path.join(data_path, f"{base_filename}.pickle")
with open(output_path, 'wb') as f:
    pickle.dump(merged_data, f)

print(f"\nSuccessfully merged {len(split_files)} files into:")
print(output_path)
print(f"Total sequences: {len(merged_data[0])}")
print(f"Total labels: {len(merged_data[1])}")

# Optional: check if counts match (just print, no fixed expected count here)
if len(merged_data[0]) == len(merged_data[1]):
    print("Verification: Sequence and label counts match")
else:
    print(f"Warning: Mismatch - sequences: {len(merged_data[0])}, labels: {len(merged_data[1])}")
