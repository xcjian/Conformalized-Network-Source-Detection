"""
for mixedSIR the file name is too long so you need this to rename.
"""

import os
import re

# The directory where your split files are stored
split_dir = "../data/MixedSIR/split"

# Pattern to match the nsrc-ratio part
pattern = re.compile(r'_nsrc1-ratio0\.5_nsrc7-ratio0\.4_nsrc10-ratio-1')

# Iterate over files in the directory
for filename in os.listdir(split_dir):
    if pattern.search(filename):
        # Replace the matched part with '_SI_low'
        new_filename = pattern.sub('_SI_low', filename)
        
        old_path = os.path.join(split_dir, filename)
        new_path = os.path.join(split_dir, new_filename)
        
        # Rename the file
        os.rename(old_path, new_path)
        
        print(f"Renamed: {filename} -> {new_filename}")

print("Batch renaming completed.")
