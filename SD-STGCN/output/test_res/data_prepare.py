import os
import pickle
import math
import argparse
from pathlib import Path

def get_file_size(file_path):
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)

def split_pickle_file(config_path, pickle_file, max_size_mb=50):
    """Split large pickle file into smaller chunks"""
    split_dir = os.path.join(config_path, 'split')
    # Check if split files already exist
    if os.path.exists(split_dir) and any(f.startswith('res_') and f.endswith('.pickle') for f in os.listdir(split_dir)):
        return
    
    # Read original pickle file
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    # Calculate number of splits
    file_size_mb = get_file_size(pickle_file)
    num_splits = math.ceil(file_size_mb / max_size_mb)
    if num_splits <= 1:
        return
        
    # Create split directory if it doesn't exist
    os.makedirs(split_dir, exist_ok=True)
    
    # Calculate items per split
    total_items = len(data['predictions'])
    items_per_split = math.ceil(total_items / num_splits)
    
    # Split data
    for i in range(num_splits):
        start_idx = i * items_per_split
        end_idx = min((i + 1) * items_per_split, total_items)
        
        split_data = {
            'predictions': data['predictions'][start_idx:end_idx],
            'ground_truth': data['ground_truth'][start_idx:end_idx],
            'inputs': data['inputs'][start_idx:end_idx],
            'logits': data['logits'][start_idx:end_idx]
        }
        
        # Save split file
        split_file = os.path.join(split_dir, f'res_{i+1}.pickle')
        with open(split_file, 'wb') as f:
            pickle.dump(split_data, f)

def combine_pickle_files(config_path):
    """Combine split pickle files into a single res.pickle"""
    split_dir = os.path.join(config_path, 'split')
    output_file = os.path.join(config_path, 'res.pickle')
    
    # Skip if res.pickle already exists or split dir doesn't exist/is empty
    if os.path.exists(output_file) or not os.path.exists(split_dir):
        return
    split_files = [f for f in os.listdir(split_dir) if f.startswith('res_') and f.endswith('.pickle')]
    if not split_files:
        return
    
    # Initialize combined data
    combined_data = {
        'predictions': [],
        'ground_truth': [],
        'inputs': [],
        'logits': []
    }
    
    # Sort split files by number
    split_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # Combine data from all split files
    for split_file in split_files:
        with open(os.path.join(split_dir, split_file), 'rb') as f:
            split_data = pickle.load(f)
            for key in combined_data:
                combined_data[key].extend(split_data[key])
    
    # Save combined data
    with open(output_file, 'wb') as f:
        pickle.dump(combined_data, f)

def process_datasets(root_dir, do_split, do_combine):
    """Process all datasets and configurations"""
    # Iterate through datasets
    for dataset in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
            
        # Iterate through configurations
        for config in os.listdir(dataset_path):
            config_path = os.path.join(dataset_path, config)
            if not os.path.isdir(config_path):
                continue
                
            pickle_file = os.path.join(config_path, 'res.pickle')
            
            if do_split and os.path.exists(pickle_file):
                # Split functionality
                if get_file_size(pickle_file) > 50:
                    split_pickle_file(config_path, pickle_file)
            
            if do_combine and not os.path.exists(pickle_file):
                # Combine functionality
                combine_pickle_files(config_path)

def main():
    parser = argparse.ArgumentParser(description='Process pickle files: split or combine')
    parser.add_argument('--split', type=int, default=1, help='1 to perform split operation')
    parser.add_argument('--combine', type=int, default=0, help='1 to perform combine operation')
    args = parser.parse_args()
    
    # Get current directory as root
    root_dir = os.getcwd()
    
    # Process datasets based on arguments
    process_datasets(root_dir, bool(args.split), bool(args.combine))

if __name__ == '__main__':
    main()