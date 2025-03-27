import os
import argparse
import subprocess
import itertools
import json

def extract_validation_accuracy(filepath):
    """
    Extract validation accuracy from the output text.
    Assumes the validation accuracy is the first number in the line after "Test results saved to".
    If the line is not found, return -1.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    # Find the line containing "Test results saved to"
    for i, line in enumerate(lines):
        if "Test results saved to" in line:
            # The next line contains the validation accuracy
            if i + 1 < len(lines):
                numbers = lines[i + 1].strip().split()
                if numbers:
                    return float(numbers[0])
    return -1

def main():
    # Define default parameters
    Rzero = 2.5  # simulation R0
    beta = 0.3   # beta
    gamma = 0    # simulation gamma
    ns = 21200    # num of sequences
    nf = 16      # num of frames
    N = 774      # num of nodes in graph
    gt = "highSchool"  # graph type
    ep = 50       # num of epochs
    save = 1     # save every # of epochs
    skip = 1     # start from skip-th snapshot
    end = -1     # the sampled snapshots will end at (skip + n_frame)-th snapshot
    T = 30       # simulation time steps
    random = 0   # randomly sample n_frame snapshots?
    run_on_cloud = 0 # run on cloud platform? if yes, the data will be read from the cloud.
    train_pct = 0.9434
    val_pct = 0.0189

    # Define parameter ranges for grid search
    batch_sizes = [16]  # batch_size
    learning_rates = [1e-3]  # learning rate
    spatio_kernel_sizes = [4]  # spatio kernel size
    temporal_kernel_sizes = [1]  # temporal kernel size

    # Parse command-line arguments (optional overrides)
    parser = argparse.ArgumentParser(description="Run SIR model with specified parameters.")
    parser.add_argument("--Rzero", type=float, default=Rzero, help="Simulation R0 (default: 2.5)")
    parser.add_argument("--beta", type=float, default=beta, help="Beta (default: 0.1)")
    parser.add_argument("--gamma", type=float, default=gamma, help="Gamma (default: 0.4)")
    parser.add_argument("--ns", type=int, default=ns, help="Number of sequences (default: 4000)")
    parser.add_argument("--nf", type=int, default=nf, help="Number of frames (default: 16)")
    parser.add_argument("--ep", type=int, default=ep, help="Number of epochs (default: 3)")
    parser.add_argument("--save", type=int, default=save, help="Save every # of epochs (default: 1)")
    parser.add_argument("--skip", type=int, default=skip, help="Start from skip-th snapshot (default: 1)")
    parser.add_argument("--random", type=int, default=random, help="Randomly sample n_frame snapshots (default: 0)")
    args = parser.parse_args()

    # Update parameters with command-line arguments
    Rzero = args.Rzero
    beta = args.beta
    gamma = args.gamma
    ns = args.ns
    nf = args.nf
    ep = args.ep
    save = args.save
    skip = args.skip
    random = args.random

    # Construct paths
    local_graph_path = f"./dataset/{gt}/data/graph/{gt}.edgelist"
    local_seq_path = f"./dataset/{gt}/data/SIR/SIR_Rzero{Rzero}_beta{beta}_gamma{gamma}_T{T}_ls{ns}_nf{nf}_entire.pickle"
    if run_on_cloud:
        graph_path = f"/home/featurize/data/CP_source_det/dataset/{gt}/data/graph/{gt}.edgelist"
        seq_path = f"/home/featurize/data/CP_source_det/dataset/{gt}/data/SIR/SIR_Rzero{Rzero}_beta{beta}_gamma{gamma}_T{T}_ls{ns}_nf{nf}_entire.pickle"
        if not os.path.exists(local_graph_path):
            os.makedirs(os.path.dirname(local_graph_path), exist_ok=True)
            subprocess.run(["cp", graph_path, local_graph_path])
        if not os.path.exists(local_seq_path):
            os.makedirs(os.path.dirname(local_seq_path), exist_ok=True)
            subprocess.run(["cp", seq_path, local_seq_path])
    else:
        graph_path = local_graph_path
        seq_path = local_seq_path
    
    pred_path = f"./output/models/{gt}/pred_{gt}_nf{nf}.pickle"
    exp_name = f"SIR_Rzero{Rzero}_beta{beta}_gamma{gamma}_T{T}_ls{ns}_nf{nf}"

    # Directory to store results
    results_dir = "./para_tune_res/" + exp_name
    os.makedirs(results_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Track the best parameter combination and smallest validation error
    best_validation_accuracy = -float('inf')  # Initialize with a large value
    best_params = None

    # Grid search over all parameter combinations
    for bs, lr, ks, kt in itertools.product(batch_sizes, learning_rates, spatio_kernel_sizes, temporal_kernel_sizes):
        print(f"Testing parameters: batch_size={bs}, lr={lr}, ks={ks}, kt={kt}")

        # Construct the filename for this parameter combination
        filename = f"params_bs{bs}_lr{lr}_ks{ks}_kt{kt}.txt"
        filepath = os.path.join(results_dir, filename)

        # Check if the result file already exists
        if os.path.exists(filepath):
            print(f"Result file found: {filepath}")
            validation_accuracy = extract_validation_accuracy(filepath)
        else:
            # Construct the command to run main_SIR_T.py
            command = [
                "python", "main_SIR_T.py",
                "--gt", gt,
                "--n_node", str(N),
                "--n_frame", str(nf),
                "--batch_size", str(bs),
                "--epoch", str(ep),
                "--ks", str(ks),
                "--kt", str(kt),
                "--sconv", "gcn",
                "--save", str(save),
                "--graph", graph_path,
                "--pred", pred_path,
                "--seq", seq_path,
                "--exp_name", exp_name,
                "--start", str(skip),
                "--end", str(end),
                "--random", str(random),
                "--lr", str(lr),
                "--valid", "1",
            ]

            # Print the command for debugging
            print("Running command:", " ".join(command))

            # Run the command and capture the output
            result = subprocess.run(command, capture_output=True, text=True)

            # Write the parameter combination and output to a .txt file
            with open(filepath, "w") as f:
                f.write(f"Parameters: batch_size={bs}, lr={lr}, ks={ks}, kt={kt}\n\n")
                f.write("Output:\n")
                f.write(result.stdout)  # Write the standard output
                if result.stderr:
                    f.write("\nErrors:\n")
                    f.write(result.stderr)  # Write the standard error

            print(f"Saved results to {filepath}")
            validation_accuracy = extract_validation_accuracy(filepath)

        # Update the best parameters if this combination is better
        if validation_accuracy != -1 and validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_params = {
                "batch_size": bs,
                "learning_rate": lr,
                "spatio_kernel_size": ks,
                "temporal_kernel_size": kt,
                "validation_accuracy": validation_accuracy,
            }

        print(f"Validation accuracy: {validation_accuracy}")

    # Print the best parameter combination and smallest validation error
    print("\nBest parameter combination:")
    print(json.dumps(best_params, indent=4))

if __name__ == "__main__":
    main()