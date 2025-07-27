import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import argparse

def load_time_data(graph='highSchool', 
                  exp_name='SIR_nsrc1_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16',
                  beta=0.0):
    """
    Load time cost data for all methods and repeats
    
    Args:
        graph: Graph dataset name
        exp_name: Experimental configuration name
        beta: Beta value (1-beta is the power)
    
    Returns:
        Dictionary of time costs (method: list of times)
    """
    base_path = f"results/{graph}/{exp_name}"
    pow_val = round(1 - beta, 1)
    pow_dir = f"pow_expected{pow_val}"
    full_path = f"{base_path}/{pow_dir}"
    
    # Initialize dictionary to store time costs
    time_data = {}
    
    # Check what methods are available
    available_methods = set()
    for f in os.listdir(full_path):
        if f.endswith('.pickle') and '_repeat' in f:
            method = f.split('_repeat')[0]
            available_methods.add(method)
    
    # Initialize data structure for each available method
    for method in available_methods:
        time_data[method] = []
    
    # Load data for each repeat
    for method in available_methods:
        for repeat in range(50):  # Assuming up to 50 repeats
            file_path = f"{full_path}/{method}_repeat{repeat}.pickle"
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                        time_data[method].append(data['time_cost'])
                except:
                    continue
    
    return time_data

def perform_time_comparisons(time_data):
    """
    Perform pairwise Wilcoxon tests for all method pairs and directions
    
    Args:
        time_data: Dictionary of time costs (method: list of times)
    
    Returns:
        DataFrame with comparison results
    """
    comparisons = []
    methods = sorted(time_data.keys())
    
    # Generate all possible method pairs
    method_pairs = [(m1, m2) for i, m1 in enumerate(methods) 
                   for j, m2 in enumerate(methods) if i < j]
    
    for m1, m2 in method_pairs:
        # Get paired time data
        m1_times = np.array(time_data[m1])
        m2_times = np.array(time_data[m2])
        
        # Find common repeats (where both methods have data)
        min_len = min(len(m1_times), len(m2_times))
        paired_m1 = m1_times[:min_len]
        paired_m2 = m2_times[:min_len]
        
        if len(paired_m1) < 10:  # Minimum sample size requirement
            continue
        
        # Test m1 < m2 (m1 is faster than m2)
        try:
            stat, p_less = wilcoxon(
                paired_m1,
                paired_m2,
                alternative='less'
                #method='approx'
            )
            comparisons.append({
                'Comparison': f"{m1} faster than {m2}",
                'p-value': p_less,
                'n_pairs': len(paired_m1),
                'mean_diff': np.mean(paired_m2 - paired_m1),
                'm1_mean': np.mean(paired_m1),
                'm2_mean': np.mean(paired_m2)
            })
        except ValueError:
            pass
        
        # Test m1 > m2 (m1 is slower than m2)
        try:
            stat, p_greater = wilcoxon(
                paired_m1,
                paired_m2,
                alternative='greater'
                #method='approx'
            )
            comparisons.append({
                'Comparison': f"{m1} slower than {m2}",
                'p-value': p_greater,
                'n_pairs': len(paired_m1),
                'mean_diff': np.mean(paired_m1 - paired_m2),
                'm1_mean': np.mean(paired_m1),
                'm2_mean': np.mean(paired_m2)
            })
        except ValueError:
            pass
    
    return pd.DataFrame(comparisons)

def save_results(results_df, graph='highSchool', 
                exp_name='SIR_nsrc1_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16',
                beta=0.0):
    """
    Save comparison results to CSV file
    
    Args:
        results_df: DataFrame with comparison results
        graph: Graph dataset name
        exp_name: Experimental configuration name
        beta: Beta value used
    """
    output_dir = f"time_comparison_results/{graph}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Format p-values with significance stars
    def format_pval(p):
        if pd.isna(p):
            return "NA"
        if p < 0.001:
            return f"{p:.2e}***"
        elif p < 0.01:
            return f"{p:.2e}**"
        elif p < 0.05:
            return f"{p:.2e}*"
        return f"{p:.3f}"
    
    results_df['formatted_p'] = results_df['p-value'].apply(format_pval)
    
    # Create output filename
    pow_val = round(1 - beta, 1)
    output_file = f"{output_dir}/{exp_name}_pow{pow_val}_time_comparison.csv"
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    print("\nTime comparison results:")
    print(results_df[['Comparison', 'n_pairs', 'mean_diff', 'formatted_p']].to_markdown(index=False))

def main(graph='highSchool', 
        exp_name='SIR_nsrc1_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16',
        beta=0.0):
    """
    Main analysis pipeline for time comparisons
    """
    # 1. Load time data
    time_data = load_time_data(graph, exp_name, beta)
    
    # 2. Perform statistical tests
    results_df = perform_time_comparisons(time_data)
    
    # 3. Save results
    save_results(results_df, graph, exp_name, beta)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compare time costs of different methods using Wilcoxon signed rank tests')
    
    parser.add_argument('--graph', 
                       type=str, 
                       default='highSchool',
                       help='Graph dataset name')
    parser.add_argument('--exp_name', 
                       type=str, 
                       default='SIR_nsrc1_Rzero2.5_beta0.25_gamma0_T30_ls8000_nf16',
                       help='Experimental configuration name')
    parser.add_argument('--beta', 
                       type=float, 
                       default=0.0,
                       help='Beta value (1-beta is the power)')
    
    args = parser.parse_args()
    main(args.graph, args.exp_name, args.beta)