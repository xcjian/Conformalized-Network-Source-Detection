import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import argparse

def load_results(graph, test_exp_name, beta_ran, alpha_ran):
    """
    Load results into 3D arrays (beta × alpha × repeats)
    
    Args:
        graph: Graph dataset name
        test_exp_name: Experimental configuration name
        beta_ran: List of beta values
        alpha_ran: List of alpha levels
    
    Returns:
        Dictionary of 3D numpy arrays (n_beta × n_alpha × n_repeats)
    """
    base_path = f"results/{graph}/{test_exp_name}"
    methods = ['ADiT_DSI', 'ArbiTree_CQC', 'set_prec', 'set_recall']
    all_alphas = [0.05, 0.07, 0.10, 0.15, 0.20]
    
    # Initialize 3D arrays for each method
    results = {
        method: np.full((len(beta_ran), len(alpha_ran), 50), np.nan)
        for method in methods
    }
    
    for beta_idx, beta in enumerate(beta_ran):
        pow_val = round(1 - beta, 1)
        pow_dir = f"pow_expected{pow_val}"
        
        for method in methods:
            for repeat in range(50):
                file_path = f"{base_path}/{pow_dir}/{method}_repeat{repeat}.pickle"
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                        # Fill alpha columns for this beta and repeat
                        for alpha_idx, alpha in enumerate(alpha_ran):
                            orig_alpha_idx = all_alphas.index(alpha)
                            results[method][beta_idx, alpha_idx, repeat] = \
                                data['set_size'][orig_alpha_idx]
    
    return results

def analyze_results(results, alpha_ran, beta_ran):
    """
    Calculate mean set sizes across repeats
    
    Args:
        results: Dictionary of 3D arrays
        alpha_ran: List of alpha levels
        beta_ran: List of beta values
    
    Returns:
        Dictionary of method statistics
    """
    method_stats = {}
    for method, data in results.items():
        # Calculate mean across repeats (shape: n_beta × n_alpha)
        means = np.nanmean(data, axis=2)
        method_stats[method] = {
            'data': data,  # Full 3D array
            'mean': means  # 2D mean array
        }
    return method_stats

def perform_statistical_tests(method_stats, alpha_ran, beta_ran):
    """
    Perform pairwise Wilcoxon tests for all method pairs and directions
    
    Args:
        method_stats: Dictionary of method statistics
        alpha_ran: List of alpha levels
        beta_ran: List of beta values
    
    Returns:
        DataFrame with comparison results
    """
    comparisons = []
    methods = sorted(method_stats.keys())
    
    # Generate all possible method pairs (4 methods → 6 pairs)
    method_pairs = [(m1, m2) for i, m1 in enumerate(methods) 
                   for j, m2 in enumerate(methods) if i < j]
    
    # For each pair, test both directions
    for m1, m2 in method_pairs:
        for beta_idx, beta in enumerate(beta_ran):
            for alpha_idx, alpha in enumerate(alpha_ran):
                # Get paired data (filtering out NaN repeats)
                m1_data = method_stats[m1]['data'][beta_idx, alpha_idx, :]
                m2_data = method_stats[m2]['data'][beta_idx, alpha_idx, :]
                
                # Remove NaN values (missing repeats)
                mask = ~(np.isnan(m1_data) | np.isnan(m2_data))
                paired_m1 = m1_data[mask]
                paired_m2 = m2_data[mask]
                
                if len(paired_m1) < 10:
                    continue
                
                # Test m1 < m2
                try:
                    stat, p_less = wilcoxon(
                        paired_m1,
                        paired_m2,
                        alternative='less'
                        #method='approx'
                    )
                    
                    comparisons.append({
                        'Comparison': f"{m1} < {m2}",
                        'Beta': beta,
                        'Alpha': alpha,
                        'p-value': p_less,
                        'n_pairs': len(paired_m1)
                    })
                except ValueError:
                    pass
                
                # Test m1 > m2
                try:
                    stat, p_greater = wilcoxon(
                        paired_m1,
                        paired_m2,
                        alternative='greater'
                        #method='approx'
                    )
                    
                    comparisons.append({
                        'Comparison': f"{m1} > {m2}",
                        'Beta': beta,
                        'Alpha': alpha,
                        'p-value': p_greater,
                        'n_pairs': len(paired_m1)
                    })
                except ValueError:
                    pass
    
    return pd.DataFrame(comparisons)

def main(graph='highSchool', 
         test_exp_name='SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16',
         beta_ran=[0.1, 0.3, 0.5, 0.7],
         alpha_ran=[0.05, 0.10, 0.15]):
    """
    Main analysis pipeline
    """
    # 1. Load data
    results = load_results(graph, test_exp_name, beta_ran, alpha_ran)
    
    # 2. Analyze data
    method_stats = analyze_results(results, alpha_ran, beta_ran)
    
    # 3. Perform statistical tests
    results_df = perform_statistical_tests(method_stats, alpha_ran, beta_ran)
    
    # 4. Save results
    output_dir = f"analysis_results/{graph}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{test_exp_name}_stats.csv"
    
    # Create pivot table
    pivot_df = results_df.pivot_table(
        index=['Comparison', 'Beta'],
        columns='Alpha',
        values='p-value',
        aggfunc='first'
    ).reset_index()
    
    # Format p-values
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
    
    for alpha in alpha_ran:
        pivot_df[alpha] = pivot_df[alpha].apply(format_pval)
    
    pivot_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    print("\nStatistical significance results:")
    print(pivot_df.to_markdown(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze experimental results with statistical tests')
    
    parser.add_argument('--graph', 
                       type=str, 
                       default='highSchool',
                       help='Graph dataset name')
    parser.add_argument('--test_exp_name', 
                       type=str, 
                       default='SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls8000_nf16',
                       help='Experimental configuration name')
    parser.add_argument('--beta_ran', 
                       nargs='+', 
                       type=float, 
                       default=[0.1, 0.3, 0.5, 0.7],
                       help='List of beta values')
    parser.add_argument('--alpha_ran', 
                       nargs='+', 
                       type=float, 
                       default=[0.05, 0.10, 0.15],
                       help='List of alpha levels')
    
    args = parser.parse_args()
    main(args.graph, args.test_exp_name, args.beta_ran, args.alpha_ran)