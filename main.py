import os
import sys
import networkx as nx
import pickle
import numpy as np
import time
from matplotlib import pyplot as plt
from utils.score_convert import *
from DSI.src.diffusion_source.infection_model import FixedTSI
from DSI.src.diffusion_source.discrepancies import ADiT_h
import argparse
import pandas as pd

np.random.seed(41)
# Set parameters

parser = argparse.ArgumentParser()

parser.add_argument('--graph', type=str, default='highSchool')
parser.add_argument('--exp_name', type=str, default='SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16') 
# the name of the experiment. e.g., SIR_Rzero${Rzero}_beta${beta}_gamma${gamma}_T${T}_ls${ns}_nf${nf}
parser.add_argument('--pow_expected', type=float, default=0.5)
parser.add_argument('--calib_ratio', type=float, default=0.8)
parser.add_argument('--prop_model', type=str, default='SIR')
parser.add_argument('--confi_levels', 
                    nargs='*',  # Accepts 0 or more values
                    type=float, 
                    default=[0.02, 0.05, 0.07, 0.10],  # Default list
                    help='confi_levels')
parser.add_argument('--set_recall', type=int, default=1) # 0 means this method will not be used.
parser.add_argument('--set_prec', type=int, default=1)
parser.add_argument('--ADiT_DSI', type=int, default=0)
parser.add_argument('--PGM_CQC', type=int, default=0)
parser.add_argument('--ArbiTree_CQC', type=int, default=0)
parser.add_argument('--mc_runs', type=int, default=3)

# parameters for different methods

# ADiT-DSI
parser.add_argument('--m_l', type=int, default=5)
parser.add_argument('--m_p', type=int, default=5)

# PGM-CQC
parser.add_argument('--n_learn_tree', type=int, default=200)

args = parser.parse_args()

graph = args.graph
prop_model = args.prop_model

## Public parameters
confi_levels = args.confi_levels
n_alpha = len(confi_levels)
mc_runs = args.mc_runs

## Parameters for Conformal Prediction
set_recall = args.set_recall
set_prec = args.set_prec
ADiT_DSI = args.ADiT_DSI
PGM_CQC = args.PGM_CQC
ArbiTree_CQC = args.ArbiTree_CQC
calib_ratio = args.calib_ratio
pow_expected = args.pow_expected
start_freq = 0
end_freq = 774

## Parameters for ADiT-DSI
discrepancies = [ADiT_h] # discrepancy function
discrepancy_str = 'ADiT_h'
m_l = args.m_l
m_p = args.m_p

## Parameters for PGM-CQC
n_learn_tree = args.n_learn_tree

# Load data
# graph_path = 'SD-STGCN/dataset/highSchool/data/graph/highSchool.edgelist'
# data_path = 'SD-STGCN/output/test_res/highSchool/exp1/res.pickle'

exp_name = args.exp_name
graph_extract_path = 'SD-STGCN/dataset/' + graph + '/data/graph/' + graph + '.edgelist'
data_extract_path = 'SD-STGCN/output/test_res/' + graph + '/' + exp_name + '/res.pickle'
save_path = 'results/' + graph + '/' + exp_name + '/pow_expected' + str(pow_expected)

# copy these data to the data folder, if not copied
graph_path = 'data/' + graph + '/graph/' + graph + '.edgelist'
data_path = 'data/' + graph + '/test_res/' + exp_name + '/res.pickle'
if not os.path.exists(graph_path):
  os.makedirs(os.path.dirname(graph_path), exist_ok=True)
os.system('cp ' + graph_extract_path + ' ' + graph_path)
if not os.path.exists(data_path):
  os.makedirs(os.path.dirname(data_path), exist_ok=True)
os.system('cp ' + data_extract_path + ' ' + data_path)
os.makedirs(save_path, exist_ok=True)

graph = nx.read_edgelist(graph_path, nodetype=int)
G = nx.Graph(graph)
G.graph = G
n_nodes = G.number_of_nodes()
print('Number of nodes:', G.number_of_nodes())
print('Number of edges:', G.number_of_edges())

# Compute the Laplacian matrix (D - A)
L = nx.laplacian_matrix(G).astype(float)  # Returns a SciPy sparse matrix
L = L.toarray()  # Convert to dense NumPy array if needed
Gfreq, Gfb = np.linalg.eigh(L)

# define high pass filter
freq_response = np.zeros(n_nodes)
freq_response[start_freq: end_freq] = 1
score_filter = Gfb @ np.diag(freq_response) @ Gfb.T


with open(data_path, 'rb') as f:
    data = pickle.load(f)

inputs_raw = data['inputs'] # n_batch x n_sample x n_nodes
pred_scores_raw = data['predictions']
ground_truths_raw = data['ground_truth']
# logits_raw = data['logits']

# Prepare the data
## unzip the data
inputs = []
pred_scores = []
ground_truths = []
logits = []

for i in range(len(pred_scores_raw)):
  for j in range(len(pred_scores_raw[i])):
    inputs.append(inputs_raw[i][j])
    pred_scores.append(pred_scores_raw[i][j])
    ground_truths.append(ground_truths_raw[i][j])
    # logits.append(logits_raw[i][j])

## partition the data
n_samples = len(pred_scores)
n_calibration = int(n_samples * calib_ratio)
n_test = n_samples - n_calibration

method_names = []
coverage_res = {}
set_size_res = {}

for mc_idx in range(mc_runs):

  index_file_name_ = '/calib_index_repeat' + str(mc_idx) + '.npy'
  try:
    calib_index = np.load(save_path + index_file_name_)
  except:
    calib_index = np.random.choice(n_samples, n_calibration, replace=False)
    np.save(save_path + index_file_name_, calib_index)
  
  test_index = np.setdiff1d(np.arange(n_samples), calib_index)
  print('Number of samples:', n_samples, 'Calibration size:', n_calibration, 'test size:', n_test)

  inputs_calib = [inputs[i] for i in calib_index]
  pred_scores_calib = [pred_scores[i] for i in calib_index]
  ground_truths_calib = [ground_truths[i] for i in calib_index]
  # logits_calib = [logits[i] for i in calib_index]

  inputs_test = [inputs[i] for i in test_index]
  pred_scores_test = [pred_scores[i] for i in test_index]
  ground_truths_test = [ground_truths[i] for i in test_index]
  # logits_test = [logits[i] for i in test_index]


  ### To compare, comute the average size of infected set.
  infected_num = 0
  for i in range(n_test):
    infected_nodes_ = np.nonzero(inputs_test[i])[0]
    # infected_nodes_ = np.nonzero(inputs_test[i][0, :])[0]
    infected_num = infected_num + len(infected_nodes_)
  avg_infected_num = infected_num / n_test
  print('average infected size:', avg_infected_num)

  if set_recall:

    # try to load data.
    file_name_ = '/set_recall_repeat' + str(mc_idx) + '.pickle'
    try:
      with open(save_path + file_name_, 'rb') as f:
        res_load = pickle.load(f)
      coverage = res_load['cover']
      avg_size = res_load['set_size']
    except:
      # Conformal Prediction
      print('computing proposed method with recall score...')
      ## Compute conformity scores on the calibration set
      cfscore_calib = []
      for i in range(n_calibration):
        infected_nodes_ = np.nonzero(inputs_calib[i])[0]
        pred_prob_ = pred_scores_calib[i][:, 1]
        ground_truth_one_hot_ = ground_truths_calib[i]
        ground_truth_part_one_hot_ = set_truncate(ground_truth_one_hot_, pred_prob_, pow_expected)
        score_ = recall_score(pred_prob_, ground_truth_part_one_hot_, prop_model, infected_nodes_)
        cfscore_calib.append(score_)
      cfscore_calib = np.array(cfscore_calib)

      ## Compute conformity scores on the test set
      cfscore_test = []
      for i in range(n_test):
        infected_nodes_ = np.nonzero(inputs_test[i])[0]
        cfscore_ = recall_score_gtunknown(pred_scores_test[i][:, 1], prop_model, infected_nodes_)
        cfscore_test.append(cfscore_)
      cfscore_test = np.array(cfscore_test)
      print('Conformity scores computed. shape of test:' + str(cfscore_test.shape))

      
      ## Compute prediction sets and evaluate the performance

      coverage = np.zeros(n_alpha)
      avg_size = np.zeros(n_alpha)

      for i in range(n_test):
        pred_sets = {}
        for alpha in confi_levels:
          pred_sets[str(alpha)] = set()

          ## compute the quantile
          tail_prop = (1 - alpha) * (1 + 1 / n_calibration)
          threshold = np.quantile(cfscore_calib, tail_prop)

          for j in range(n_nodes):
            if cfscore_test[i][j] <= threshold:
              pred_sets[str(alpha)].add(j)
        #print('Prediction sets computed. index:', i)

        for j, alpha in enumerate(confi_levels):
          ground_truth_source = np.nonzero(ground_truths_test[i])[0]
          predicted_set = list(pred_sets[str(alpha)])

          # Compute intersection between ground truth and predicted set
          intersection_ = np.intersect1d(ground_truth_source, predicted_set)

          # Calculate power as the ratio of correctly detected true signals
          power = len(intersection_) / len(ground_truth_source) if len(ground_truth_source) > 0 else 0.0

          # Create power flag based on comparison with expected power
          power_flag = power >= pow_expected
          if power_flag:
            coverage[j] = coverage[j] + 1
          
          avg_size[j] = avg_size[j] + len(predicted_set)    
      
      coverage = coverage / n_test
      avg_size = avg_size / n_test

      # save the results
      res_ = {'cover': coverage, 'set_size': avg_size}
      with open(save_path + '/' + file_name_, 'wb') as f:
        pickle.dump(res_, f)
    
    print('coverage:', coverage)
    print('set size:', avg_size)

    if 'set_recall' not in method_names:
      method_names.append('set_recall')
    if 'set_recall' not in coverage_res:
      coverage_res['set_recall'] = np.zeros((mc_runs, n_alpha))
      set_size_res['set_recall'] = np.zeros((mc_runs, n_alpha))
    coverage_res['set_recall'][mc_idx, :] = coverage
    set_size_res['set_recall'][mc_idx, :] = avg_size
  
  if set_prec:

    # try to load data.
    file_name_ = '/set_prec_repeat' + str(mc_idx) + '.pickle'
    try:
      with open(save_path + file_name_, 'rb') as f:
        res_load = pickle.load(f)
      coverage = res_load['cover']
      avg_size = res_load['set_size']
    except:
      # Conformal Prediction
      print('computing proposed method with precision score...')
      ## Compute conformity scores on the calibration set
      cfscore_calib = []
      for i in range(n_calibration):
        infected_nodes_ = np.nonzero(inputs_calib[i])[0]
        pred_prob_ = pred_scores_calib[i][:, 1]
        ground_truth_one_hot_ = ground_truths_calib[i]
        ground_truth_part_one_hot_ = set_truncate(ground_truth_one_hot_, pred_prob_, pow_expected)
        score_ = avg_score(pred_prob_, ground_truth_part_one_hot_, prop_model, infected_nodes_)
        cfscore_calib.append(score_)
      cfscore_calib = np.array(cfscore_calib)

      ## Compute conformity scores on the test set
      cfscore_test = []
      for i in range(n_test):
        infected_nodes_ = np.nonzero(inputs_test[i])[0]
        cfscore_ = avg_score_gtunknown(pred_scores_test[i][:, 1], prop_model, infected_nodes_)
        cfscore_test.append(cfscore_)
      cfscore_test = np.array(cfscore_test)
      print('Conformity scores computed. shape of test:' + str(cfscore_test.shape))

      
      ## Compute prediction sets and evaluate the performance

      coverage = np.zeros(n_alpha)
      avg_size = np.zeros(n_alpha)

      for i in range(n_test):
        pred_sets = {}
        for alpha in confi_levels:
          pred_sets[str(alpha)] = set()

          ## compute the quantile
          tail_prop = (1 - alpha) * (1 + 1 / n_calibration)
          threshold = np.quantile(cfscore_calib, tail_prop)

          for j in range(n_nodes):
            if cfscore_test[i][j] <= threshold:
              pred_sets[str(alpha)].add(j)
        # print('Prediction sets computed. index:', i)

        for j, alpha in enumerate(confi_levels):
          ground_truth_source = np.nonzero(ground_truths_test[i])[0]
          predicted_set = list(pred_sets[str(alpha)])

          # Compute intersection between ground truth and predicted set
          intersection_ = np.intersect1d(ground_truth_source, predicted_set)

          # Calculate power as the ratio of correctly detected true signals
          power = len(intersection_) / len(ground_truth_source) if len(ground_truth_source) > 0 else 0.0

          # Create power flag based on comparison with expected power
          power_flag = power >= pow_expected
          if power_flag:
            coverage[j] = coverage[j] + 1
          
          avg_size[j] = avg_size[j] + len(predicted_set)    
      
      coverage = coverage / n_test
      avg_size = avg_size / n_test

      # save the results
      res_ = {'cover': coverage, 'set_size': avg_size}
      with open(save_path + '/' + file_name_, 'wb') as f:
        pickle.dump(res_, f)

    print('coverage:', coverage)
    print('set size:', avg_size)

    if 'set_prec' not in method_names:
      method_names.append('set_prec')
    if 'set_prec' not in coverage_res:
      coverage_res['set_prec'] = np.zeros((mc_runs, n_alpha))
      set_size_res['set_prec'] = np.zeros((mc_runs, n_alpha))
    coverage_res['set_prec'][mc_idx, :] = coverage
    set_size_res['set_prec'][mc_idx, :] = avg_size

  if ADiT_DSI:
    # ADiT-DSI

    # try to load data.
    file_name_ = '/ADiT_DSI_repeat' + str(mc_idx) + '.pickle'
    try:
      with open(save_path + file_name_, 'rb') as f:
        res_load = pickle.load(f)
      coverage = res_load['cover']
      avg_size = res_load['set_size']
    except:
      print('computing ADiT-DSI...')
      if prop_model != 'SI':
        print('SIR model is not supported by ADiT-DSI. Skipping...')
      else:

        coverage = np.zeros(n_alpha)
        avg_size = np.zeros(n_alpha)

        for i in range(n_test):
          infected_nodes_ = np.nonzero(inputs_test[i])[0]

          model = FixedTSI(G, discrepancies, canonical=True, expectation_after=False, m_l=m_l, m_p=m_p, T=len(infected_nodes_) - 1)

          start_time = time.time()
          confidence_sets = model.confidence_set_mp(infected_nodes_, confi_levels, new_run=True) 
          confidence_sets = confidence_sets[discrepancy_str]
          print('Time:', time.time() - start_time, 'index:', i)

          for j, alpha in enumerate(confi_levels):
            if ground_truths_test[i] in confidence_sets[str(alpha)]:
              coverage[j] = coverage[j] + 1
            
            avg_size[j] = avg_size[j] + len(confidence_sets[str(alpha)])        
        
        coverage = coverage / n_test
        avg_size = avg_size / n_test

        # save the results
        res_ = {'cover': coverage, 'set_size': avg_size}
        with open(save_path + '/' + file_name_, 'wb') as f:
          pickle.dump(res_, f)

        print('coverage:', coverage)
        print('set size:', avg_size)

        if 'ADiT_DSI' not in method_names:
          method_names.append('ADiT_DSI')
        if 'ADiT_DSI' not in coverage_res:
          coverage_res['ADiT_DSI'] = np.zeros((mc_runs, n_alpha))
          set_size_res['ADiT_DSI'] = np.zeros((mc_runs, n_alpha))
        coverage_res['ADiT_DSI'][mc_idx, :] = coverage
        set_size_res['ADiT_DSI'][mc_idx, :] = avg_size  

  if PGM_CQC:

    # Remark: This method require more complex partition of dataset.
    # For example, besides the calibration set, this algorithm need another hold-out set for learning the tree structure.
    # At current step we first simply hold out a part of calibration set for it to learn the tree.

    # Conformal Prediction
    print('computing PGM-CQC...')

    # compute the score function for each vertex (i.e., label) on the calibration set
    nodewise_score_calib = []
    Y_calib = []
    for i in range(n_calibration):
      infected_nodes_ = np.nonzero(inputs_calib[i])[0]
      # infected_nodes_ = np.nonzero(inputs_calib[i][0, :])[0]
      pred_prob_binary_ = pred_scores_calib[i]
      pred_prob_ = pred_prob_binary_[:, 1]
      ground_truth_one_hot_ = ground_truths_calib[i]
      ground_truth_part_one_hot_ = set_truncate(ground_truth_one_hot_, pred_prob_, pow_expected)
      logits_ = logits_calib[i]
      
      # score_ = nodewise_APS_score(pred_prob_binary_, ground_truth_part_one_hot_, infected_nodes_, prop_model)
      score_ = logits_[:, 1]
      nodewise_score_calib.append(score_)
      Y_calib.append(2 * (ground_truth_part_one_hot_ - 1/2))
    nodewise_score_calib = np.array(nodewise_score_calib)
    Y_calib = np.array(Y_calib)

    # learn the set scoring function from a set of scores
    Y_learn_tree = Y_calib[:n_learn_tree, :]
    score_learn_tree = nodewise_score_calib[:n_learn_tree, :]
    tree_edges, tree_alpha, tree_beta = PGMTree(Y_learn_tree, score_learn_tree, from_graph=True, G = G)

    # Compute conformity scores on the calibration set

    # Construct prediction set on the test set

  if ArbiTree_CQC:

    # Remark: This method require more complex partition of dataset.
    # For example, besides the calibration set, this algorithm need another hold-out set for learning the tree structure.
    # At current step we first simply hold out a part of calibration set for it to learn the tree.

    # try to load data.
    file_name_ = '/ArbiTree_CQC_repeat' + str(mc_idx) + '.pickle'
    try:
      with open(save_path + file_name_, 'rb') as f:
        res_load = pickle.load(f)
      coverage = res_load['cover']
      avg_size = res_load['set_size']
    except:
      # Conformal Prediction
      print('computing ArbiTree-CQC...')

      # compute Y_hat for each vertex (i.e., label) on the calibration set
      Y_calib = []
      Y_hat_calib = []
      for i in range(n_calibration):
        infected_nodes_ = np.nonzero(inputs_calib[i])[0]
        # infected_nodes_ = np.nonzero(inputs_calib[i][0, :])[0]
        pred_prob_ = pred_scores_calib[i][:, 1]
        # if prop_model == 'SI':
        #   non_infected_nodes = np.setdiff1d(np.arange(n_nodes), infected_nodes_)
        #   pred_prob_[non_infected_nodes] = 0

        ground_truth_one_hot_ = ground_truths_calib[i]
        ground_truth_part_one_hot_ = set_truncate(ground_truth_one_hot_, pred_prob_, pow_expected)
        
        Y_hat_ = (pred_prob_ - 1/2) * 2 # align with {-1, 1} Y values.
        Y_hat_calib.append(Y_hat_)
        Y_calib.append(2 * (ground_truth_part_one_hot_ - 1/2))
      Y_calib = np.array(Y_calib)
      Y_hat_calib = np.array(Y_hat_calib)

      # learn the set scoring function from a set of scores
      Y_learn_tree = Y_calib[:n_learn_tree, :]
      Y_hat_learn_tree = Y_hat_calib[:n_learn_tree, :]
      tree_edges, tree_alpha, tree_beta = ArbiTree(Y_learn_tree, Y_hat_learn_tree)

      # Compute conformity scores on the calibration set
      cfscore_calib = []
      for i in range(n_learn_tree, n_calibration):
        Y_ = Y_calib[i, :]
        Y_hat_ = Y_hat_calib[i, :]

        score_ = ArbiTreescore(Y_, Y_hat_, tree_edges, tree_alpha, tree_beta)
        cfscore_calib.append(score_)
      cfscore_calib = np.array(cfscore_calib)

      # Compute conformity scores on the test set
      cfscore_test = []
      for i in range(n_test):
        infected_nodes_ = np.nonzero(inputs_test[i])[0]
        pred_prob_ = pred_scores_test[i][:, 1]
        # if prop_model == 'SI':
        #   non_infected_nodes = np.setdiff1d(np.arange(n_nodes), infected_nodes_)
        #   pred_prob_[non_infected_nodes] = 0
        Y_hat_ = (pred_prob_ - 1/2) * 2 # align with {-1, 1} Y values.

        cfscore_ = MPmaxscore(Y_hat_, tree_edges, tree_alpha, tree_beta)
        cfscore_test.append(cfscore_)
      cfscore_test = np.array(cfscore_test)
      
      # Construct prediction set on the test set
      coverage = np.zeros(n_alpha)
      avg_size = np.zeros(n_alpha)

      for i in range(n_test):
        pred_sets = {}
        for alpha in confi_levels:
          pred_sets[str(alpha)] = set()

          ## compute the quantile
          tail_prop = 1 - (1 - alpha) * (1 + 1 / (n_calibration - n_learn_tree))
          threshold = np.quantile(cfscore_calib, tail_prop)

          for j in range(n_nodes):
            if cfscore_test[i][j] >= threshold:
              pred_sets[str(alpha)].add(j)
        print('Prediction sets computed. index:', i)

        # evaluate the performance
        for j, alpha in enumerate(confi_levels):
          ground_truth_source = np.nonzero(ground_truths_test[i])[0]
          predicted_set = list(pred_sets[str(alpha)])

          # Compute intersection between ground truth and predicted set
          intersection_ = np.intersect1d(ground_truth_source, predicted_set)

          # Calculate power as the ratio of correctly detected true signals
          power = len(intersection_) / len(ground_truth_source) if len(ground_truth_source) > 0 else 0.0

          # Create power flag based on comparison with expected power
          power_flag = power >= pow_expected
          if power_flag:
            coverage[j] = coverage[j] + 1
          
          avg_size[j] = avg_size[j] + len(predicted_set)    
        
      coverage = coverage / n_test
      avg_size = avg_size / n_test

      # save the results
      res_ = {'cover': coverage, 'set_size': avg_size}
      with open(save_path + '/' + file_name_, 'wb') as f:
        pickle.dump(res_, f)

    print('coverage:', coverage)
    print('set size:', avg_size)

    if 'ArbiTree-CQC' not in method_names:
      method_names.append('ArbiTree-CQC')
    if 'ArbiTree-CQC' not in coverage_res:
      coverage_res['ArbiTree-CQC'] = np.zeros((mc_runs, n_alpha))
      set_size_res['ArbiTree-CQC'] = np.zeros((mc_runs, n_alpha))
    coverage_res['ArbiTree-CQC'][mc_idx, :] = coverage
    set_size_res['ArbiTree-CQC'][mc_idx, :] = avg_size


# make plots and tables for the results

"""
### plot coverage results
plt.figure()
for method in method_names:
  plt.plot(confi_levels, coverage_res[method], label=method)
plt.xlabel('Confidence level')
plt.ylabel('Coverage')
plt.legend()

plt.savefig('coverage.pdf')

### plot set size results
plt.figure()
for method in method_names:
  plt.plot(confi_levels, set_size_res[method], label=method)
plt.xlabel('Confidence level')
plt.ylabel('Average set size')
plt.legend()

plt.savefig('set_size.pdf')
"""

## make table

def format_mean_stderr(data, decimals=3):
    """
    Compute mean and standard error for each column of data and format as 'mean ± stderr'.
    Args:
        data: np.ndarray, shape (mc_runs, n_alpha)
        decimals: int, number of decimal places for formatting
    Returns:
        list of strings: ['mean1 ± stderr1', 'mean2 ± stderr2', ...]
    """
    mean = np.mean(data, axis=0)
    stderr = np.std(data, axis=0)
    return [f"{m:.{decimals}f} ± {s:.{decimals}f}" for m, s in zip(mean, stderr)]

# Initialize lists for tables
coverage_table = []
set_size_table = []

# Build tables
for method in method_names:
    coverage_data = coverage_res[method]
    set_size_data = set_size_res[method]
    coverage_row = [method] + format_mean_stderr(coverage_data)
    set_size_table.append([method] + format_mean_stderr(set_size_data))
    coverage_table.append(coverage_row)

# Define headers for tables and DataFrames
headers = ['Method'] + [f'α={alpha:.2f}' for alpha in confi_levels]

# Print tables
print("\nCoverage Results:")
header = "Method".ljust(10) + "".join(f"α={alpha:.2f}".ljust(15) for alpha in confi_levels)
print(header)
print("-" * len(header))
for row in coverage_table:
    print(row[0].ljust(10) + "".join(val.ljust(15) for val in row[1:]))

print("\nSet Size Results:")
print(header)
print("-" * len(header))
for row in set_size_table:
    print(row[0].ljust(10) + "".join(val.ljust(15) for val in row[1:]))

coverage_df = pd.DataFrame(coverage_table, columns=headers)
set_size_df = pd.DataFrame(set_size_table, columns=headers)
coverage_df.to_csv(save_path + '/coverage_table.csv', index=False)
set_size_df.to_csv(save_path + '/set_size_table.csv', index=False)

print('finished.')