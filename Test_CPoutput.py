import os
import sys
import networkx as nx
import pickle
import numpy as np
import time
from matplotlib import pyplot as plt
from utils.score_convert import *
# from DSI.src.diffusion_source.infection_model import FixedTSI
from DSI.src.diffusion_source.discrepancies import ADiT_h

np.random.seed(41)
# Set parameters

## Parameters for propagation model
nsrc = 7 # number of sources
Rzero = 2.5
beta = 0.25
gamma = 0
T = 30
ls = 21200
nf = 16
graph = 'highSchool'

if gamma <= 0:
  prop_model = 'SI'
else:
  prop_model = 'SIR'

## Public parameters
confi_levels = [0.02, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]
n_alpha = len(confi_levels)

## Parameters for Conformal Prediction
proposed_method = True
ADiT_DSI = False
PGM_CQC = False
ArbiTree_CQC = False
calib_ratio = 0.5
pow_expected = 0.7
start_freq = 0
end_freq = 774

## Parameters for ADiT-DSI
discrepancies = [ADiT_h] # discrepancy function
discrepancy_str = 'ADiT_h'
m_l = 5
m_p = 5

## Parameters for PGM-CQC
n_learn_tree = 200

# Load data
# graph_path = 'SD-STGCN/dataset/highSchool/data/graph/highSchool.edgelist'
# data_path = 'SD-STGCN/output/test_res/highSchool/exp1/res.pickle'

exp_name = f"SIR_nsrc{nsrc}_Rzero{Rzero}_beta{beta}_gamma{gamma}_T{T}_ls{ls}_nf{nf}"
graph_extract_path = 'SD-STGCN/dataset/' + graph + '/data/graph/highSchool.edgelist'
data_extract_path = 'SD-STGCN/output/test_res/' + graph + '/' + exp_name + '/res.pickle'

'''
# copy these data to the data folder, if not copied
graph_path = 'data/' + graph + '/graph/' + graph + '.edgelist'
data_path = 'data/' + graph + '/test_res/' + exp_name + '/res.pickle'
if not os.path.exists(graph_path):
  os.makedirs(os.path.dirname(graph_path), exist_ok=True)
os.system('cp ' + graph_extract_path + ' ' + graph_path)
if not os.path.exists(data_path):
  os.makedirs(os.path.dirname(data_path), exist_ok=True)
os.system('cp ' + data_extract_path + ' ' + data_path)

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
'''

data_path = 'data/highSchool/test_res/SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16_GNN.pickle'


with open(data_path, 'rb') as f:
    data = pickle.load(f)

inputs_raw = data['inputs'] # n_batch x n_sample x n_nodes
pred_scores_raw = data['predictions']
ground_truths_raw = data['ground_truth']

# Prepare the data
## unzip the data
inputs = []
pred_scores = []
ground_truths = []

for i in range(len(pred_scores_raw)):
  for j in range(len(pred_scores_raw[i])):
    inputs.append(inputs_raw[i][j])
    pred_scores.append(pred_scores_raw[i][j])
    ground_truths.append(ground_truths_raw[i][j])

## partition the data
n_samples = len(pred_scores)
n_calibration = int(n_samples * calib_ratio)
n_test = n_samples - n_calibration

calib_index = np.random.choice(n_samples, n_calibration, replace=False)
test_index = np.setdiff1d(np.arange(n_samples), calib_index)
print('Number of samples:', n_samples, 'Calibration size:', n_calibration, 'test size:', n_test)

inputs_calib = [inputs[i] for i in calib_index]
pred_scores_calib = [pred_scores[i] for i in calib_index]
ground_truths_calib = [ground_truths[i] for i in calib_index]

inputs_test = [inputs[i] for i in test_index]
pred_scores_test = [pred_scores[i] for i in test_index]
ground_truths_test = [ground_truths[i] for i in test_index]

method_names = []
coverage_res = {}
set_size_res = {}

if proposed_method:
  # Conformal Prediction
  print('computing proposed method...')
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
    print('Prediction sets computed. index:', i)

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
  print('coverage:', coverage)
  print('set size:', avg_size)

  method_names.append('proposed')
  coverage_res['proposed'] = coverage
  set_size_res['proposed'] = avg_size

  ### To compare, comute the average size of infected set.
  infected_num = 0
  for i in range(n_test):
    infected_nodes_ = np.nonzero(inputs_test[i])[0]
    # infected_nodes_ = np.nonzero(inputs_test[i][0, :])[0]
    infected_num = infected_num + len(infected_nodes_)
  avg_infected_num = infected_num / n_test
  print('average infected size:', avg_infected_num)

if ADiT_DSI:
  # ADiT-DSI
  print('computing ADiT-DSI...')
  if prop_model == 'SI':

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

    print('coverage:', coverage)
    print('set size:', avg_size)

    method_names.append('ADiT-DSI')
    coverage_res['ADiT-DSI'] = coverage
    set_size_res['ADiT-DSI'] = avg_size

  else:
    print('SIR model is not supported by ADiT-DSI.')

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
    
    score_ = nodewise_APS_score(pred_prob_binary_, ground_truth_part_one_hot_, infected_nodes_, prop_model)
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

  # Conformal Prediction
  print('computing ArbiTree-CQC...')

  # compute Y_hat for each vertex (i.e., label) on the calibration set
  Y_calib = []
  Y_hat_calib = []
  for i in range(n_calibration):
    infected_nodes_ = np.nonzero(inputs_calib[i])[0]
    # infected_nodes_ = np.nonzero(inputs_calib[i][0, :])[0]
    pred_prob_ = pred_scores_calib[i][:, 1]
    if prop_model == 'SI':
      non_infected_nodes = np.setdiff1d(np.arange(n_nodes), infected_nodes_)
      pred_prob_[non_infected_nodes] = 0

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
    if prop_model == 'SI':
      non_infected_nodes = np.setdiff1d(np.arange(n_nodes), infected_nodes_)
      pred_prob_[non_infected_nodes] = 0
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
  print('coverage:', coverage)
  print('set size:', avg_size)

  method_names.append('ArbiTree-CQC')
  coverage_res['ArbiTree-CQC'] = coverage
  set_size_res['ArbiTree-CQC'] = avg_size


## make plots for the results

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


print('finished.')