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
calib_ratio = 0.5
pow_expected = 0.5

## Parameters for ADiT-DSI
discrepancies = [ADiT_h] # discrepancy function
discrepancy_str = 'ADiT_h'
m_l = 5
m_p = 5

# Load data
# graph_path = 'SD-STGCN/dataset/highSchool/data/graph/highSchool.edgelist'
# data_path = 'SD-STGCN/output/test_res/highSchool/exp1/res.pickle'

exp_name = f"SIR_nsrc{nsrc}_Rzero{Rzero}_beta{beta}_gamma{gamma}_T{T}_ls{ls}_nf{nf}_00exfin"
graph_extract_path = 'SD-STGCN/dataset/' + graph + '/data/graph/highSchool.edgelist'
data_extract_path = 'SD-STGCN/output/test_res/' + graph + '/' + exp_name + '/res.pickle'

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
"""
for i in range(len(pred_scores_raw)):
  for j in range(len(pred_scores_raw[i])):
    inputs.append(inputs_raw[i][j])
    pred_scores.append(pred_scores_raw[i][j])
    ground_truths.append(ground_truths_raw[i][j])
"""
for i in range(len(pred_scores_raw)):
  inputs.append(inputs_raw[i])
  pred_scores.append(pred_scores_raw[i])
  ground_truths.append(ground_truths_raw[i])

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
    # infected_nodes_ = np.nonzero(inputs_calib[i])[0]
    infected_nodes_ = np.nonzero(inputs_calib[i][0, :])[0]
    pred_prob_ = pred_scores_calib[i][:, 1]
    ground_truth_one_hot_ = ground_truths_calib[i]
    ground_truth_part_one_hot_ = set_truncate(ground_truth_one_hot_, pred_prob_, pow_expected)
    score_ = avg_score(pred_prob_, ground_truth_part_one_hot_, prop_model, infected_nodes_)
    cfscore_calib.append(score_)
  cfscore_calib = np.array(cfscore_calib)

  ## Compute conformity scores on the test set
  cfscore_test = []
  for i in range(n_test):
    # infected_nodes_ = np.nonzero(inputs_test[i])[0]
    infected_nodes_ = np.nonzero(inputs_test[i][0, :])[0]
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
      threshold = np.quantile(cfscore_calib, tail_prop, method = 'lower')

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
    # infected_nodes_ = np.nonzero(inputs_test[i])[0]
    infected_nodes_ = np.nonzero(inputs_test[i][0, :])[0]
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