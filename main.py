import networkx as nx
import pickle
import numpy as np
import os
from matplotlib import pyplot as plt
from utils.score_convert import APS_score

# Set parameters

## Parameters for propagation model
Rzero = 2.5
beta = 0.3
gamma = 0
T = 30
ls = 4000
nf = 16
graph = 'highSchool'

if gamma <= 0:
  prop_model = 'SI'
else:
  prop_model = 'SIR'

## Parameters for Conformal Prediction
calib_ratio = 0.5
confi_level = 0.05

# Load data
# graph_path = 'SD-STGCN/dataset/highSchool/data/graph/highSchool.edgelist'
# data_path = 'SD-STGCN/output/test_res/highSchool/exp1/res.pickle'

exp_name = f"SIR_Rzero{Rzero}_beta{beta}_gamma{gamma}_T{T}_ls{ls}_nf{nf}"
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
n_nodes = G.number_of_nodes()
print('Number of nodes:', G.number_of_nodes())
print('Number of edges:', G.number_of_edges())

with open(data_path, 'rb') as f:
    data = pickle.load(f)

inputs_raw = data['inputs']
pred_scores_raw = data['predictions']
ground_truths_raw = data['ground_truth']

print(len(inputs_raw))

# Prepare the data
## unzipe the data
pred_scores = []
ground_truths = []
for i in range(len(pred_scores_raw)):
  for j in range(len(pred_scores_raw[i])):
    pred_scores.append(pred_scores_raw[i][j])
    ground_truths.append(ground_truths_raw[i][j])

## partition the data
n_samples = len(pred_scores)
n_calibration = int(n_samples * calib_ratio)
n_test = n_samples - n_calibration

calib_index = np.random.choice(n_samples, n_calibration, replace=False)
test_index = np.setdiff1d(np.arange(n_samples), calib_index)
print('Number of samples:', n_samples, 'Calibration size:', n_calibration, 'test size:', n_test)

pred_scores_calib = [pred_scores[i] for i in calib_index]
ground_truths_calib = [ground_truths[i] for i in calib_index]

pred_scores_test = [pred_scores[i] for i in test_index]
ground_truths_test = [ground_truths[i] for i in test_index]

# Conformal Prediction
## Compute conformity scores on the calibration set
cfscore_calib = []
for i in range(n_calibration):
  cfscore_calib.append(APS_score(pred_scores_calib[i], ground_truths_calib[i]))
cfscore_calib = np.array(cfscore_calib)

## Compute conformity scores on the test set
cfscore_test = []
for i in range(n_test):
  cfscore_ = []
  for j in range(n_nodes):
    cfscore_.append(APS_score(pred_scores_test[i], j))
  cfscore_test.append(cfscore_)
cfscore_test = np.array(cfscore_test)
print('Conformity scores computed. shape of test:' + str(cfscore_test.shape))

## compute the quantile
threshold = np.quantile(cfscore_calib, 1 - confi_level, method = 'lower')
print('Quantile:', threshold)

## Apply the quantile on the test set
pred_sets = []
for i in range(n_test):
  pred_set_ = []
  for j in range(n_nodes):
    if cfscore_test[i][j] <= threshold:
      pred_set_.append(j)
  pred_sets.append(pred_set_)
print('Prediction sets computed.')

## Verify the test results
### coverage
cover_num = 0
for i in range(n_test):
  if ground_truths_test[i] in pred_sets[i]:
    cover_num = cover_num + 1
coverage = cover_num / n_test
print('coverage:', coverage)

### set size
avg_size = 0
for i in range(n_test):
  avg_size = avg_size + len(pred_sets[i])
avg_size = avg_size / n_test
print('set size:', avg_size)

print('finished.')