import os
import sys
import networkx as nx
import pickle
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from utils.score_convert import *
from multiprocessing import Pool
from DSI.src.diffusion_source.infection_model import FixedTSI
from DSI.src.diffusion_source.discrepancies import ADiT_h
import argparse
import pandas as pd

np.random.seed(41)
# Set parameters

parser = argparse.ArgumentParser()

parser.add_argument('--graph', type=str, default='highSchool')
parser.add_argument('--train_exp_name', type=str, default='SIR_nsrc10-15_Rzero1-15_gamma0.1-0.4_ls40400_nf16') 
parser.add_argument('--test_exp_name', type=str, default='SIR_nsrc5-10_Rzero1-15_gamma0.1-0.4_ls1000_nf16')
parser.add_argument('--save_exp_name', type=str, default='nsrc5-10est10-15')
# the name of the experiment. e.g., SIR_Rzero${Rzero}_beta${beta}_gamma${gamma}_T${T}_ls${ns}_nf${nf}
parser.add_argument('--pow_expected', type=float, default=0.5)
parser.add_argument('--prop_model', type=str, default='SI')
parser.add_argument('--confi_levels', 
                    nargs='*',  # Accepts 0 or more values
                    type=float, 
                    default=[0.05, 0.07, 0.10, 0.15, 0.20],  # Default list
                    help='confi_levels')
parser.add_argument('--set_recall', type=int, default=0) # 0 means this method will not be used.
parser.add_argument('--set_prec', type=int, default=0)
parser.add_argument('--ADiT_DSI', type=int, default=0)
parser.add_argument('--PGM_CQC', type=int, default=0)
parser.add_argument('--ArbiTree_CQC', type=int, default=0)
parser.add_argument('--mc_runs', type=int, default=50)

# parameters for different methods

# ADiT-DSI
parser.add_argument('--m_l', type=int, default=20)
parser.add_argument('--m_p', type=int, default=20)

# PGM-CQC
parser.add_argument('--n_learn_tree', type=int, default=1000)
parser.add_argument('--n_jobs_Arbitree', type=int, default=5)

args = parser.parse_args()

graph_name = args.graph
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
pow_expected = args.pow_expected

## Parameters for ADiT-DSI
discrepancies = [ADiT_h] # discrepancy function
discrepancy_str = 'ADiT_h'
m_l = args.m_l
m_p = args.m_p

## Parameters for PGM-CQC
n_learn_tree = args.n_learn_tree
n_jobs_Arbitree = args.n_jobs_Arbitree

# Load data
# graph_path = 'SD-STGCN/dataset/highSchool/data/graph/highSchool.edgelist'
# data_path = 'SD-STGCN/output/test_res/highSchool/exp1/res.pickle'

train_exp_name = args.train_exp_name
test_exp_name = args.test_exp_name
save_exp_name = args.save_exp_name

graph_extract_path = 'SD-STGCN/dataset/' + graph_name + '/data/graph/' + graph_name + '.edgelist'
save_path = 'results/' + graph_name + '/' + save_exp_name + '/pow_expected' + str(pow_expected) # path for saving CP results

test_data_file = 'SD-STGCN/dataset/' + graph_name + '/data/SIR/' + test_exp_name + '_entire.pickle'
train_model_file = 'SD-STGCN/output/models/' + graph_name + '/' + train_exp_name
test_res_path = 'SD-STGCN/output/test_res/' + graph_name + '/' + save_exp_name

# read in the graph
graph_path = 'data/' + graph_name + '/graph/' + graph_name + '.edgelist'
if not os.path.exists(graph_path):
  os.makedirs(os.path.dirname(graph_path), exist_ok=True)
  os.system('cp ' + graph_extract_path + ' ' + graph_path)

graph = nx.read_edgelist(graph_path, nodetype=int)
G = nx.Graph(graph)
G.graph = G
n_nodes = G.number_of_nodes()
print('Number of nodes:', G.number_of_nodes())
print('Number of edges:', G.number_of_edges())

get_test_results(test_data_file, train_model_file, test_res_path, n_nodes) # evaluate the pre-trained model on the test set.

# copy the result.
os.makedirs(save_path, exist_ok=True)
## copy the calibration data
calibration_data_extract_path = 'SD-STGCN/output/test_res/' + graph_name + '/' + train_exp_name + '/res.pickle'
calibration_data_path = 'data/' + graph_name + '/test_res/' + save_exp_name + '/est_res.pickle'
if not os.path.exists(calibration_data_path):
  os.makedirs(os.path.dirname(calibration_data_path), exist_ok=True)
  os.system('cp ' + calibration_data_extract_path + ' ' + calibration_data_path)

## copy the test data
test_data_extract_path = test_res_path + '/res.pickle'
test_data_path = 'data/' + graph_name + '/test_res/' + save_exp_name + '/test_res.pickle'
if not os.path.exists(test_data_path):
  os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
  os.system('cp ' + test_data_extract_path + ' ' + test_data_path)

## load the calibration data set
with open(calibration_data_path, 'rb') as f:
    calib_data_set = pickle.load(f)

inputs_raw_calib = calib_data_set['inputs'] # n_batch x n_sample x n_nodes
pred_scores_raw_calib = calib_data_set['predictions']
ground_truths_raw_calib = calib_data_set['ground_truth']
logits_raw_calib = calib_data_set['logits']

inputs_calib = []
pred_scores_calib = []
ground_truths_calib = []
logits_calib = []
for i in range(len(pred_scores_raw_calib)):
  for j in range(len(pred_scores_raw_calib[i])):
    inputs_calib.append(inputs_raw_calib[i][j])
    pred_scores_calib.append(pred_scores_raw_calib[i][j])
    ground_truths_calib.append(ground_truths_raw_calib[i][j])
    logits_calib.append(logits_raw_calib[i][j])

## load the test data set
with open(test_data_path, 'rb') as f:
    test_data_set = pickle.load(f)

inputs_raw_test = test_data_set['inputs'] # n_batch x n_sample x n_nodes
pred_scores_raw_test = test_data_set['predictions']
ground_truths_raw_test = test_data_set['ground_truth']
logits_raw_test = test_data_set['logits']

inputs_test = []
pred_scores_test = []
ground_truths_test = []
logits_test = []
for i in range(len(pred_scores_raw_test)):
  for j in range(len(pred_scores_raw_test[i])):
    inputs_test.append(inputs_raw_test[i][j])
    pred_scores_test.append(pred_scores_raw_test[i][j])
    ground_truths_test.append(ground_truths_raw_test[i][j])
    logits_test.append(logits_raw_test[i][j])

## partition the data
n_calibration = 7600
calibration_index_all = np.arange(len(inputs_calib))
n_test = 400
test_index_all = np.arange(len(inputs_test))

method_names = []
coverage_res = {}
set_size_res = {}
time_cost_res = {}

for mc_idx in range(mc_runs):

  start_mc_time = time.time()

  calib_index_file_name_ = '/calib_index_repeat' + str(mc_idx) + '.npy'
  test_index_file_name_ = '/test_index_repeat' + str(mc_idx) + '.npy'

  try:
    calib_index = np.load(save_path + calib_index_file_name_)
    test_index = np.load(save_path + test_index_file_name_)
  except:
    calib_index = np.random.choice(calibration_index_all, n_calibration, replace=False)
    test_index = np.random.choice(test_index_all, n_test, replace=False)
    np.save(save_path + calib_index_file_name_, calib_index)
    np.save(save_path + test_index_file_name_, test_index)
  
  print('Calibration size:', n_calibration, 'test size:', n_test)

  inputs_calib_ = [inputs_calib[i] for i in calib_index]
  pred_scores_calib_ = [pred_scores_calib[i] for i in calib_index]
  ground_truths_calib_ = [ground_truths_calib[i] for i in calib_index]
  logits_calib_ = [logits_calib[i] for i in calib_index]

  inputs_test_ = [inputs_test[i] for i in test_index]
  pred_scores_test_ = [pred_scores_test[i] for i in test_index]
  ground_truths_test_ = [ground_truths_test[i] for i in test_index]
  logits_test_ = [logits_test[i] for i in test_index]


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
      time_cost = res_load['time_cost']
    except:
      # Conformal Prediction
      print('computing proposed method with recall score...')
      
      ## Compute conformity scores on the calibration set
      cfscore_calib = []
      for i in range(n_calibration):
        infected_nodes_ = np.nonzero(inputs_calib_[i])[0]
        pred_prob_ = pred_scores_calib_[i][:, 1]
        ground_truth_one_hot_ = ground_truths_calib_[i]
        ground_truth_part_one_hot_ = set_truncate(ground_truth_one_hot_, pred_prob_, pow_expected)
        score_ = recall_score(pred_prob_, ground_truth_part_one_hot_, prop_model, infected_nodes_)
        cfscore_calib.append(score_)
      cfscore_calib = np.array(cfscore_calib)

      ## Compute conformity scores on the test set
      start_time_method_ = time.time()
      cfscore_test = []
      for i in range(n_test):
        infected_nodes_ = np.nonzero(inputs_test_[i])[0]
        cfscore_ = recall_score_gtunknown(pred_scores_test_[i][:, 1], prop_model, infected_nodes_)
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
          threshold = cpquantile(cfscore_calib, tail_prop)

          for j in range(n_nodes):
            if cfscore_test[i][j] <= threshold:
              pred_sets[str(alpha)].add(j)
        #print('Prediction sets computed. index:', i)

        for j, alpha in enumerate(confi_levels):
          ground_truth_source = np.nonzero(ground_truths_test_[i])[0]
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
      time_cost = time.time() - start_time_method_

      # save the results
      res_ = {'cover': coverage, 'set_size': avg_size, 'time_cost': time_cost}
      with open(save_path + '/' + file_name_, 'wb') as f:
        pickle.dump(res_, f)
    
    print('coverage:', coverage)
    print('set size:', avg_size)
    print('time cost:', time_cost)

    if 'set_recall' not in method_names:
      method_names.append('set_recall')
    if 'set_recall' not in coverage_res:
      coverage_res['set_recall'] = np.zeros((mc_runs, n_alpha))
      set_size_res['set_recall'] = np.zeros((mc_runs, n_alpha))
      time_cost_res['set_recall'] = np.zeros((mc_runs, n_alpha))
    coverage_res['set_recall'][mc_idx, :] = coverage
    set_size_res['set_recall'][mc_idx, :] = avg_size
    time_cost_res['set_recall'][mc_idx, :] = time_cost
  
  if set_prec:

    # try to load data.
    file_name_ = '/set_prec_repeat' + str(mc_idx) + '.pickle'
    try:
      with open(save_path + file_name_, 'rb') as f:
        res_load = pickle.load(f)
      coverage = res_load['cover']
      avg_size = res_load['set_size']
      time_cost = res_load['time_cost']
    except:
      # Conformal Prediction
      print('computing proposed method with precision score...')
      
      ## Compute conformity scores on the calibration set
      cfscore_calib = []
      for i in range(n_calibration):
        infected_nodes_ = np.nonzero(inputs_calib_[i])[0]
        pred_prob_ = pred_scores_calib_[i][:, 1]
        ground_truth_one_hot_ = ground_truths_calib_[i]
        ground_truth_part_one_hot_ = set_truncate(ground_truth_one_hot_, pred_prob_, pow_expected)
        score_ = avg_score(pred_prob_, ground_truth_part_one_hot_, prop_model, infected_nodes_)
        cfscore_calib.append(score_)
      cfscore_calib = np.array(cfscore_calib)

      ## Compute conformity scores on the test set
      start_time_method_ = time.time()
      cfscore_test = []
      for i in range(n_test):
        infected_nodes_ = np.nonzero(inputs_test_[i])[0]
        cfscore_ = avg_score_gtunknown(pred_scores_test_[i][:, 1], prop_model, infected_nodes_)
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
          threshold = cpquantile(cfscore_calib, tail_prop)

          for j in range(n_nodes):
            if cfscore_test[i][j] <= threshold:
              pred_sets[str(alpha)].add(j)
        # print('Prediction sets computed. index:', i)

        for j, alpha in enumerate(confi_levels):
          ground_truth_source = np.nonzero(ground_truths_test_[i])[0]
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
      time_cost = time.time() - start_time_method_

      # save the results
      res_ = {'cover': coverage, 'set_size': avg_size, 'time_cost': time_cost}
      with open(save_path + '/' + file_name_, 'wb') as f:
        pickle.dump(res_, f)

    print('coverage:', coverage)
    print('set size:', avg_size)
    print('time cost:', time_cost)

    if 'set_prec' not in method_names:
      method_names.append('set_prec')
    if 'set_prec' not in coverage_res:
      coverage_res['set_prec'] = np.zeros((mc_runs, n_alpha))
      set_size_res['set_prec'] = np.zeros((mc_runs, n_alpha))
      time_cost_res['set_prec'] = np.zeros((mc_runs, n_alpha))
    coverage_res['set_prec'][mc_idx, :] = coverage
    set_size_res['set_prec'][mc_idx, :] = avg_size
    time_cost_res['set_prec'][mc_idx, :] = time_cost

  if ADiT_DSI:
    # ADiT-DSI

    # try to load data.
    file_name_ = '/ADiT_DSI_repeat' + str(mc_idx) + '.pickle'
    try:
      with open(save_path + file_name_, 'rb') as f:
        res_load = pickle.load(f)
      coverage = res_load['cover']
      avg_size = res_load['set_size']
      time_cost = res_load['time_cost']
      print('coverage:', coverage)
      print('set size:', avg_size)
      print('time cost:', time_cost)
    except:
      print('computing ADiT-DSI...')
      if prop_model != 'SI':
        print('SIR model is not supported by ADiT-DSI. Skipping...')
      else:
        start_time_method_ = time.time()
        coverage = np.zeros(n_alpha)
        avg_size = np.zeros(n_alpha)

        for i in range(n_test):
          infected_nodes_ = np.nonzero(inputs_test[i])[0]

          model = FixedTSI(G, discrepancies, canonical=True, expectation_after=False, m_l=m_l, m_p=m_p, T=len(infected_nodes_) - 1)

          start_time = time.time()
          confidence_sets = model.confidence_set_mp(infected_nodes_, confi_levels, new_run=True) 
          confidence_sets = confidence_sets[discrepancy_str]
          #print('Time:', time.time() - start_time, 'index:', i)

          for j, alpha in enumerate(confi_levels):
            ground_truth_source_ = np.nonzero(ground_truths_test[i])[0]
            if int(ground_truth_source_) in confidence_sets[str(alpha)]:
              coverage[j] = coverage[j] + 1
            
            avg_size[j] = avg_size[j] + len(confidence_sets[str(alpha)])        
        
        coverage = coverage / n_test
        avg_size = avg_size / n_test
        time_cost = time.time() - start_time_method_

        # save the results
        res_ = {'cover': coverage, 'set_size': avg_size, 'time_cost': time_cost}
        with open(save_path + '/' + file_name_, 'wb') as f:
          pickle.dump(res_, f)

        print('coverage:', coverage)
        print('set size:', avg_size)
        print('time cost:', time_cost)

    if 'ADiT_DSI' not in method_names:
      method_names.append('ADiT_DSI')
    if 'ADiT_DSI' not in coverage_res:
      coverage_res['ADiT_DSI'] = np.zeros((mc_runs, n_alpha))
      set_size_res['ADiT_DSI'] = np.zeros((mc_runs, n_alpha))
      time_cost_res['ADiT_DSI'] = np.zeros((mc_runs, n_alpha))
    coverage_res['ADiT_DSI'][mc_idx, :] = coverage
    set_size_res['ADiT_DSI'][mc_idx, :] = avg_size  
    time_cost_res['ADiT_DSI'][mc_idx, :] = time_cost

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
      time_cost = res_load['time_cost']
    except:
      # Conformal Prediction
      print('computing ArbiTree-CQC...')
      
      # compute Y_hat for each vertex (i.e., label) on the calibration set
      Y_calib = []
      Y_hat_calib = []
      for i in range(n_calibration):
        infected_nodes_ = np.nonzero(inputs_calib_[i])[0]
        # infected_nodes_ = np.nonzero(inputs_calib[i][0, :])[0]
        pred_prob_ = pred_scores_calib_[i][:, 1]
        # if prop_model == 'SI':
        #   non_infected_nodes = np.setdiff1d(np.arange(n_nodes), infected_nodes_)
        #   pred_prob_[non_infected_nodes] = 0

        ground_truth_one_hot_ = ground_truths_calib_[i]
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
      start_time_method_ = time.time()
      '''
      # non-parallel version:
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
      '''
      # parallel version:
      def compute_cfscore(i):
          """Helper function for parallel computation using global variables"""
          infected_nodes_ = np.nonzero(inputs_test_[i])[0]
          pred_prob_ = pred_scores_test_[i][:, 1]
          Y_hat_ = (pred_prob_ - 1/2) * 2  # align with {-1, 1} Y values
          return MPmaxscore(Y_hat_, tree_edges, tree_alpha, tree_beta)

      # Parallel computation of conformity scores
      def parallel_compute_cfscores(n_jobs=4):
          """Parallel version using global variables"""
          n_test = len(inputs_test_)
          with Pool(processes=n_jobs) as pool:
              cfscore_test = pool.map(compute_cfscore, range(n_test))
          return np.array(cfscore_test)

      cfscore_test = parallel_compute_cfscores(n_jobs=5)
      
      # Construct prediction set on the test set
      coverage = np.zeros(n_alpha)
      avg_size = np.zeros(n_alpha)

      for i in range(n_test):
        pred_sets = {}
        for alpha in confi_levels:
          pred_sets[str(alpha)] = set()

          ## compute the quantile
          tail_prop = (1 - alpha) * (1 + 1 / (n_calibration - n_learn_tree))
          threshold = -cpquantile(-cfscore_calib, tail_prop)

          for j in range(n_nodes):
            if cfscore_test[i][j] >= threshold:
              pred_sets[str(alpha)].add(j)
        #print('Prediction sets computed. index:', i)

        # evaluate the performance
        for j, alpha in enumerate(confi_levels):
          ground_truth_source = np.nonzero(ground_truths_test_[i])[0]
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
      time_cost = time.time() - start_time_method_

      # save the results
      res_ = {'cover': coverage, 'set_size': avg_size, 'time_cost': time_cost}
      with open(save_path + '/' + file_name_, 'wb') as f:
        pickle.dump(res_, f)

    print('coverage:', coverage)
    print('set size:', avg_size)
    print('time cost:', time_cost)

    if 'ArbiTree-CQC' not in method_names:
      method_names.append('ArbiTree-CQC')
    if 'ArbiTree-CQC' not in coverage_res:
      coverage_res['ArbiTree-CQC'] = np.zeros((mc_runs, n_alpha))
      set_size_res['ArbiTree-CQC'] = np.zeros((mc_runs, n_alpha))
      time_cost_res['ArbiTree-CQC'] = np.zeros((mc_runs, n_alpha))
    coverage_res['ArbiTree-CQC'][mc_idx, :] = coverage
    set_size_res['ArbiTree-CQC'][mc_idx, :] = avg_size
    time_cost_res['ArbiTree-CQC'][mc_idx, :] = time_cost

  print('finished repeatition:', mc_idx, 'time cost:', time.time()-start_mc_time)


# make plots and tables for the results


# --- Box Plot for Coverage ---
plt.figure(figsize=(12, 6))
data_to_plot = []
positions = []
xtick_labels = []

# Generate distinct colors for each method
colors = plt.cm.tab10(np.linspace(0, 1, len(method_names)))

# Prepare data for box plots
for i, alpha in enumerate(confi_levels):
    for j, method in enumerate(method_names):
        data_to_plot.append(coverage_res[method][:, i])  # Coverage data
        positions.append(i + 0.2 * j)  # Offset positions for each method
    xtick_labels.append(f'α={alpha:.2f}')

# Create box plots
box = plt.boxplot(data_to_plot, 
                 positions=positions, 
                 widths=0.15, 
                 patch_artist=True,
                 showfliers=False)  # Hide outliers for cleaner plot

# Assign colors to boxes (FIXED: no multiplication of colors array)
for i, patch in enumerate(box['boxes']):
    method_idx = i % len(method_names)  # Cycle through method colors
    patch.set_facecolor(colors[method_idx])

# Customize plot
plt.xticks(np.arange(len(confi_levels)) + 0.2*(len(method_names)-1)/2, 
           xtick_labels)
plt.xlabel('Confidence Level (α)', fontsize=12)
plt.ylabel('Coverage', fontsize=12)
plt.title('Coverage Across Methods and Confidence Levels', fontsize=14)

# Create legend
legend_elements = [Patch(facecolor=colors[i], label=method_names[i]) 
                  for i in range(len(method_names))]
plt.legend(handles=legend_elements, loc='upper left', fontsize=10)

plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig(save_path + '/coverage_boxplot.pdf', dpi=300, bbox_inches='tight')
plt.close()

# --- Box Plot for Set Size ---
plt.figure(figsize=(12, 6))
# plt.ylim(0, 40)
data_to_plot = []
positions = []

# Reuse the same color scheme
for i, alpha in enumerate(confi_levels):
    for j, method in enumerate(method_names):
        data_to_plot.append(set_size_res[method][:, i])  # Set size data
        positions.append(i + 0.2 * j)  # Same offset as coverage plot

# Create box plots
box = plt.boxplot(data_to_plot,
                 positions=positions,
                 widths=0.15,
                 patch_artist=True,
                 showfliers=False)

# Color boxes using the same method
for i, patch in enumerate(box['boxes']):
    method_idx = i % len(method_names)
    patch.set_facecolor(colors[method_idx])

# Customize plot (same x-axis as coverage plot)
plt.xticks(np.arange(len(confi_levels)) + 0.2*(len(method_names)-1)/2,
           xtick_labels)
plt.xlabel('Confidence Level (α)', fontsize=12)
plt.ylabel('Average Set Size', fontsize=12)
plt.title('Prediction Set Size Across Methods and Confidence Levels', fontsize=14)
plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig(save_path + '/set_size_boxplot.pdf', dpi=300, bbox_inches='tight')
plt.close()

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
time_cost_table = []

# Build tables
for method in method_names:
    coverage_data = coverage_res[method]
    set_size_data = set_size_res[method]
    time_cost_data = time_cost_res[method]

    coverage_row = [method] + format_mean_stderr(coverage_data)
    set_size_row = [method] + format_mean_stderr(set_size_data)
    time_cost_row = [method] + format_mean_stderr(time_cost_data)

    coverage_table.append(coverage_row)
    set_size_table.append(set_size_row)
    time_cost_table.append(time_cost_row)

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

print("\nTime Cost Results:")  # NEW: Print time_cost table
print(header)
print("-" * len(header))
for row in time_cost_table:
    print(row[0].ljust(10) + "".join(val.ljust(15) for val in row[1:]))

coverage_df = pd.DataFrame(coverage_table, columns=headers)
set_size_df = pd.DataFrame(set_size_table, columns=headers)
time_cost_df = pd.DataFrame(time_cost_table, columns=headers)
coverage_df.to_csv(save_path + '/coverage_table.csv', index=False)
set_size_df.to_csv(save_path + '/set_size_table.csv', index=False)
time_cost_df.to_csv(save_path + '/time_cost_table.csv', index=False)

print('finished.')