import pickle
import torch
import networkx as nx

gpath = '/home/featurize/work/CP_source_det/SD-STGCN/dataset/highSchool/data/graph/highSchool.edgelist'
dpath = '/home/featurize/work/CP_source_det/SD-STGCN/dataset/highSchool/data/SIR/'

with open(dpath + 'SIR_Rzero2.5_beta0.3_gamma0_T30_ls13600_nf16_entire.pickle', 'rb') as f:
    data = pickle.load(f)

graph = nx.read_edgelist(gpath, nodetype=int)
G = nx.Graph(graph)
n_nodes = G.number_of_nodes()

X = []
y = data[1]
for i in range(len(data[0])):
    snapshots_ = data[0][i]
    sample_snapshot_ = snapshots_[1]
    infected_nodes_ = list(sample_snapshot_['status'].keys())

    # find the infected nodes in the 0-th snapshot
    zeroth_snapshot_ = snapshots_[0]['status']
    # find the key with value 1
    source_node_ = [k for k, v in zeroth_snapshot_.items() if v == 1][0]
    infected_nodes_.append(source_node_)

    onehot_vec_ = torch.zeros(n_nodes)
    onehot_vec_[infected_nodes_] = 1
    X.append(onehot_vec_)

data_torch = {'X': X, 'y': y}
with open(dpath + 'SIR_Rzero2.5_beta0.3_gamma0_T30_ls13600_nf16_entire_torch.pickle', 'wb') as f:
    pickle.dump(data_torch, f)

print('ok')

"""
The data loaded above takes the following form:
X: a list which contains observed propogation data for each iteration.
each element include the following elements:
'iteration': the current iteration number.
'status': indices of nodes in each status.
'node_count': the number of nodes in each status.
'status_delta': the change in the number of nodes in each status.
y: a list which contains the source node index for each iteration.
"""