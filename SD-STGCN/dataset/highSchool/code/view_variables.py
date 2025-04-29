import pickle
import os

gpath = os.path.dirname(os.getcwd())

with open(gpath + '/data/SIR/SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls10_nf16_entire.pickle', 'rb') as f:
    data = pickle.load(f)

for i in range(6):
    print(data[0][0][i]['node_count'])
print(data[1][0])

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