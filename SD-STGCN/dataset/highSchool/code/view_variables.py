import pickle

gpath = 'SD-STGCN/dataset/highSchool/data/SIR/'

with open(gpath + 'SIR_Rzero2.5_beta0.3_gamma0_T30_ls4000_nf16_entire.pickle', 'rb') as f:
    data = pickle.load(f)

print(data[0][0])
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