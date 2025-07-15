# -------------------------------------
# simulations with mix R0, gamma randomly picked from given ranges
# simulations end when I = 0
# -----------------------------------
from sim_utils import *
import sys
import argparse
import os

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--n_frame', type=int, default=16)
parser.add_argument('--len_seq', type=int, default=2000)
parser.add_argument('--graph_type', type=str, default='ER')
parser.add_argument('--sim_type', type=str, default='SIR')
parser.add_argument('--f0', type=float, default=0.02)

parser.add_argument('--nsrc_lo', type=int, default=1)
parser.add_argument('--nsrc_hi', type=int, default=15)
parser.add_argument('--R0_lo', type=int, default=1)
parser.add_argument('--R0_hi', type=int, default=15)
parser.add_argument('--gamma_lo', type=float, default=0.1)
parser.add_argument('--gamma_hi', type=float, default=0.4)
parser.add_argument('--skip', type=int, default=1)

args = parser.parse_args()


sim_type = args.sim_type # simulation type


gpath = '../data/graph/'
spath = '../data/%s/' % (sim_type)

os.makedirs(spath, exist_ok=True)

graph_type = args.graph_type
graph_name = gpath + '%s.edgelist' % (graph_type)

# load graph
g = nx.read_edgelist(graph_name, nodetype=int)

N = len(g.nodes)

A = nx.adjacency_matrix(g).todense().astype(float)
lambda_max = eigs(A, k=1, which='LR')[0][0].real

len_seq = args.len_seq # num of sequences = num of sources
num_frames = args.n_frame # num of frames = num of snapshots

nsrc_lo = args.nsrc_lo
nsrc_hi = args.nsrc_hi

R0_lo = args.R0_lo
R0_hi = args.R0_hi
gamma_lo = args.gamma_lo
gamma_hi = args.gamma_hi

nsrc_k = str(int(nsrc_lo)) + '-' + str(int(nsrc_hi))
R0_k = str(R0_lo) + '-' + str(R0_hi)
gamma_k = str(gamma_lo) + '-' + str(gamma_hi)

# minimum outbreak fraction
f0 = args.f0
#alpha = round(args.alpha, 2)


sim_file = '%s_nsrc%s_Rzero%s_gamma%s_ls%s_nf%s_entire.pickle' %\
            (sim_type,nsrc_k,R0_k,gamma_k,len_seq,num_frames)

# sequence
X = [] # features, shape [len_seq, num_frames, N, num_channels]
y = [] # labels, i.e. source node index, shape [len_seq, 1]
meta = [] # meta data, shape [len_seq], each element is [skip].

# simulation from different sources
i = 0
sim_length = []
while i < len_seq:

    n_sources = np.random.randint(nsrc_lo, nsrc_hi + 1)
    R0 = np.random.uniform(R0_lo, R0_hi)
    gamma = np.random.uniform(gamma_lo, gamma_hi)
    skip = 1
    beta = R0 * gamma / lambda_max

    sim = MultiSIR(N, g, beta, gamma, num_sources=int(n_sources), min_outbreak_frac=f0)
    sim.init()
    sim.run()

    if sim.is_outbreak and len(sim.iterations) > num_frames:
        sim_length.append(len(sim.iterations))

        i += 1
        X.append(sim.iterations)
        y.append(sim.src)
        meta.append([skip])
    
    print(str(i) + '-th sample finishd=ed.')

print('-----------------------------------------------------')
print('mean epidemic length: %.1f' % (np.mean(sim_length)))
print('stdev epidemic length: %.2f' % (np.std(sim_length)))
print('-----------------------------------------------------')

pickle.dump((X,y,meta), open(spath+sim_file,'wb'))


