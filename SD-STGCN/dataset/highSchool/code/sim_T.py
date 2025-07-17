from sim_utils import *
import sys
import argparse
import os

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--n_frame', type=int, default=16)
parser.add_argument('--len_seq', type=int, default=400)
parser.add_argument('--graph_type', type=str, default='highSchool')
parser.add_argument('--sim_type', type=str, default='MixedSIR')
parser.add_argument('--R0', type=float, default=2.5)
parser.add_argument('--beta', type=float, default=0.3) # If gamma > 0, then ignore this input variable.
parser.add_argument('--gamma', type=float, default=0) # If gamma <= 0, then set gamma = 0, the SIR model = SI model.
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--f0', type=float, default=0.02)
parser.add_argument('--T', type=int, default=30)
parser.add_argument('--n_sources', type=str, default='[[1,0.5],[7,0.4],[10,-1]]')
parser.add_argument('--skip', type=str, default='[[1,2],[7,1],[10,1]]') # set the skip step for different nsrc.

args = parser.parse_args()

T = args.T
n_sources = args.n_sources

sim_type = args.sim_type # simulation type


gpath = '../data/graph/'
spath = '../data/%s/' % (sim_type)

graph_type = args.graph_type
graph_name = gpath + '%s.edgelist' % (graph_type)

# load graph
g = nx.read_edgelist(graph_name, nodetype=int)

N = len(g.nodes)

len_seq = args.len_seq # num of sequences = num of sources
num_frames = args.n_frame # num of frames = num of snapshots

# minimum outbreak fraction
f0 = args.f0
R0 = round(args.R0, 2) #2.5
gamma = round(args.gamma, 2) #0.4 is the one used by the finding patient zero paper
alpha = round(args.alpha, 2)

A = nx.adjacency_matrix(g).todense().astype(float)
lambda_max = eigs(A, k=1, which='LR')[0][0].real
if gamma > 0:
    beta = round(R0 * gamma / lambda_max, 3)
    print(lambda_max)
    print(beta)
else:
    gamma = 0
    beta = args.beta
if sim_type == 'SIR':
    sim_file = '%s_nsrc%s_Rzero%s_beta%s_gamma%s_T%s_ls%s_nf%s_entire.pickle' % \
               (sim_type, n_sources, R0, beta, gamma, T, len_seq, num_frames)
    skip_list = eval(args.skip)
elif sim_type == 'MixedSIR':
    nsrc_list = eval(args.n_sources)  # convert to list
    skip_list = eval(args.skip)  # convert to list
    sim_file = 'MixedSIR'
    for nsrc, ratio in nsrc_list:
        sim_file += '_nsrc%d-ratio%s' % (nsrc, ratio)
    sim_file += '_Rzero%s_beta%s_gamma%s_T%s_ls%s_nf%s_entire.pickle' % \
                (R0, beta, gamma, T, len_seq, num_frames)
elif sim_type == 'SEIR':
    sim_file = '%s_nsrc%s_Rzero%s_beta%s_gamma%s_alpha%s_T%s_ls%s_nf%s_entire.pickle' %\
    (sim_type,n_sources,R0,beta,gamma,alpha,T,len_seq,num_frames)
else:
    raise Exception('unknown simulation type')

# sequence
X = [] # features, shape [len_seq, num_frames, N, num_channels]
y = [] # labels, i.e. source node index, shape [len_seq, 1]
meta = [] # meta data, shape [len_seq], each element is [skip].

# simulation from different sources
sim_length = []


if sim_type == 'MixedSIR':
    mixed_sir = MixedSIR(N, g, beta, gamma, nsrc_ratios=eval(args.n_sources), min_outbreak_frac=f0)
    mixed_sir.prepare_plan(len_seq)

    for i in range(len_seq):
        if (i + 1) % 100 == 0:
            print(f"Processing {i + 1} / {len_seq} samples")
        mixed_sir.init()
        mixed_sir.iteration_bunch(T)

        if mixed_sir.is_outbreak and len(mixed_sir.iterations) > num_frames:
            X.append(mixed_sir.iterations)
            y.append(mixed_sir.src)

            n_src_ = len(mixed_sir.src)
            for n_src__, skip__ in skip_list:
                if n_src__ == n_src_:
                    skip_ = skip__
                    break
            meta.append([skip_])
            sim_length.append(len(mixed_sir.iterations))
        else:
            while True:
                mixed_sir.init()
                mixed_sir.iteration_bunch(T)
                if mixed_sir.is_outbreak and len(mixed_sir.iterations) > num_frames:
                    X.append(mixed_sir.iterations)
                    y.append(mixed_sir.src)
                    sim_length.append(len(mixed_sir.iterations))
                    break
else:
    i = 0
    while i < len_seq:
        if (i + 1) % 100 == 0:
            print(f"Processing {i + 1} / {len_seq} samples")

        if sim_type == 'SIR':
            sim = MultiSIR(N, g, beta, gamma, num_sources=int(n_sources), min_outbreak_frac=f0)
        elif sim_type == 'SEIR':
            sim = SEIR(N, g, beta, gamma, alpha, min_outbreak_frac=f0)
        else:
            raise Exception('unknown simulation type')

        sim.init()
        sim.iteration_bunch(T)

        if sim.is_outbreak and len(sim.iterations) > num_frames:
            X.append(sim.iterations)
            y.append(sim.src)

            n_src_ = len(sim.src)
            for n_src__, skip__ in skip_list:
                if n_src__ == n_src_:
                    skip_ = skip__
                    break
            meta.append([skip_])
            sim_length.append(len(sim.iterations))
            i += 1


if not os.path.exists(spath):
    os.makedirs(spath)

pickle.dump((X,y,meta), open(spath+sim_file,'wb'))

print('-----------------------------------------------------')
print('mean epidemic length: %.1f' % (np.mean(sim_length)))
print('stdev epidemic length: %.2f' % (np.std(sim_length)))
print('-----------------------------------------------------')

