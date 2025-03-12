# --------------
# main.py
# --------------

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin

import tensorflow as tf

tf.compat.v1.disable_eager_execution() # disable eager execution

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import model_train
from models.tester import model_test

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_node', type=int, default=774)

parser.add_argument('--n_frame', type=int, default=16)

parser.add_argument('--n_channel', type=int, default=3)

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epoch', type=int, default=4)
parser.add_argument('--save', type=int, default=1)

parser.add_argument('--ks', type=int, default=4)
parser.add_argument('--kt', type=int, default=1)

parser.add_argument('--sconv', type=str, default='gcn') # spatio-convolution method, cheb or gcn
                                                            # cheb --chebyshev polinomials
                                                            # gcn -- kipf's gcn

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')

parser.add_argument('--dropout', type=float, default=0)

parser.add_argument('--graph', type=str, default='./dataset/highSchool/data/graph/highSchool.edgelist')
parser.add_argument('--seq', type=str, default='./dataset/highSchool/data/SIR/SIR_Rzero2.5_beta0.3_gamma0_T30_ls13600_nf16_entire.pickle')

parser.add_argument('--pred', type=str, default='./output/models/highSchool/pred_highSchool_nf16.pickle') # file to save predictions
parser.add_argument('--exp_name', type=str, default='SIR_Rzero2.5_beta0.3_gamma0_T30_ls13600_nf16') 
# the name of the experiment. e.g., SIR_Rzero${Rzero}_beta${beta}_gamma${gamma}_T${T}_ls${ns}_nf${nf}


parser.add_argument('--start', type=int, default=1)
parser.add_argument('--end', type=int, default=-1)

parser.add_argument('--gt', type=str, default='highSchool')

parser.add_argument('--valid', type=int, default=1)
parser.add_argument('--random', type=int, default=0)

parser.add_argument("--config", help="no use. just a placeholder", default="xx")

args = parser.parse_args()
print(f'Training configs: {args}')

n, n_frame = args.n_node, args.n_frame
Ks, Kt = args.ks, args.kt

sconv = args.sconv

# blocks: settings of channel size in st_conv_blocks / bottleneck design
blocks = [[3, 36, 144], [144, 36, 72]]

# Load weighted adjacency matrix W
if args.graph == 'default':
    gfile = './dataset/ER/data/graph/ER_N1000_p0.02_g0.edgelist'
else:
    gfile = args.graph



# load customized graph weight matrix
W = weight_matrix(gfile)

if sconv == 'cheb':
    # Calculate graph kernel
    L = scaled_laplacian(W)
    # Alternative approximation method: 1st approx - first_approx(W, n).
    Lk = cheb_poly_approx(L, Ks, n)
elif sconv == 'gcn':
    Lk = first_approx(W, n)
else:
    raise Exception('unknown spatio-conv method')


tf.compat.v1.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))

# Data Preprocessing
train_pct, val_pct = 130/136, 2/136

if args.seq == 'default':
    sfile =\
    './dataset/ER/data/SIR/SIR_Rzero2.5_gamma0.4_T30_ls2000_nf16_N1000_p0.02_g0.pickle'
else:
    sfile = args.seq


save_path='./output/models/%s/%s/' % (args.gt,args.exp_name)
load_path='./output/models/%s/%s/' % (args.gt,args.exp_name)

save_test_path = './output/test_res/%s/%s/' % (args.gt,args.exp_name)

if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists(save_test_path):
    os.makedirs(save_test_path)

dataset = data_gen(sfile, n, n_frame, train_pct, val_pct)

if __name__ == '__main__':

    model_train(dataset, blocks, args, save_path=save_path)
    model_test(dataset, args, load_path=load_path, save_test_path=save_test_path)


