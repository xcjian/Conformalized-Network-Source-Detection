# --------------
# main.py
# --------------

import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin

# import tensorflow as tf
import torch

# tf.compat.v1.disable_eager_execution() # disable eager execution

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.compat.v1.Session(config=config)

from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import *
from models.tester import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_node', type=int, default=100)

parser.add_argument('--n_frame', type=int, default=16)

parser.add_argument('--n_channel', type=int, default=3)

parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--save', type=int, default=10)

parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)

parser.add_argument('--sconv', type=str, default='cheb') # spatio-convolution method, cheb or gcn
                                                            # cheb --chebyshev polinomials
                                                            # gcn -- kipf's gcn

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')

parser.add_argument('--dropout', type=float, default=0)

parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--seq', type=str, default='default')

parser.add_argument('--pred', type=str, default='default') # file to save predictions
parser.add_argument('--exp_name', type=str, default='exp') 
# the name of the experiment. e.g., SIR_Rzero${Rzero}_beta${beta}_gamma${gamma}_T${T}_ls${ns}_nf${nf}


parser.add_argument('--start', type=int, default=1)
parser.add_argument('--end', type=int, default=-1)

parser.add_argument('--gt', type=str, default='ER')

parser.add_argument('--valid', type=int, default=0)
parser.add_argument('--random', type=int, default=1)

parser.add_argument('--train_pct', type=float, default=0.8)
parser.add_argument('--val_pct', type=float, default=0.1)

parser.add_argument('--pos_weight', type=float, default=10)
parser.add_argument('--train_flag', action='store_false', help='Flag to disable training (default: True)')
# if do not want train then just write --train_flag in the command line.
parser.add_argument('--prop_model', type=str, default='SIR') # can also be 'SI'.

args = parser.parse_args()
print(f'Training configs: {args}')

n, n_frame = args.n_node, args.n_frame
Ks, Kt = args.ks, args.kt

sconv = args.sconv

train_flag = args.train_flag
print(str(train_flag))

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
    Lk_np = cheb_poly_approx(L, Ks, n)
elif sconv == 'gcn':
    Lk_np = first_approx(W, n)
else:
    raise Exception('unknown spatio-conv method')

def process_Lk(Lk_np: np.ndarray, sconv: str, Ks: int) -> list[torch.Tensor]:
    """
    Process Lk from np.ndarray to list[torch.Tensor[n, n]] for STGCN usage.

    - If sconv == 'cheb': assume Lk_np.shape == [n, Ks * n]
    - If sconv == 'gcn':  assume Lk_np.shape == [n, n]
    """
    if sconv == 'cheb':
        n = Lk_np.shape[0]
        Lk_list = [Lk_np[:, k * n : (k + 1) * n] for k in range(Ks)]
    elif sconv == 'gcn':
        Lk_list = [Lk_np]  # wrap into list
    else:
        raise ValueError(f"Unknown sconv: {sconv}")

    return [torch.tensor(L, dtype=torch.float32) for L in Lk_list]



# tf.compat.v1.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))




# Data Preprocessing
train_pct, val_pct = args.train_pct, args.val_pct

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

dataset.Lk = process_Lk(Lk_np, sconv, Ks)
    
if __name__ == '__main__':
    # model_train_nodewise(dataset, blocks, args, save_path=save_path)
    # model_test_nodewise(dataset, args, load_path=load_path, save_test_path=save_test_path)
    args.blocks = blocks
    # model_train_pytorch_nodewise(dataset, blocks, args, save_path=save_path)

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    config_path = os.path.join(load_path, "args.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        if not hasattr(args, "blocks"):
            args.blocks = config["blocks"]
        if not hasattr(args, "keep_prob"):
            args.keep_prob = config.get("keep_prob", 1.0)

    model_test_pytorch_nodewise(dataset, args, load_path=load_path, save_test_path=save_test_path)


