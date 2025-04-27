from data_loader.data_utils_pyt import *
from utils.metric_utils import *
from models_pyt.base_model import STGCN_SI

import numpy as np
import torch
import torch.nn.functional as F
import pickle
import os

def model_test(inputs, args, load_path='./output/models/', save_test_path=None):
    '''
    Load and test saved model from the checkpoint.
    inputs: instance of class Dataset, data source for test.
    args: instance of class argparse, args for training.
    load_path: str, the path of loaded model.
    '''
    n_frame = args.n_frame
    num_node = args.n_node
    n_channel = args.n_channel
    batch_size = args.batch_size
    start, end = args.start, args.end
    random = args.random

    # Load model
    checkpoint = torch.load(os.path.join(load_path, 'STGCN.pth'), map_location=args.device)
    model = STGCN_SI(n_frame, args.ks, args.kt, args.blocks, args.sconv, args.Lk, keep_prob=1.0).to(args.device)
                              
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'>> Loaded model from {os.path.join(load_path, "STGCN.pth")}')

    acc_test_list, mrr_test_list = [], []
    hit_5_list, hit_10_list, hit_20_list = [], [], []
    all_inputs, all_pred_results, all_y_test = [], [], []

    with torch.no_grad():
        for x_test, y_test in gen_xy_batch(inputs.get_data('test'), batch_size, dynamic_batch=True, shuffle=False):
            x_test_ = onehot(iteration2snapshot(x_test, n_frame, start=start, end=end, random=random), n_channel)
            x_tensor = torch.tensor(x_test_).float().to(args.device)

            logits = model(x_tensor)
            pred_test = F.softmax(logits, dim=-1).cpu().numpy()

            acc_test_list.append(batch_acc(pred_test, y_test))
            mrr_test_list.append(batch_mrr(pred_test, y_test))
            hit_5_list += hit_at(pred_test, y_test, 5)
            hit_10_list += hit_at(pred_test, y_test, 10)
            hit_20_list += hit_at(pred_test, y_test, 20)

            if save_test_path:
                x_test_array = np.array([list(x_test_[i][0]) for i in range(len(x_test_))])
                all_inputs.append(x_test_array[:, :, 1])
                all_pred_results.append(pred_test)
                all_y_test.append(y_test)

    if save_test_path:
        res = {'predictions': all_pred_results, 'ground_truth': all_y_test, 'inputs': all_inputs}
        with open(save_test_path + 'res.pickle', 'wb') as f:
            pickle.dump(res, f)
        print(f'>> Test results saved to {save_test_path}')

    acc_test = np.mean(acc_test_list)
    mrr_test = np.mean(mrr_test_list)
    hit_5 = np.mean(hit_5_list)
    hit_10 = np.mean(hit_10_list)
    hit_20 = np.mean(hit_20_list)

    print(f'{acc_test:.3f} {mrr_test:.3f} {hit_5:.3f} {hit_10:.3f} {hit_20:.3f}')






