from data_loader.data_utils import *
from utils.math_utils import evaluation
from utils.metric_utils import *
import os
from os.path import join as pjoin
# import tensorflow as tf
import torch
import numpy as np
import time
import torch
import os
import pickle
import numpy as np
from torch.nn import functional as F
from models.base_model import STGCN_SI_Nodewise


# def model_test(inputs, args, load_path='./output/models/', save_test_path=None):
#     '''
#     Load and test saved model from the checkpoint.
#     inputs: instance of class Dataset, data source for test.
#     args: instance of class argparse, args for training.
#     load_path: str, the path of loaded model.
#     '''
#     n_frame = args.n_frame
#     num_node = args.n_node
#     n_channel = args.n_channel

#     batch_size = args.batch_size

#     start, end = args.start, args.end

#     random = args.random

#     model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path

#     test_graph = tf.Graph()

#     with test_graph.as_default():
#         saver = tf.compat.v1.train.import_meta_graph(pjoin(f'{model_path}.meta'))

#     all_inputs = []
#     all_pred_results = []
#     all_y_test = []

#     with tf.compat.v1.Session(graph=test_graph) as test_sess:
#         saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
#         print(f'>> Loading saved model from {model_path} ...')

#         pred = test_graph.get_collection('y_pred')[0]


#         acc_test_list = []
#         mrr_test_list = []
#         hit_5_list = []
#         hit_10_list = []
#         hit_20_list = []
#         for (x_test, y_test) in gen_xy_batch(inputs.get_data('test'), batch_size,\
#         dynamic_batch=True, shuffle=False):
#             x_test_ = onehot(iteration2snapshot(x_test, n_frame, start=start, end=end, random=random),n_channel)
#             pred_test = test_sess.run(pred, feed_dict={'data_input:0': x_test_, 'data_label:0': onehot(y_test, num_node), 'keep_prob:0': 1.0})

#             acc_test_list.append(batch_acc(pred_test, y_test))

#             mrr_test_list.append(batch_mrr(pred_test, y_test))

#             hit_5_list += hit_at(pred_test, y_test, 5)
#             hit_10_list += hit_at(pred_test, y_test, 10)
#             hit_20_list += hit_at(pred_test, y_test, 20)

#             if save_test_path:
#                 x_test_array = np.array([list(x_test_[i][0]) for i in range(len(x_test_))])
#                 all_inputs.append(x_test_array[:, :, 1]) # only use the infected status
#                 all_pred_results.append(pred_test)
#                 all_y_test.append(y_test)
        
#         if save_test_path:
#             res = {'predictions': all_pred_results, 'ground_truth': all_y_test, 'inputs': all_inputs}
#             with open(save_test_path + 'res.pickle', 'wb') as f:
#                 pickle.dump(res, f)
#             print(f'>> Test results saved to {save_test_path}')

#         acc_test = np.mean(acc_test_list)
#         mrr_test = np.mean(mrr_test_list)
#         hit_5 = np.mean(hit_5_list)
#         hit_10 = np.mean(hit_10_list)
#         hit_20 = np.mean(hit_20_list)
#         print(f'{acc_test:.3f} {mrr_test:.3f} {hit_5:.3f} {hit_10:.3f} {hit_20:.3f}')


# def model_test_nodewise(inputs, args, load_path='./output/models/', save_test_path=None):
#     '''
#     Load and test saved model from the checkpoint.
#     inputs: instance of class Dataset, data source for test.
#     args: instance of class argparse, args for training.
#     load_path: str, the path of loaded model.
#     '''
#     n_frame = args.n_frame
#     num_node = args.n_node
#     n_channel = args.n_channel

#     batch_size = args.batch_size

#     start, end = args.start, args.end

#     random = args.random

#     model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path

#     test_graph = tf.Graph()

#     with test_graph.as_default():
#         saver = tf.compat.v1.train.import_meta_graph(pjoin(f'{model_path}.meta'))

#     all_inputs = []
#     all_pred_results = []
#     all_y_test = []

#     with tf.compat.v1.Session(graph=test_graph) as test_sess:
#         saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
#         print(f'>> Loading saved model from {model_path} ...')

#         pred = test_graph.get_collection('y_pred')[0]


#         acc_test_list = []
#         prec_test_list = []
#         recall_test_list = []
#         for (x_test, y_test) in gen_xy_batch(inputs.get_data('test'), batch_size,\
#         dynamic_batch=True, shuffle=False):
#             x_test_ = onehot(iteration2snapshot(x_test, n_frame, start=start, end=end, random=random),n_channel)
#             y_test_ = snapshot_to_labels(y_test, num_node)
#             pred_test = test_sess.run(pred, feed_dict={'data_input:0': x_test_, 'data_label:0': y_test_, 'keep_prob:0': 1.0})

#             acc_test_list.append(batch_acc_nodewise(pred_test, y_test_))
#             prec_test_list.append(batch_prec_nodewise(pred_test, y_test_))
#             recall_test_list.append(batch_recall_nodewise(pred_test, y_test_))

#             if save_test_path:
#                 x_test_array = np.array([list(x_test_[i][0]) for i in range(len(x_test_))])
#                 all_inputs.append(x_test_array[:, :, 1]) # only use the infected status
#                 all_pred_results.append(pred_test)
#                 all_y_test.append(y_test_)
        
#         if save_test_path:
#             res = {'predictions': all_pred_results, 'ground_truth': all_y_test, 'inputs': all_inputs}
#             with open(save_test_path + 'res.pickle', 'wb') as f:
#                 pickle.dump(res, f)
#             print(f'>> Test results saved to {save_test_path}')

#         acc_test = np.mean(acc_test_list)
#         prec_test = np.mean(prec_test_list)
#         recall_test = np.mean(recall_test_list)
        
#         print(f'{acc_test:.3f} {prec_test:.3f} {recall_test:.3f}')

def model_test_pytorch_nodewise(inputs, args, load_path='./output/models/', save_test_path=None):

    model_files = [f for f in os.listdir(load_path) if f.endswith('.pt')]
    if not model_files:
        raise FileNotFoundError(f"No PyTorch model found in {load_path}")
    model_files.sort(key=lambda f: os.path.getmtime(os.path.join(load_path, f)))
    model_path = os.path.join(load_path, model_files[-1])
    print(f"Loading model from: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STGCN_SI_Nodewise(
        n_frame=args.n_frame,
        Ks=args.ks,
        Kt=args.kt,
        blocks=args.blocks,
        sconv=args.sconv,
        Lk=inputs.Lk,
        keep_prob=args.keep_prob,
        pos_weight=args.pos_weight,
        n_node=args.n_node
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    acc_list, prec_list, recall_list = [], [], []
    all_pred_results, all_y_test, all_inputs = [], [], []

    n_frame = args.n_frame
    num_node = args.n_node
    n_channel = args.n_channel
    batch_size = args.batch_size
    start, end = args.start, args.end
    random = args.random

    with torch.no_grad():
        for x_batch, y_batch in gen_xy_batch(inputs.get_data('test'), batch_size=batch_size, dynamic_batch=True, shuffle=False):
            for x_test, y_test in zip(x_batch, y_batch):
                x = onehot(iteration2snapshot([x_test], n_frame, start, end, random), n_channel)  # [1, T, N, C]
                y = snapshot_to_labels([y_test], num_node)  # [1, N]

                x_tensor = torch.tensor(x[0], dtype=torch.float32).unsqueeze(0).to(device)  # [1, T, N, C]
                y_tensor = torch.tensor(y[0], dtype=torch.long).unsqueeze(0).to(device)    # [1, N]
                
                logits = model(x_tensor, training=False)  # [1, N, 2]
                prob = F.softmax(logits, dim=-1)
                pred = torch.argmax(prob, dim=-1)  # [1, N]
                prob_np = prob.cpu().numpy()[0]        # [N, 2]
                label_np = y_tensor.cpu().numpy()[0]   # [N]
                infected_np = x_tensor[0, :, :, 1].cpu().numpy()  # [T, N]

                y_true = y_tensor.cpu().numpy().flatten()       # [N]
                y_pred = pred.cpu().numpy().flatten()    # [N]

                tp = np.sum((y_pred == 1) & (y_true == 1))
                fp = np.sum((y_pred == 1) & (y_true == 0))
                fn = np.sum((y_pred == 0) & (y_true == 1))

                prec = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                acc = 2 * prec * recall / (prec + recall + 1e-8)  # F1-score

                all_pred_results.append(prob_np)       # [N, 2]
                all_y_test.append(label_np)            # [N]
                all_inputs.append(infected_np)         # [T, N]

                # acc = batch_acc_nodewise(pred.cpu().numpy(), y_tensor.cpu().numpy())
                # prec = batch_prec_nodewise(pred.cpu().numpy(), y_tensor.cpu().numpy())
                # recall = batch_recall_nodewise(pred.cpu().numpy(), y_tensor.cpu().numpy())
                acc_list.append(acc)
                prec_list.append(prec)
                recall_list.append(recall)

    print("Test F1: {:.4f}".format(np.mean(acc_list)))
    print("Test Prec: {:.4f}".format(np.mean(prec_list)))
    print("Test Recall: {:.4f}".format(np.mean(recall_list)))

    if save_test_path:
        os.makedirs(save_test_path, exist_ok=True)  

    save_file_path = os.path.join(save_test_path, 'res.pickle')  

    res = {
        'predictions': all_pred_results,
        'ground_truth': all_y_test,
        'inputs': all_inputs
    }

    with open(save_file_path, 'wb') as f:
        pickle.dump(res, f)

    print(f"Saved test result to {save_file_path}")


