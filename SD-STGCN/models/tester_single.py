from data_loader.data_utils import *
from utils.math_utils import evaluation
from utils.metric_utils import *
from os.path import join as pjoin
import tensorflow as tf
import numpy as np
import time



def model_test(inputs, args, load_path='./output/models/'):
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

    model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path

    test_graph = tf.Graph()

    with test_graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    with tf.compat.v1.Session(graph=test_graph) as test_sess:
        saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
        print(f'>> Loading saved model from {model_path} ...')

        pred = test_graph.get_collection('y_pred')[0]


        x_test = inputs.get_x('test')
        y_test = inputs.get_y('test')
        x_test_ = onehot(iteration2snapshot(x_test, n_frame, start=start, end=end, random=random),n_channel)
        pred_test = test_sess.run(pred, feed_dict={'data_input:0': x_test_, 'keep_prob:0': 1.0})

        n_src = len(y_test[0]) # num of sources

        top5 = get_top_idx(pred_test, 5)[0] # top 10 idx
        hit5 = np.mean([1. if idx in y_test[0] else 0. for idx in top5])

        top10 = get_top_idx(pred_test, 10)[0] # top 10 idx
        hit10 = np.mean([1. if idx in y_test[0] else 0. for idx in top10])

        top20 = get_top_idx(pred_test, 20)[0] # top 10 idx
        hit20 = np.mean([1. if idx in y_test[0] else 0. for idx in top20])

        topk = get_top_idx(pred_test, n_src)[0] # top k idx, k = n_src
        jc = jaccard_similarity(topk, y_test[0]) # jaccard similarity

        relavent = np.zeros([1,num_node])
        relavent[0,y_test[0]] = 1
        score_ndcg = ndcg(relavent, pred_test)

        print(f'{hit5:.3f} {hit10:.3f} {hit20:.3f} {jc:.3f} {score_ndcg:.3f}')



def model_test_single_nodewise(inputs, args, load_path='./output/models/', save_test_path=None):
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

    model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path

    test_graph = tf.Graph()

    with test_graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    with tf.compat.v1.Session(graph=test_graph) as test_sess:
        saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
        print(f'>> Loading saved model from {model_path} ...')

        pred = test_graph.get_collection('y_pred')[0]
        logits = test_graph.get_collection('logits_pred')[0]

        x_test = inputs.get_x('test')
        y_test = inputs.get_y('test')
        meta_test = inputs.get_z('test')
        x_test_ = onehot(iteration2snapshot(x_test, n_frame, start=meta_test, end=end, random=random),n_channel)
        y_test_ = snapshot_to_labels(y_test, num_node)
        pred_test = test_sess.run(pred, feed_dict={'data_input:0': x_test_, 'keep_prob:0': 1.0})
        logits_test = test_sess.run(logits, feed_dict={'data_input:0': x_test_, 'keep_prob:0': 1.0})

        # Convert predictions to binary labels (0 or 1) by taking argmax over class dimension
        pred_labels = np.argmax(pred_test, axis=-1)  # Shape: (batch_size, num_node)

        # Compute precision, recall, and F1 score
        true_positives = np.sum((pred_labels == 1) & (y_test_ == 1))
        predicted_positives = np.sum(pred_labels == 1)
        actual_positives = np.sum(y_test_ == 1)

        # Precision: TP / (TP + FP)
        prec_test = true_positives / predicted_positives if predicted_positives > 0 else 0.0

        # Recall: TP / (TP + FN)
        recall_test = true_positives / actual_positives if actual_positives > 0 else 0.0

        # F1 Score: 2 * (precision * recall) / (precision + recall)
        acc_test = (2 * prec_test * recall_test) / (prec_test + recall_test) if (prec_test + recall_test) > 0 else 0.0

        if save_test_path:
            x_test_array = np.array([list(x_test_[i][0]) for i in range(len(x_test_))])
            res = {'predictions': [pred_test], 'ground_truth': [y_test_], 'inputs': [x_test_array[:, :, 1]], 'logits': [logits_test]}
            with open(save_test_path + 'res.pickle', 'wb') as f:
                pickle.dump(res, f)
            print(f'>> Test results saved to {save_test_path}')

        print(f'{acc_test:.3f} {prec_test:.3f} {recall_test:.3f}')