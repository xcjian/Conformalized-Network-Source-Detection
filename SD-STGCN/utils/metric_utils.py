# ---------------------
# evaluation metrics
# -----------------------


import pickle
import numpy as np
from sklearn.metrics import accuracy_score, ndcg_score


# accuracy
def batch_acc(y_pred, y):
    '''
    y_pred: prediction, shape: [batch_size, num_node]
    y: true label, shape: [batch_size,]
    return: fraction of correctly classified samples in one batch
    '''
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    y_pred_ = np.argmax(y_pred,axis=1)
    return accuracy_score(y,y_pred_,normalize=True)

def batch_acc_nodewise(pred_batch, y_batch):
    '''
    Compute node-wise precision, recall, F1 for a batch.
    pred_batch: predicted softmax outputs, shape [batch_size, n_node, 2]
    y_batch: true labels, shape [batch_size, n_node], values {0, 1}
    
    Returns:
        f1, precision, recall
    '''
    pred_labels = np.argmax(pred_batch, axis=-1)  # [batch_size, n_node]

    # Flatten everything
    pred_labels = pred_labels.flatten()
    y_batch = y_batch.flatten()

    true_positives = np.sum((pred_labels == 1) & (y_batch == 1))
    false_positives = np.sum((pred_labels == 1) & (y_batch == 0))
    false_negatives = np.sum((pred_labels == 0) & (y_batch == 1))

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # return f1, precision, recall
    return f1

def batch_prec_nodewise(pred_batch, y_batch):
    '''
    Compute node-wise precision for a batch.
    pred_batch: predicted softmax outputs, shape [batch_size, n_node, 2]
    y_batch: true labels, shape [batch_size, n_node], values {0, 1}
    
    Returns:
        precision
    '''
    pred_labels = np.argmax(pred_batch, axis=-1)  # [batch_size, n_node]

    # Flatten everything
    pred_labels = pred_labels.flatten()
    y_batch = y_batch.flatten()

    true_positives = np.sum((pred_labels == 1) & (y_batch == 1))
    false_positives = np.sum((pred_labels == 1) & (y_batch == 0))
    false_negatives = np.sum((pred_labels == 0) & (y_batch == 1))

    precision = true_positives / (true_positives + false_positives + 1e-8)

    # return precision
    return precision

def batch_recall_nodewise(pred_batch, y_batch):
    '''
    Compute node-wise recall for a batch.
    pred_batch: predicted softmax outputs, shape [batch_size, n_node, 2]
    y_batch: true labels, shape [batch_size, n_node], values {0, 1}
    
    Returns:
        recall
    '''
    pred_labels = np.argmax(pred_batch, axis=-1)  # [batch_size, n_node]

    # Flatten everything
    pred_labels = pred_labels.flatten()
    y_batch = y_batch.flatten()

    true_positives = np.sum((pred_labels == 1) & (y_batch == 1))
    false_positives = np.sum((pred_labels == 1) & (y_batch == 0))
    false_negatives = np.sum((pred_labels == 0) & (y_batch == 1))

    recall = true_positives / (true_positives + false_negatives + 1e-8)

    # return recall
    return recall

# mean reciprocal rank
def batch_mrr(y_pred, y):
    '''
    y_pred: prediction, shape: [batch_size, num_node]
    y: true label, shape: [batch_size,]
    return: mean reciprocal rank
    '''
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    mrr = []
    for i,x in enumerate(y_pred):
        x_ = 1.+sorted(range(len(x)), key=lambda a: -x[a]).index(y[i])
        mrr.append(1./x_)

    return np.mean(mrr)

def hit_at(pred, y, top):
    n = len(pred)
    res = []
    for i in range(n):
        p = pred[i]
        y_ = y[i]
        top_ind = np.argsort(p)[::-1][:top]
        if y_ in top_ind:
            res.append(1.0)
        else:
            res.append(0.0)

    return res

# return indices of top predictions
def get_top_idx(pred, top):
    n = len(pred)
    res = []
    for i in range(n):
        top_ind = np.argsort(pred[i])[::-1][:top]
        res.append(top_ind)

    return res

# return mrr of sources
def get_src_mrr(pred, src):
    '''
    pred: list of list of probabilities
    src: list of list of sources
    '''
    res = []
    n = len(pred)
    for i in range(n):
        p = pred[i]
        s = src[i]
        rank = np.argsort(p)[::-1]
        tmp = []
        for ss in s:
            idx = np.argwhere(rank == ss)[0,0]+1
            tmp.append(1./idx)

        res.append(np.mean(tmp))

    return res

# jaccard similarity
def jaccard_similarity(x,y):
    '''
    x: list
    y: list
    '''
    intercept = set([a for a in x if a in y])
    union = set()
    for a in x:
        union.add(a)
    for b in y:
        union.add(b)

    return len(intercept)*1./len(union)


# ndcg (normalized discounted cumulative gain)
def ndcg(y_true, y_pred):
    return ndcg_score(y_true, y_pred)


def mrr_at(pred, y, top):
    n = len(pred)
    res = []
    for i in range(n):
        p = pred[i]
        y_ = y[i]
        top_ind = np.argsort(p)[::-1][:top]
        if y_ in top_ind:
            idx = np.argwhere(top_ind == y_)[0,0]+1
            res.append(1.0/idx)
        else:
            res.append(0.0)

    return res


# get prediction
def get_pred(y_pred):
    '''
    y_pred: prediction, shape: [batch_size, num_node]
    y: true label, shape: [batch_size,]
    return: predicted nodes
    '''
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    y_pred_ = np.argmax(y_pred,axis=1)
    return y_pred_


