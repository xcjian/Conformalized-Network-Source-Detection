import numpy as np
from utils.functions import *

def nodewise_APS_score(pred_prob, ground_truth, infected_nodes, prop_model = 'SIR'):
    """
    This function computes the nodewise APS score.

    Args:
    pred_prob: (n_nodes x 2) array of prediction probabilities that each node is the source or not.
    ground_truth: (n_nodes) one-hot array of ground truth source vertices.
    infected_nodes: (int list) a list of infected nodes in the first observed snapshot.

    Returns:
    APS: (n_nodes) array of non-conformity score for each vertex.
    """
    n_node = pred_prob.shape[0]

    if prop_model == 'SI':
        non_infected_nodes = np.setdiff1d(np.arange(n_node), infected_nodes)
        pred_prob[non_infected_nodes, 0] = 1
        pred_prob[non_infected_nodes, 1] = 0

    APS = np.zeros(n_node)
    for node_ in range(n_node):
        pred_prob_ = pred_prob[node_, :]
        ground_truth_ = ground_truth[node_]
        APS[node_] = APS_score(pred_prob_, ground_truth_)
            
    return APS

def APS_score(pred_score, ground_truth):
    '''
    Args:
    pred_score: list of float, predicted scores of nodes
    ground_truth: int, index of ground truth node

    Returns:
    APS: float, APS score.
    '''

    pred_score_array = np.array(pred_score)
    # Compute APS score
    
    ## indices of the nodes with score > this node's score
    indices = np.where(pred_score_array > pred_score_array[ground_truth])[0]

    ## sum of the scores > ground truth's score
    sum_scores = np.sum(pred_score_array[indices])

    ## add random variable
    u = np.random.uniform(0, 1)

    ## compute APS score
    APS = sum_scores + u * pred_score_array[ground_truth]
    ## Note that in this case, small score implies good prediction.

    return APS

def APS_score_SI(pred_score, infected_nodes, ground_truth):
    '''
    This function computes the APS score for the SI model.
    It encodes the prior information that the source node must belong to the infected nodes.

    Args:
    pred_score: list of float, predicted scores of nodes
    infected_nodes: list of int, indices of infected nodes at the earlist available time step
    ground_truth: int, index of ground truth node

    Returns:
    APS: float, APS score.
    '''

    pred_score_array = np.zeros(len(pred_score))

    # Compute conditional APS score
    for node in infected_nodes:
        pred_score_array[node] = pred_score[node]
    pred_score_array = pred_score_array / np.sum(pred_score_array)

    # Compute APS score
    APS = APS_score(pred_score_array, ground_truth)
    
    return APS

def avg_score(pred_prob, ground_truth, prop_model, infected_nodes):
    '''
    This function computes the proposed score for a single sample.

    Args:
    pred_prob: list of float, predicted scores of nodes, representing the probability of belonging to the initial infected nodes
    ground_truth: list of 0 and 1, one-hot vector of ground truth node
    prop_model: str, propagation model used
    infected_nodes: list of int, indices of infected nodes at the earlist available time step

    Returns:
    Average score: float.
    '''

    n_nodes = len(pred_prob)

    # adjust the pred_prob accoding to the propagation model
    pi_hat = np.array(pred_prob)

    indicator_vec = np.zeros(n_nodes)
    indicator_vec[infected_nodes] = 1

    pi_hat = pi_hat * indicator_vec
    # pi_hat = np.zeros(n_nodes)
    # for node in range(n_nodes):
    #     if prop_model == 'SI' and node not in infected_nodes:
    #         pi_hat[node] = 0
    #     else:
    #         pi_hat[node] = pred_prob[node]
    
    # find the smallest pi_hat inside the ground truth
    gtsource = np.nonzero(ground_truth)[0]
    min_pi_hat = np.min(pi_hat[gtsource])
    # min_pi_hat = np.inf
    # for node in range(n_nodes):
    #     if ground_truth[node] == 1:
    #         min_pi_hat = min(min_pi_hat, pi_hat[node])
    
    # find all the nodes with pi_hat >= min_pi_hat
    indices = np.where(pi_hat >= min_pi_hat)[0]

    # compute the average score inside the indices
    pred_score = np.mean(pred_prob[indices])
    pred_score = - pred_score # Note that in this case, small score implies good prediction.
    
    return pred_score

def avg_score_gtunknown(pred_prob, prop_model, infected_nodes):
    '''
    This function computes the proposed score for a single sample, when ground truth label is unknown.
    For example, on the test set, scores for all labels need to be computed. Then this function will provide faster results than avg_score.

    Args:
    pred_prob: list of float, predicted scores of nodes, representing the probability of belonging to the initial infected nodes
    prop_model: str, propagation model used
    infected_nodes: list of int, indices of infected nodes at the earlist available time step

    Returns:
    Average score: float.
    '''

    n_nodes = len(pred_prob)

    # adjust the pred_prob accoding to the propagation model
    pi_hat = np.array(pred_prob)

    indicator_vec = np.zeros(n_nodes)
    indicator_vec[infected_nodes] = 1

    pi_hat = pi_hat * indicator_vec
    # pi_hat = np.zeros(n_nodes)
    # for node in range(n_nodes):
    #     if prop_model == 'SI' and node not in infected_nodes:
    #         pi_hat[node] = 0
    #     else:
    #         pi_hat[node] = pred_prob[node]

    # compute conformity score for all labels

    ## sort all probabilities in descending order
    sorted_indices = np.argsort(-pi_hat)
    sorted_pi_hat = pi_hat[sorted_indices]

    ## compute scores
    pred_scores = np.zeros(n_nodes)
    probability_sum = 0
    for sort_idx_ in range(n_nodes):

        node_ = sorted_indices[sort_idx_]
        # compute the average score for the upper set
        probability_sum = probability_sum + sorted_pi_hat[sort_idx_]
        score_ = probability_sum / (sort_idx_ + 1)

        pred_scores[node_] = score_

    pred_scores = - pred_scores # Note that in this case, small score implies good prediction.
    
    return pred_scores

def set_truncate(set_onehot, prob, pow):
    """
    This function truncate the input set containing the largest pow proportion probabilites.

    Args:
    set_onehot: the one-hot vector representing the original set that you want to truncate.
    prob: the predicted probability that this entry should be included.
    pow: the proportion of elements you want to retain in the truncated set.

    Returns:
    the one-hot vector for the truncated set.
    """

    set_origin = np.nonzero(np.array(set_onehot))[0]
    # prob_quantile = np.quantile(prob[set_origin], 1 - pow)
    prob_quantile = -cpquantile(-prob[set_origin], pow)

    set_truncated = np.array(set_onehot)
    set_truncated[np.where(prob < prob_quantile)[0]] = 0

    return set_truncated

def recall_score(pred_prob, ground_truth, prop_model, infected_nodes):
    '''
    This function computes the proposed score for a single sample.

    Args:
    pred_prob: list of float, predicted scores of nodes, representing the probability of belonging to the initial infected nodes
    ground_truth: list of 0 and 1, one-hot vector of ground truth node
    prop_model: str, propagation model used
    infected_nodes: list of int, indices of infected nodes at the earlist available time step

    Returns:
    Average score: float.
    '''

    n_nodes = len(pred_prob)

    # adjust the pred_prob accoding to the propagation model
    pi_hat = np.array(pred_prob)

    indicator_vec = np.zeros(n_nodes)
    indicator_vec[infected_nodes] = 1

    pi_hat = pi_hat * indicator_vec
    # pi_hat = np.zeros(n_nodes)
    # for node in range(n_nodes):
    #     if prop_model == 'SI' and node not in infected_nodes:
    #         pi_hat[node] = 0
    #     else:
    #         pi_hat[node] = pred_prob[node]
    
    # find the smallest pi_hat inside the ground truth
    gt_indices = np.where(ground_truth > 0)[0]
    min_pi_hat = np.min(pi_hat[gt_indices])
    
    # find all the nodes with pi_hat >= min_pi_hat
    indices = np.where(pi_hat >= min_pi_hat)[0]

    # compute the sum of probability inside the indices
    pred_score = np.sum(pi_hat[indices]) / np.sum(pi_hat)
    
    return pred_score


def recall_score_gtunknown(pred_prob, prop_model, infected_nodes):
    '''
    This function computes the proposed score for a single sample, when ground truth label is unknown.
    For example, on the test set, scores for all labels need to be computed. Then this function will provide faster results than avg_score.

    Args:
    pred_prob: list of float, predicted scores of nodes, representing the probability of belonging to the initial infected nodes
    prop_model: str, propagation model used
    infected_nodes: list of int, indices of infected nodes at the earlist available time step

    Returns:
    Average score: float.
    '''

    n_nodes = len(pred_prob)

    # adjust the pred_prob accoding to the propagation model
    pi_hat = np.array(pred_prob)

    indicator_vec = np.zeros(n_nodes)
    indicator_vec[infected_nodes] = 1

    pi_hat = pi_hat * indicator_vec
    # pi_hat = np.zeros(n_nodes)
    # for node in range(n_nodes):
    #     if prop_model == 'SI' and node not in infected_nodes:
    #         pi_hat[node] = 0
    #     else:
    #         pi_hat[node] = pred_prob[node]

    # compute conformity score for all labels

    ## sort all probabilities in descending order
    sorted_indices = np.argsort(-pi_hat)
    sorted_pi_hat = pi_hat[sorted_indices]

    ## compute scores
    pred_scores = np.zeros(n_nodes)
    probability_sum = 0
    total_probability = np.sum(pi_hat)
    for sort_idx_ in range(n_nodes):

        node_ = sorted_indices[sort_idx_]
        # compute the sum of probability for the upper set
        probability_sum = probability_sum + sorted_pi_hat[sort_idx_]
        score_ = probability_sum / total_probability

        pred_scores[node_] = score_
    
    return pred_scores

def F1_comb_score(pred_prob, ground_truth, prop_model, infected_nodes):
    '''
    This function computes the proposed score for a single sample.

    Args:
    pred_prob: list of float, predicted scores of nodes, representing the probability of belonging to the initial infected nodes
    ground_truth: list of 0 and 1, one-hot vector of ground truth node
    prop_model: str, propagation model used
    infected_nodes: list of int, indices of infected nodes at the earlist available time step

    Returns:
    Average score: float.
    '''

    rec_score = recall_score(pred_prob, ground_truth, prop_model, infected_nodes)
    prec_score = avg_score(pred_prob, ground_truth, prop_model, infected_nodes)

    lam = 0.8
    comb_score = rec_score * lam + prec_score * (1 - lam)

    return comb_score

def F1_comb_score_gtunknown(pred_prob, prop_model, infected_nodes):
    '''
    This function computes the proposed score for a single sample, when ground truth label is unknown.
    For example, on the test set, scores for all labels need to be computed. Then this function will provide faster results than avg_score.

    Args:
    pred_prob: list of float, predicted scores of nodes, representing the probability of belonging to the initial infected nodes
    prop_model: str, propagation model used
    infected_nodes: list of int, indices of infected nodes at the earlist available time step

    Returns:
    Average score: float.
    '''

    rec_score = recall_score_gtunknown(pred_prob, prop_model, infected_nodes)
    prec_score = avg_score_gtunknown(pred_prob, prop_model, infected_nodes)
    
    lam = 0.8
    comb_score = rec_score * lam + prec_score * (1 - lam)

    return comb_score

def scoring_func(feature, coefficients, intercepts):
    """
    This function computes the "scoring function" in the baseline "knowing what to know" paper.
    Strictly speaking, this is not a conformity score.
    Args:
    feature: (n_node x feature-dim)
    coefficients: (n_node x feature-dim)

    Returns:
    scoring_vals: (n_node)
    """

    scoring_vals = np.sum(feature * coefficients, axis=1) + intercepts

    return scoring_vals

def PGMscore(Y, S, edges, alpha, beta):
    """
    This function computes the score of any given set, represented by Y.

    Arguments:
    Y: a n_label array with {-1, 1} entries.
    S: a n_label array with non-conformity scores.
    edges, alpha, beta: the maximum spanning tree and associated parameters learned by PGMTree.

    Return:
    score: the score of the subset.
    """

    n_node = len(Y)

    # summation over the edges
    edge_score = 0
    for edge_ in edges:

        beta_ = beta[edge_]
        l_ = edge_[0]
        k_ = edge_[1]
        Y_l_ = Y[l_]
        Y_k_ = Y[k_]

        psi_ = psi_func(Y_l_, Y_k_)
        edge_score = edge_score + beta_ @ psi_
    
    # summation over the nodes
    node_score = 0
    for node_ in range(n_node):

        alpha_ = alpha[node_]
        Y_k_ = Y[node_]
        S_k_ = S[node_]

        phi_ = phi_func(Y_k_, S_k_)
        node_score = node_score + alpha_ @ phi_
    
    score = edge_score + node_score

    return score

def ArbiTreescore(Y, Y_hat, edges, alpha, beta):
    """
    This function computes the score of any given set, represented by Y.

    Arguments:
    Y: a n_label array with {-1, 1} entries.
    Y_hat: a n_label array with {-1, 1} or real entries.
    edges, alpha, beta: the maximum spanning tree and associated parameters learned by ArbiTree.

    Return:
    score: the score of the subset.
    """

    n_node = len(Y)

    # summation over the edges
    edge_score = 0
    for edge_ in edges:

        beta_ = beta[edge_]
        l_ = edge_[0]
        k_ = edge_[1]
        Y_l_ = Y[l_]
        Y_k_ = Y[k_]

        edge_score = edge_score + beta_ * Y_l_ * Y_k_

    # summation over the nodes
    node_score = 0
    for node_ in range(n_node):

        alpha_ = alpha[node_]
        Y_k_ = Y[node_]
        Y_hat_k_ = Y_hat[node_]

        node_score = node_score + alpha_ * Y_k_ * Y_hat_k_
    
    score = edge_score + node_score

    return score