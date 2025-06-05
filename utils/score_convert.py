import numpy as np

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
    pi_hat = np.zeros(n_nodes)
    for node in range(n_nodes):
        if prop_model == 'SI' and node not in infected_nodes:
            pi_hat[node] = 0
        else:
            pi_hat[node] = pred_prob[node]
    
    # find the smallest pi_hat inside the ground truth
    min_pi_hat = np.inf
    for node in range(n_nodes):
        if ground_truth[node] == 1:
            min_pi_hat = min(min_pi_hat, pi_hat[node])
    
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
    pi_hat = np.zeros(n_nodes)
    for node in range(n_nodes):
        if prop_model == 'SI' and node not in infected_nodes:
            pi_hat[node] = 0
        else:
            pi_hat[node] = pred_prob[node]

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