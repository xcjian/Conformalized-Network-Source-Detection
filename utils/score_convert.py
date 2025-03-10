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