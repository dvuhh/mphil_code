import os
from shutil import rmtree
import numpy as np
import torch
from sklearn import metrics

# edge AUROC (Area Under the Receiver Operating Characteristic) --> understand discrimination ability
def edge_auroc(pred_edges, true_edges):
    # Handle CPDAG case (Completed Partially Directed Acyclic Graph)
    if true_edges.min() < 0 or true_edges.max() > 1:
        print("Groundtruth is CPDAG")
        true_edges = torch.clamp(true_edges, 0, 1)

    # Convert edge matrices to flat arrays and calculate ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(
        true_edges.reshape(-1).cpu().detach().numpy(),
        pred_edges.reshape(-1).cpu().detach().numpy()
    )

    return metrics.auc(fpr, tpr)

# edge APR (Average Precision-Recall) --> understand precision in edge prediction
# APR is like a more demanding grading system that focuses on precision - how many of our predicted edges are actually correct?
# This is particularly useful when our graph is sparse (has relatively few true edges), which is common in real-world networks.
def edge_apr(pred_edges, true_edges):
    # Handle CPDAG case
    if true_edges.min() < 0 or true_edges.max() > 1:
        print("Groundtruth is CPDAG")
        true_edges = torch.clamp(true_edges, 0, 1)

    # Ensure binary values
    true_edges = torch.clamp(true_edges, 0, 1)

    return metrics.average_precision_score(
        true_edges.reshape(-1).cpu().detach().numpy(),
        pred_edges.reshape(-1).cpu().detach().numpy()
    )


def edge_fn_fp_rev(pred_edges, true_edges,thresh=0.5):
    #added
    # First, convert our probabilistic predictions to binary (0 or 1) using a threshold
    new_pred_edges = torch.zeros_like(pred_edges)
    for i in range(len(pred_edges)):
        for j in range(len(pred_edges[0])):
            if pred_edges[i][j] < thresh:
                new_pred_edges[i][j]=0
            else:
                new_pred_edges[i][j]=1

    diff = true_edges - new_pred_edges

    # Find reversed edges - these are cases where we got the connection right
    # but the direction wrong
    rev = (((diff + diff.T) == 0) & (diff != 0)).sum() / 2.

    # Each reversed edge necessarily leads to one fp and one fn so we need to subtract those
    # Calculate fn (missed edges) and fp (extra edges)
    # We subtract reversed edges because they're counted differently
    fn = (diff == 1).sum() - rev
    fp = (diff == -1).sum() - rev

    return fn.detach().cpu().numpy(), fp.detach().cpu().numpy(), rev.detach().cpu().numpy()

# False Discovery Rate
def edge_fdr(pred_edges, true_edges, thresh=0.5):
    # Binary conversion
    new_pred_edges = torch.zeros_like(pred_edges)
    for i in range(len(pred_edges)):
        for j in range(len(pred_edges[0])):
            if pred_edges[i][j] < thresh:
                new_pred_edges[i][j]=0
            else:
                new_pred_edges[i][j]=1
    diff = true_edges - new_pred_edges
    rev = (((diff + diff.T) == 0) & (diff != 0)).sum() / 2.

    # Each reversed edge necessarily leads to one fp and one fn so we need to subtract those
    fn = (diff == 1).sum() - rev
    fp = (diff == -1).sum() - rev
    tp = ((true_edges == 1) & (diff == 0)).sum()

    return fp/(fp+tp)



#ADDED
def print_prob_true(pred_edges, true_edges):
    print("pred_edges {} , true_edges {}".format(pred_edges,true_edges))

def accuracy_edges(pred_edges, true_edges, thresh=0.5):
    # Create a new tensor for binary predictions
    new_pred_edges = torch.zeros_like(pred_edges)

    # Handle CPDAG case by clamping values to 0 or 1
    if true_edges.min() < 0 or true_edges.max() > 1:
        print("Groundtruth is CPDAG")
        true_edges = torch.clamp(true_edges, 0, 1)
    true_edges = torch.clamp(true_edges, 0, 1)

    # Convert probabilistic predictions to binary using threshold
    for i in range(len(pred_edges)):
        for j in range(len(pred_edges[0])):
            if pred_edges[i][j] < thresh:
                new_pred_edges[i][j]=0
            else:
                new_pred_edges[i][j]=1

    return metrics.accuracy_score(
        new_pred_edges.reshape(-1).cpu().detach().numpy(),
        true_edges.reshape(-1).cpu().detach().numpy()
    )


# We can also compute the Matthews correlation coefficient (MCC)
# MCC takes value between -1 and 1 with 1 being perfect selection and 0 being random guess.
# works well even when our classes are imbalanced (which is common in graphs, where we typically have many more non-edges than edges).
def edge_mcc(pred_edges, true_edges,thresh=0.5):

    new_pred_edges = torch.zeros_like(pred_edges)
    if true_edges.min() < 0 or true_edges.max() > 1:
        print("Groundtruth is CPDAG")
        true_edges = torch.clamp(true_edges, 0, 1)
    true_edges = torch.clamp(true_edges, 0, 1)

    # Convert to binary predictions
    for i in range(len(pred_edges)):
        for j in range(len(pred_edges[0])):
            if pred_edges[i][j] < thresh:
                new_pred_edges[i][j]=0
            else:
                new_pred_edges[i][j]=1

    return metrics.matthews_corrcoef(
        new_pred_edges.reshape(-1).cpu().detach().numpy(),
        true_edges.reshape(-1).cpu().detach().numpy()
    )

# ADDED (DS)
def structural_hamming_distance(pred_edges, true_edges, thresh=0.5):
    """
    - Calculate structural hamming distance (SDH) between predicted and true graph structures.
    - SDH counts the number of edge operations (additions, deletions, or reversals) needed to transform the pred graph into true graph

    Parameters:
    -----------
    pred_edges : torch.Tensor
        Tensor of predicted edge probabilities
    true_edges : torch.Tensor
        Tensor of ground truth edges (0/1 or CPDAG values)
    thresh : float, default=0.5
        Threshold for converting probabilities to binary predictions

    Returns:
    --------
    float
        The structural hamming distance
    """

    new_pred_edges = torch.zeros_like(pred_edges)

    if true_edges.min() < 0 or true_edges.max() > 1:
        print("Groundtruth is CPDAG")
        true_edges = torch.clamp(true_edges, 0, 1)

    # ensure true_edges is binary
    true_edges = torch.clamp(true_edges, 0, 1)

    for i in range(len(pred_edges)):
        for j in range(len(pred_edges[0])):
            if pred_edges[i][j] < thresh:
                new_pred_edges[i][j] = 0
            else:
                new_pred_edges[i][j] = 1

    diff = true_edges - new_pred_edges
    missing_edges = (diff == 1).sum().item()    # FN
    extra_edges = (diff == -1).sum().item()     # FP
    print("Missing edges at:", np.argwhere(diff == 1))
    print("Extra edges at:", np.argwhere(diff == -1))

    reversed_edges = 0
    for i in range(len(pred_edges)):
        for j in range(i + 1, len(pred_edges)): # only check upper triangle, to avoid double counting
            if (diff[i, j] == 1 and diff[j, i] == -1) or (diff[i, j] == -1 and diff[j, i] == 1):
                reversed_edges += 1

    shd_score = missing_edges + extra_edges - 2 * reversed_edges + reversed_edges

    return shd_score

