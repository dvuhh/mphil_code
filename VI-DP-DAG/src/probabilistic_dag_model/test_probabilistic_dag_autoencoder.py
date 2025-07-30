import torch
from torch import nn
import numpy as np
from src.metrics.dag_metrics import edge_auroc, edge_apr, edge_fn_fp_rev, edge_mcc, edge_fdr, print_prob_true, accuracy_edges

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


# def test_autoencoder(model, true_dag_adj, train_loader, test_loader, result_path, seed_dataset):
#     model.eval()
#
#     # Structure metrics
#     if model.pd_initial_adj is None: # DAG is learned
#         prob_mask = model.probabilistic_dag.get_prob_mask()
#     else: # DAG is fixed
#         prob_mask = model.pd_initial_adj
#
#     # DAG learning
#     metrics = {'undirected_edge_auroc': edge_auroc(pred_edges=prob_mask + prob_mask.T, true_edges=true_dag_adj + true_dag_adj.T),
#                'undirected_edge_apr': edge_apr(pred_edges=prob_mask + prob_mask.T, true_edges=true_dag_adj + true_dag_adj.T),
#                'reverse_edge_auroc': edge_auroc(pred_edges=prob_mask, true_edges=true_dag_adj.T),
#                'reverse_edge_apr': edge_apr(pred_edges=prob_mask, true_edges=true_dag_adj.T),
#                'edge_auroc': edge_auroc(pred_edges=prob_mask, true_edges=true_dag_adj),
#                'edge_apr': edge_apr(pred_edges=prob_mask, true_edges=true_dag_adj),
#                'edge_mcc': edge_mcc(pred_edges=prob_mask, true_edges=true_dag_adj), #added
#                'undirected_edge_mcc': edge_mcc(pred_edges=prob_mask + prob_mask.T, true_edges=true_dag_adj + true_dag_adj.T), #added
#                'edge_fn_fp_rev': edge_fn_fp_rev(pred_edges=prob_mask, true_edges=true_dag_adj), #added
#                'edge_fdr': edge_fdr(pred_edges=prob_mask, true_edges=true_dag_adj), #added
#                #'print_prob_true': print_prob_true(pred_edges=prob_mask, true_edges=true_dag_adj), #added
#                #'print_undirected_prob_true': print_prob_true(pred_edges=prob_mask + prob_mask.T, true_edges=true_dag_adj + true_dag_adj.T), #added
#                'accuracy_edges': accuracy_edges(pred_edges=prob_mask, true_edges=true_dag_adj), #added
#                'accuracy_undirected_edges': accuracy_edges(pred_edges=prob_mask + prob_mask.T, true_edges=true_dag_adj + true_dag_adj.T) #added
#                }
#
#     # Causal mechanims learning
#     # evaluates how well we learned the causal mechanisms
#     with torch.no_grad():
#         for batch_index, (X) in enumerate(test_loader):
#             X = X.to(device)
#             model.update_mask(type='deterministic')
#             X_pred = model(X)
#
#             # Accumulate predictions and true values
#             if batch_index == 0:
#                 X_pred_all = X_pred.reshape(-1).to("cpu")
#                 X_all = X.reshape(-1).to("cpu")
#             else:
#                 X_pred_all = torch.cat([X_pred_all, X_pred.reshape(-1).to("cpu")], dim=0)
#                 X_all = torch.cat([X_all, X.reshape(-1).to("cpu")], dim=0)
#
#         # For Normal Data
#         #reconstruction = ((X_all - X_pred_all)**2).mean().item()
#
#         # For Poisson Data
#         nll = nn.PoissonNLLLoss(reduction='mean')
#         reconstruction = torch.mean(nll(X_pred_all, X_all))
#
#     metrics['reconstruction'] = reconstruction
#     return metrics


def test_autoencoder(model, true_dag_adj, train_loader, test_loader, result_path, seed_dataset):
    model.eval()

    # Structure metrics
    if model.pd_initial_adj is None: # DAG is learned
        prob_mask = model.probabilistic_dag.get_prob_mask()
    else: # DAG is fixed
        prob_mask = model.pd_initial_adj

    # DAG learning
    metrics = {'undirected_edge_auroc': edge_auroc(pred_edges=prob_mask + prob_mask.T, true_edges=true_dag_adj + true_dag_adj.T),
               'undirected_edge_apr': edge_apr(pred_edges=prob_mask + prob_mask.T, true_edges=true_dag_adj + true_dag_adj.T),
               'reverse_edge_auroc': edge_auroc(pred_edges=prob_mask, true_edges=true_dag_adj.T),
               'reverse_edge_apr': edge_apr(pred_edges=prob_mask, true_edges=true_dag_adj.T),
               'edge_auroc': edge_auroc(pred_edges=prob_mask, true_edges=true_dag_adj),
               'edge_apr': edge_apr(pred_edges=prob_mask, true_edges=true_dag_adj),
               'edge_mcc': edge_mcc(pred_edges=prob_mask, true_edges=true_dag_adj), #added
               'undirected_edge_mcc': edge_mcc(pred_edges=prob_mask + prob_mask.T, true_edges=true_dag_adj + true_dag_adj.T), #added
               'edge_fn_fp_rev': edge_fn_fp_rev(pred_edges=prob_mask, true_edges=true_dag_adj), #added
               'edge_fdr': edge_fdr(pred_edges=prob_mask, true_edges=true_dag_adj), #added
               #'print_prob_true': print_prob_true(pred_edges=prob_mask, true_edges=true_dag_adj), #added
               #'print_undirected_prob_true': print_prob_true(pred_edges=prob_mask + prob_mask.T, true_edges=true_dag_adj + true_dag_adj.T), #added
               'accuracy_edges': accuracy_edges(pred_edges=prob_mask, true_edges=true_dag_adj), #added
               'accuracy_undirected_edges': accuracy_edges(pred_edges=prob_mask + prob_mask.T, true_edges=true_dag_adj + true_dag_adj.T) #added
               }

    # Causal mechanims learning
    # evaluates how well we learned the causal mechanisms
    total_reconstruction_metric = 0.0
    total_samples = 0
    nll = nn.PoissonNLLLoss(reduction='sum')    # use 'sum' to accumulate correctly, then average at the end

    with torch.no_grad():
        for X in test_loader:
            X = X.to(device)
            model.update_mask(type='deterministic')
            X_pred = model(X)

            batch_reconstruction = nll(X_pred.reshape(-1), X.reshape(-1)).item() # getting scalar sum
            total_reconstruction_metric += batch_reconstruction
            total_samples += X.numel()

    avg_reconstruction = total_reconstruction_metric / total_samples if total_samples > 0 else 0
    metrics['reconstruction'] = avg_reconstruction

    return metrics