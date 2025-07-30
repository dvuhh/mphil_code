import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
from src.probabilistic_dag_model.soft_sort import SoftSort_p1, gumbel_sinkhorn

# ------------------------------------------------------------------------------

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class ProbabilisticDAG(nn.Module):

    def __init__(self, n_nodes, temperature=1.0, hard=True, order_type='sinkhorn', noise_factor=1.0, initial_adj=None, lr=1e-3, seed=0):
        """Base Class for Probabilistic DAG Generator based on topological order sampling

        Args:
            n_nodes (int): Number of nodes
            temperature (float, optional): Temperature parameter for order sampling. Defaults to 1.0. Controls the sharpness of the sampling (higher = more random)
            hard (bool, optional): If True output hard (discrete) DAG. Defaults to True.
            order_type (string, optional): Type of differentiable sorting (Method for learning node ordering). Defaults to 'sinkhorn'.
            noise_factor (float, optional): Noise factor for Sinkhorn sorting. Defaults to 1.0.
            initial_adj (torch.tensor, optional): Initial binary adjacency matrix from e.g. PNS.
                Edges with value 0 will not be learnt in further process Defaults to None.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            seed (int, optional): Random seed. Defaults to 0.
        """
        super().__init__()

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.n_nodes = n_nodes
        # self.temperature =
        self.temperature = max(0.1, temperature) # prevent too small values
        self.hard = hard
        self.order_type = order_type

        # Mask for ordering (creates an upper triangular mask to ensure acyclicity)
        self.mask = torch.triu(torch.ones(self.n_nodes, self.n_nodes, device=device), 1)

        # define initial parameters
        if self.order_type == 'sinkhorn':
            self.noise_factor = noise_factor
            p = torch.zeros(n_nodes, n_nodes, requires_grad=True, device=device)
            self.perm_weights = torch.nn.Parameter(p)
        elif self.order_type == 'topk':
            p = torch.zeros(n_nodes, requires_grad=True, device=device)
            self.perm_weights = torch.nn.Parameter(p)
            self.sort = SoftSort_p1(hard=self.hard, tau=self.temperature)
        else:
            raise NotImplementedError

        e = torch.zeros(n_nodes, n_nodes, requires_grad=True, device=device)
        # torch.nn.init.uniform_(e)
        torch.nn.init.xavier_uniform_(e) # use Xavier initialization instead of uniform, better init for training stability

        if initial_adj is not None:
            initial_adj = initial_adj.to(device)
            zero_indices = (1 - initial_adj).bool()
            # set masked edges to zero probability
            e.requires_grad = False
            e[zero_indices] = -300
            e.requires_grad = True

        e = torch.zeros(n_nodes, n_nodes, requires_grad=True, device=device) # modified (n_nodes instead of 10)
        clone=e.clone() # added
        torch.diagonal(clone).fill_(-300) # remplaced e by clone, edge parameters to prevent self-loops (-300 on diagonal)
        self.edge_log_params = torch.nn.Parameter(clone) # remplaced e by clone

        if initial_adj is not None:
            self.edge_log_params.register_hook(lambda grad: grad * initial_adj.float())

        # optimizer setup
        self.lr = lr
        # Adam optimizer for parameter updates
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr) #initializes an Adam optimizer to update the model's trainable parameters during training (perm_weights p and edge_log_params e, learning rate)
        #added for negative binomial (dispersion parameter)
        #disp = torch.zeros(n_nodes, requires_grad=True, device=device)
        #self.r = torch.nn.Parameter(disp)

    # Generates the graph edges using Gumbel-Softmax for differentiable sampling of discrete structures
    def sample_edges(self):
        # Add clipping to prevent extreme values in edge_log_params
        clipped_params = torch.clamp(self.edge_log_params, min = -100, max = 100)
        p_log = F.logsigmoid(torch.stack((clipped_params, -clipped_params)))
        dag = gumbel_softmax(p_log, hard=True, dim=0)[0]
        # p_log = F.logsigmoid(torch.stack((self.edge_log_params, -self.edge_log_params)))
        ##p_log = F.tanh(torch.stack((self.edge_log_params, -self.edge_log_params)))
        # dag = gumbel_softmax(p_log, hard=True, dim=0)[0]
        return dag

    # Samples a permutation matrix representing the topological ordering, to maintain acyclicity
    def sample_permutation(self):
        if self.order_type == 'sinkhorn':
            clipped_weights = torch.clamp(self.perm_weights, min=-100, max=100)
            log_alpha = F.logsigmoid(clipped_weights)
            # log_alpha = F.logsigmoid(self.perm_weights)
            #log_alpha = F.tanh(self.perm_weights)
            P, _ = gumbel_sinkhorn(log_alpha, noise_factor=self.noise_factor, temp=self.temperature, hard=self.hard)
            P = P.squeeze().to(device)
        elif self.order_type == 'topk':
            logits = F.log_softmax(self.perm_weights, dim=0).view(1, -1)
            gumbels = -torch.empty_like(logits).exponential_().log()
            gumbels = (logits + gumbels) / 1
            P = self.sort(gumbels)
            P = P.squeeze()
        else:
            raise NotImplementedError
        return P

    def sample(self):
        P = self.sample_permutation()
        P_inv = P.transpose(0, 1)
        dag_adj = self.sample_edges()
        dag_adj = dag_adj * torch.matmul(torch.matmul(P_inv, self.mask), P)  # apply autoregressive masking
        return dag_adj

    def log_prob(self, dag_adj):
        raise NotImplementedError

    def deterministic_permutation(self, hard=True):
        if self.order_type == 'sinkhorn':
            clipped_weights = torch.clamp(self.perm_weights, min=-100, max=100)
            log_alpha = F.logsigmoid(clipped_weights)
            # log_alpha = F.logsigmoid(self.perm_weights)
            #log_alpha = F.tanh(self.perm_weights)
            P, _ = gumbel_sinkhorn(log_alpha, temp=self.temperature, hard=hard, noise_factor=0)
            P = P.squeeze().to(device)
        elif self.order_type == 'topk':
            sort = SoftSort_p1(hard=hard, tau=self.temperature)
            P = sort(self.perm_weights.detach().view(1, -1))
            P = P.squeeze()
        return P

    def get_threshold_mask(self, threshold):
        P = self.deterministic_permutation()
        P_inv = P.transpose(0, 1)
        dag = (torch.sigmoid(self.edge_log_params.detach()) > threshold).float()
        #dag = (torch.tanh(self.edge_log_params.detach()) > threshold).float()
        dag = dag * torch.matmul(torch.matmul(P_inv, self.mask), P)  # apply autoregressive masking
        return dag

    def get_prob_mask(self):
        # generates a probabilistic adjacency matrix with autoregressive masking to ensure the result is a valid DAG
        # The result is a matrix where each entry (i,j) represents the probability of an edge from node i to node j in a way that respects the acyclicity constraint.
        P = self.deterministic_permutation()
        P_inv = P.transpose(0, 1)
        e = torch.sigmoid(self.edge_log_params.detach()) #sigmoid function
        #e = torch.tanh(self.edge_log_params.detach()) #tanh function
        e = e * torch.matmul(torch.matmul(P_inv, self.mask), P)  # apply autoregressive masking
        return e

    def print_parameters(self, prob=True):
        print('Permutation Weights')
        print(torch.sigmoid(self.perm_weights) if prob else self.perm_weights)
        #print(torch.tanh(self.perm_weights) if prob else self.perm_weights)
        print('Edge Probs')
        print(torch.sigmoid(self.edge_log_params) if prob else self.edge_log_params)
        #print(torch.tanh(self.edge_log_params) if prob else self.edge_log_params)
