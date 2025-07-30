import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.probabilistic_dag_model.masked_autoencoder import MaskedAutoencoder
from src.probabilistic_dag_model.masked_autoencoder import MaskedAutoencoderFast
from src.probabilistic_dag_model.probabilistic_dag import ProbabilisticDAG

'''
This class combines a ProbabilisticDAG model with a MaskedAutoencoder to learn both the structure and the relationships in a dataset.
'''

masked_autoencoder = {True: MaskedAutoencoderFast,
                      False: MaskedAutoencoder}


class ProbabilisticDAGAutoencoder(nn.Module):
    def __init__(self,
                 # General parameters
                 input_dim,  # Input dimension (i.e. number of input signals). list of ints
                 output_dim,  # Output dimension (i.e. dimension of one input signal). list of ints
                 loss='ELBO',  # Loss name. string
                 regr=0,  # Regularization factor in ELBO loss. float
                 prior_p=.001,  # Regularization factor in ELBO loss. float
                 seed=123,

                 # Mask autoencoder parameters
                 ma_hidden_dims=[64, 64, 64],  # Hidden dimensions. list of ints
                 ma_architecture='linear',  # Encoder architecture name. string
                 ma_lr=1e-3,  # Learning rate. float
                 ma_fast=False,  # Use fast masked autoencoder implementation. Boolean

                 # Probabilistic dag parameters
                 pd_temperature=1.0,  # Temperature for differentiable sorting. int
                 pd_hard=True,  # Hard or soft sorting. boolean
                 pd_order_type='sinkhorn',  # Type of differentiable sorting. string
                 pd_noise_factor=1.0,  # Noise factor for Sinkhorn sorting. int
                 pd_initial_adj=None,  # If None, the adjacency matrix is learned.
                 pd_lr=1e-3):  # Random seed for init. int
        super().__init__()

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.set_default_tensor_type(torch.FloatTensor)

        self.loss, self.regr, self.prior_p = loss, regr, prior_p

        # Autoencoder parameters
        self.mask_autoencoder = masked_autoencoder[ma_fast](input_dim=input_dim,
                                                            output_dim=output_dim,
                                                            hidden_dims=ma_hidden_dims,
                                                            architecture=ma_architecture,
                                                            lr=ma_lr,
                                                            seed=seed)

        # Probabilistic DAG parameters
        if pd_order_type == 'sinkhorn' or pd_order_type == 'topk':
            self.probabilistic_dag = ProbabilisticDAG(n_nodes=input_dim,
                                                      temperature=pd_temperature,
                                                      hard=pd_hard,
                                                      order_type=pd_order_type,
                                                      noise_factor=pd_noise_factor,
                                                      initial_adj=pd_initial_adj,
                                                      lr=pd_lr,
                                                      seed=seed)
        else:
            raise NotImplementedError
        self.pd_initial_adj = pd_initial_adj

    def forward(self, X, compute_loss=True):
        X_pred = self.mask_autoencoder(X)

        # Loss
        if compute_loss:
            if self.loss == 'ELBO':
                self.grad_loss = self.ELBO_loss(X_pred, X)
            else:
                raise NotImplementedError

        return X_pred

    def update_mask(self, type=None, threshold=.5):
        if type == 'deterministic':
            new_mask = self.probabilistic_dag.get_threshold_mask(threshold)
        elif type == 'id':
            new_mask = torch.eye(self.mask_autoencoder.input_dim)
        else:
            new_mask = self.probabilistic_dag.sample()
        if self.pd_initial_adj is not None:  # We do no sample if the DAG adjacency is not learned/specified from start
            new_mask = self.pd_initial_adj.to(next(self.mask_autoencoder.parameters()).device)
        self.mask_autoencoder.mask = new_mask

    def ELBO_loss(self, X_pred, X): #HERE: has to be changed depending on the data distribution

    #Normal
        #loss = nn.MSELoss(reduction='mean')
        #ELBO_loss = loss(X_pred, X)


    #Poisson
        #X_pred = F.softplus(X_pred)
        #loss = nn.PoissonNLLLoss(reduction='mean')
        #ELBO_loss = loss(X_pred, X)
        # or we can rewrite it :
        # X_pred = F.softplus(X_pred)
        ELBO_loss = -torch.mean(X * torch.log(X_pred) - X_pred - torch.lgamma(X + 1))

    #Negative Binomial
        #r=self.probabilistic_dag.r
        #r=F.softplus(r)
        #r=torch.tensor(2)
        #X_pred = F.softplus(X_pred)
        #ELBO_loss= -torch.mean(torch.lgamma(X + r) - torch.lgamma(X + 1) - torch.lgamma(r)+X*torch.log(X_pred)+ r*torch.log(r)-(r+X)*torch.log(X_pred+r))


        # #print to check X, X_pred and ELBO
        # print("X: {}, X_pred:{}".format(X,X_pred))
        FT=ELBO_loss
        # print("first term: {}".format(FT))

        if self.regr > 0:
            kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
            logsigmoid = torch.nn.LogSigmoid()
            ones = torch.ones_like(self.probabilistic_dag.edge_log_params)
            regularizer = kl_loss(logsigmoid(self.probabilistic_dag.edge_log_params), self.prior_p * ones) #KLDivLoss expects logits as first argument
            regularizer += kl_loss(torch.log(ones - torch.sigmoid(self.probabilistic_dag.edge_log_params)), (1 - self.prior_p) * ones)
            ELBO_loss = ELBO_loss + self.regr * regularizer

        # #print to check the ELBO
        #     ST=ELBO_loss - FT
        #     print("second term: {}".format(ST))
        # print("total ELBO: {}".format(ELBO_loss))
        return ELBO_loss

    def step(self):
        # Zero gradients
        self.mask_autoencoder.optimizer.zero_grad()
        if self.pd_initial_adj is None:
            self.probabilistic_dag.optimizer.zero_grad()

        # Backward pass
        self.grad_loss.backward()

        # Update parameters
        self.mask_autoencoder.optimizer.step()
        if self.pd_initial_adj is None:
            self.probabilistic_dag.optimizer.step()
