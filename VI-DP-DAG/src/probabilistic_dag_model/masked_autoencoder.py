import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.linear_sequential import linear_sequential
from src.architectures.parallel_linear_sequential import parallel_linear_sequential
# ------------------------------------------------------------------------------

'''
Purpose:
    - a NN that learns to predict / reconstruct each variable in your dataset based on a subset of other variables, where the rsp between variables are determined by a mask (representing a DAG structure)
    - purpose: Each autoencoder learns to predict one variable based on some subset of other variables
    - The mask determines which variables can be used to predict each target variable

Key idea:
    - learn and represent causal or structural relationships between variables in your data, 
    - where the mask (often learned by a DAG model) specifies which relationships are allowed.
'''


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class MaskedAutoencoder(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64, 64], architecture='linear', lr=1e-3, seed=0):
        """Base Class for Probabilistic DAG Generator based on topological order sampling.
        - determines which inputs each autoencoder can use for its predictions,
        - enforcing the DAG structure learned by the probabilistic DAG model

        Args:
            input_dim (Tensor int): Number of signals
            output_dim (Tensor int, optional): dimension of one signal (usually 1 if we do not group signals).
            hidden_dims (list int, optional): Hidden dimensions of decoder.
            architecture (string, optional): Architecture of decoder.
            batch_size (int, optional): Batch size.
            lr (float, optional): Learning rate. Defaults to 1e-3
            seed (int, optional): Random seed. Defaults to 0.
        """
        super().__init__()

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.mask = None
        self.input_dim, self.output_dim, self.hidden_dims = input_dim, output_dim, hidden_dims

        # creates a separate autoencoder for each input dimension
        # each autoencoder predicts one dim based on masked inputs
        if architecture == 'linear':
            self.autoencoders = nn.ModuleList(
                [linear_sequential(
                    input_dims=self.input_dim,
                    hidden_dims=self.hidden_dims,
                    output_dim=self.output_dim,
                    k_lipschitz=None
                ) for i in range(self.input_dim)
            ])
        else:
            raise NotImplementedError

        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, X):
        # create masked copies of input
        masked_duplicate_X = X.unsqueeze(1).expand([-1, self.input_dim, -1]) * self.mask.T.unsqueeze(0) # [batch_size, input_dim, input_dim]

        # init output tensor
        pred_X = torch.zeros_like(X) # [batch_size, input_dim]

        # generate predictions for each dim
        for i in range(self.input_dim):
            pred_X[:, i] = self.autoencoders[i](masked_duplicate_X[:, i, :]).squeeze()

        return pred_X


class MaskedAutoencoderNoise(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64, 64], architecture='linear', lr=1e-3, seed=0):
        """Base Class for Probabilistic DAG Generator based on topological order sampling

        Args:
            input_dim (Tensor int): Number of signals
            output_dim (Tensor int, optional): dimension of one signal (usually 1 if we do not group signals).
            hidden_dims (list int, optional): Hidden dimensions of decoder.
            architecture (string, optional): Architecture of decoder.
            batch_size (int, optional): Batch size.
            lr (float, optional): Learning rate. Defaults to 1e-3
            seed (int, optional): Random seed. Defaults to 0.
        """
        super().__init__()

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.mask = None
        self.input_dim, self.output_dim, self.hidden_dims = input_dim, output_dim, hidden_dims

        # each autoencoder processes concatenated data and noise vectors
        if architecture == 'linear':
            self.autoencoders = nn.ModuleList([linear_sequential(
                input_dims=2*self.input_dim,    # double the input dim
                hidden_dims=self.hidden_dims,
                output_dim=self.output_dim,
                k_lipschitz=None
            ) for i in range(self.input_dim)])
        else:
            raise NotImplementedError

        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, X, Z):
        # Mask the input data
        masked_duplicate_X = X.unsqueeze(1).expand([-1, self.input_dim, -1]) * self.mask.T.unsqueeze(0) # [batch_size, input_dim, input_dim]

        # Mask the noise variables with identity matrix
        masked_duplicate_Z = Z.unsqueeze(1).expand([-1, self.input_dim, -1]) * torch.eye(self.input_dim).unsqueeze(0).to(Z.device) # [batch_size, input_dim, input_dim]

        pred_X = torch.zeros_like(X) # [batch_size, input_dim]

        # Generate predictions using concatenated masked data and noise
        for i in range(self.input_dim):
            pred_X[:, i] = self.autoencoders[i](torch.cat((masked_duplicate_X[:, i, :], masked_duplicate_Z[:, i, :]), 1)).squeeze()

        return pred_X


class MaskedAutoencoderFast(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64, 64], architecture='linear', lr=1e-3, seed=0):
        """Base Class for Probabilistic DAG Generator based on topological order sampling
        Key optimizations:
            - No loop over dimensions
            = Single forward pass processes all nodes
            - Uses tensor operations for masking and reshaping

        Args:
            input_dim (Tensor int): Number of signals
            output_dim (Tensor int, optional): dimension of one signal (usually 1 if we do not group signals).
            hidden_dims (list int, optional): Hidden dimensions of decoder.
            architecture (string, optional): Architecture of decoder.
            batch_size (int, optional): Batch size.
            lr (float, optional): Learning rate. Defaults to 1e-3
            seed (int, optional): Random seed. Defaults to 0.
        """

        super().__init__()

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.mask = None
        self.input_dim, self.output_dim, self.hidden_dims = input_dim, output_dim, hidden_dims
        if architecture == 'linear':
            self.autoencoders = parallel_linear_sequential(
                input_dims=self.input_dim,
                hidden_dims=self.hidden_dims,
                output_dim=self.output_dim,
                parallel_dim=self.input_dim)
        else:
            raise NotImplementedError

        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.to(device)

    def forward(self, X):
        # Create masked copies and reshape
        masked_duplicate_X = X.unsqueeze(1).expand([-1, self.input_dim, -1]) * self.mask.T.unsqueeze(0)  # [batch_size, input_dim, input_dim]

        # Transpose to [input_dim, batch_size, input_dim] for parallel processing
        masked_duplicate_X = masked_duplicate_X.transpose(0, 1)  # [input_dim, batch_size, input_dim]

        # Process all dimensions at once
        pred_X = self.autoencoders(masked_duplicate_X).transpose(0, 1).squeeze()  # [batch_size, input_dim]

        return pred_X
