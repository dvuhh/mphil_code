import numpy as np
import torch.nn as nn
from src.architectures import SpectralLinear


def linear_sequential(input_dims, hidden_dims, output_dim, k_lipschitz=None, p_drop=None):
    '''

    Args:
        input_dims: input dim
        hidden_dims: list of hidden layer dim
        output_dim: output dim
        k_lipschitz: Lipschitz constant for spectral normalization (optional)
        p_drop: dropout prob (optional)

    Returns:

    '''

    # flattens input dim
    dims = [np.prod(input_dims)] + hidden_dims + [output_dim]
    num_layers = len(dims) - 1
    layers = []
    for i in range(num_layers):
        # Add linear or spectral linear layer
        if k_lipschitz is not None:
            l = SpectralLinear(dims[i], dims[i + 1], k_lipschitz ** (1./num_layers))
            layers.append(l)
        else:
            layers.append(nn.Linear(dims[i], dims[i + 1]))

        # Add ReLU and dropout for all but the last layer
        if i < num_layers - 1:
            layers.append(nn.ReLU())
            if p_drop is not None:
                layers.append(nn.Dropout(p=p_drop))

    # Add non-negative activation to ensure positivity for non normal
    layers.append(nn.Softplus())

    return nn.Sequential(*layers)
