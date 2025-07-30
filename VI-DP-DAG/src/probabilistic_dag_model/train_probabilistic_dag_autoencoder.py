import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# def compute_loss_reconstruction(model, loader):
#     model.eval()
#     total_loss = 0.0
#     total_reconstruction_metric = 0.0
#     total_samples = 0
#     nll = nn.PoissonNLLLoss(reduction='sum')
#
#     with torch.no_grad():
#         for X in loader:
#             X = X.to(device)
#             batch_size = X.size(0)
#             model.update_mask(type='deterministic')
#             X_pred = model(X)
#
#             batch_loss = model.grad_loss.item() * batch_size    # TODO: check if grad_loss is per sample!!!
#             total_loss += batch_loss
#
#             # Reconstruction metric for the batch, and ensure X_pred and X have compatible shapes for the loss function
#             # for Poisson NLL
#             batch_reconstruction = nll(X_pred.reshape(-1), X.reshape(-1)).item() # getting scalar sum
#             total_reconstruction_metric += batch_reconstruction
#             total_samples += batch_size
#
#     avg_loss = total_loss / total_samples if total_samples > 0 else 0
#     avg_reconstruction = total_reconstruction_metric / total_samples if total_samples > 0 else 0
#
#     model.train()
#
#     return avg_loss, avg_reconstruction

def compute_loss_reconstruction(model, loader):
    """
    Computes the average loss and reconstruction NLL for a given data loader.

    Args:
        model: The model being evaluated.
        loader: DataLoader for the dataset (train or validation).

    Returns:
        tuple: (average loss, average reconstruction NLL per element).
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_reconstruction_metric = 0.0
    num_elements = 0

    # Ensure log_input=False is explicitly set for clarity, matching training logic
    # Use reduction='sum' to sum NLL over all elements, then average manually
    nll_fn = nn.PoissonNLLLoss(log_input=False, reduction='sum')
    epsilon = 1e-8  # Small value to prevent log(0)

    with torch.no_grad(): # Disable gradient calculations for evaluation
        for X in loader:
            X = X.to(device)
            batch_size = X.size(0)
            num_features = X.shape[1] if len(X.shape) > 1 else 1 # Get number of features per sample

            model.update_mask(type='deterministic')
            X_pred = model(X)
            stable_X_pred = X_pred + epsilon

            # Assuming model.grad_loss holds the ELBO loss component (or similar) per batch
            # It's often better to recalculate the loss components here if possible
            # If grad_loss is *per sample*, this is correct. If it's *per batch*, remove * batch_size
            batch_loss = model.grad_loss.item() * batch_size # Check if grad_loss is per sample or per batch
            total_loss += batch_loss

            # Calculate Poisson NLL for the batch using the stabilized predictions
            batch_reconstruction = nll_fn(stable_X_pred.reshape(-1), X.reshape(-1).float()).item() # getting scalar sum
            total_reconstruction_metric += batch_reconstruction

            # Keep track of the total number of elements processed
            num_elements += batch_size * num_features

    # Calculate average loss per sample
    total_samples = num_elements / num_features if num_features > 0 else 0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    # Calculate average NLL per element
    avg_reconstruction_nll = total_reconstruction_metric / num_elements if num_elements > 0 else 0

    model.train() # Set model back to training mode

    # return average loss (ELBO component) and the average NLL
    return avg_loss, avg_reconstruction_nll

def train_autoencoder(model, train_loader, val_loader, max_epochs=200, frequency=2, patience=50, model_path='saved_model', full_config_dict={}):
    model.to(device)
    model.train()
    train_losses, val_losses, train_nlls, val_nlls = [], [], [], []
    best_val_loss = float("Inf")
    for epoch in range(max_epochs):
        # for batch_index, (X_train) in enumerate(train_loader):
        #     X_train = X_train.to(device)
        #     model.update_mask()
        #     X_pred = model(X_train)
        #     model.step()

        for batch_index, X_train in enumerate(train_loader):
            X_train = X_train.to(device)
            model.update_mask()
            X_pred = model(X_train)
            model.step()

        if epoch % frequency == 0:
            # Stats on data sets
            train_loss, train_nll = compute_loss_reconstruction(model, train_loader)
            train_losses.append(round(train_loss, 5))
            train_nlls.append(train_nll)

            val_loss, val_nll = compute_loss_reconstruction(model, val_loader)
            val_losses.append(val_loss)
            val_nlls.append(val_nll)

            # print("Epoch {} -> Val loss {} | Val NLL.: {}".format(epoch,  round(val_losses[-1], 5), val_nlls[-1]))
            # print("Epoch ", epoch,
            #       "-> Train loss: ", train_losses[-1], "| Val loss: ", val_losses[-1],
            #       "| Train Acc.: ", train_nlls[-1], "| Val Acc.: ", val_nlls[-1])

            if best_val_loss > val_losses[-1]:
                best_val_loss = val_losses[-1]
                torch.save({'epoch': epoch, 'model_config_dict': full_config_dict, 'model_state_dict': model.state_dict(), 'loss': best_val_loss}, model_path)
                # print('Model saved')

            if np.isnan(val_losses[-1]):
                print('Detected NaN Loss')
                break

            if int(epoch / frequency) > patience and val_losses[-patience] <= min(val_losses[-patience:]):
                print('Early Stopping.')
                break

    return train_losses, val_losses, train_nlls, val_nlls
