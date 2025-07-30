import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


def train(model, true_dag_adj, max_epochs=30000, frequency=10, patience=30000, model_path='saved_model', full_config_dict={}):
    model.to(device)
    true_dag_adj = true_dag_adj.to(device)
    model.train()
    prob_abs_losses, sampled_nll_losses = [], []
    best_losses = float("Inf")

    for epoch in range(max_epochs):
        sampled_dag_adj = model.sample()

        epsilon = 1e-8  # Small value to prevent log(0)
        stable_sampled_dag_adj = sampled_dag_adj + epsilon

        # for Normal data
        #sampled_nll_loss = ((true_dag_adj - stable_sampled_dag_adj) ** 2).sum()

        # for Poisson data (V1)
        loss=nn.PoissonNLLLoss(reduction='mean')
        sampled_nll_loss = loss(true_dag_adj, stable_sampled_dag_adj)

        # for Poisson data (V2)
        #sampled_dag_adj = F.softplus(stable_sampled_dag_adj)
        #sampled_nll_loss=-torch.mean(true_dag_adj * torch.log(sampled_dag_adj) - sampled_dag_adj - torch.lgamma(true_dag_adj + 1))

        # for NB
        #r=torch.tensor(2)
        #X_pred_all = F.softplus(X_pred_all, beta=1, threshold=20)
        #sampled_nll_loss= -torch.mean(torch.lgamma(true_dag_adj + r) - torch.lgamma(true_dag_adj + 1) - torch.lgamma(r)+true_dag_adj*torch.log(stable_sampled_dag_adj)+ r*torch.log(r)-(r+true_dag_adj)*torch.log(sampled_dag_adj+r))

    # training loop

        model.optimizer.zero_grad() #zeros out the gradients of the model's parameters. Gradients indicate the direction and magnitude of the adjustments that need to be made to the model's parameters during the learning process. By zeroing out the gradients before each iteration, we ensure that the gradients from the previous iteration don't accumulate and interfere with the current iteration.
        sampled_nll_loss.backward() #computes the gradients of the loss function with respect to the model's parameters. In machine learning, we define a loss function that quantifies how well the model is performing on a given task. By calling backward(), we propagate the loss backward through the model, calculating the gradients of the loss with respect to each parameter. These gradients indicate how the parameters should be adjusted to minimize the loss.

    # added gradient clipping to prevent spikes during training
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

        model.optimizer.step() #updates the model's parameters using the calculated gradients. The optimizer is an algorithm responsible for adjusting the model's parameters based on the gradients. By calling step(), the optimizer takes a step in the direction that reduces the loss, effectively updating the model's parameters. This step is crucial for the model to learn and improve its performance over time.

        if epoch % frequency == 0:
            prob_mask = model.get_prob_mask()
            prob_abs_loss = torch.abs(true_dag_adj - prob_mask).sum() #probabilistic absolute loss to be minimized
            prob_abs_losses.append(prob_abs_loss.detach().cpu().numpy())
            sampled_nll_losses.append(sampled_nll_loss.detach().cpu().numpy())

            print("Epoch {} -> prob_abs_loss {} | sampled_nll_loss {}".format(epoch, prob_abs_losses[-1], sampled_nll_losses[-1]))

            if best_losses > prob_abs_losses[-1]: #we compare the loss of the new model to the previous one and keep the one with a smaller prob absolute loss
                best_losses = prob_abs_losses[-1]
                torch.save({'epoch': epoch, 'model_config_dict': full_config_dict, 'model_state_dict': model.state_dict(), 'loss': best_losses}, model_path)
                print('Model saved')

            if np.isnan(prob_abs_losses[-1]):
                print('Detected NaN Loss')
                break

            if int(epoch / frequency) > patience and prob_abs_losses[-patience] <= min(prob_abs_losses[-patience:]):
                print('Early Stopping.')
                break

    return model, prob_abs_losses, sampled_nll_losses
