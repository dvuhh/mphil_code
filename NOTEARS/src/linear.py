import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid


def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            # M is X @ W, which are the linear predictors eta = XW
            # For Poisson, rate lambda = exp(eta)
            # The loss is proportional to sum(lambda_i - x_i * eta_i) which is from Poisson log-likelihood: sum(x_i * eta_i - lambda_i)
            
            # --- START MODIFICATION ---
            MAX_ARG_FOR_EXP_IN_LOSS = 15.0  # Start with a conservative value (e.g., 10.0 to 20.0)
                                          # This means max rate will be exp(15) ~ 3.2e6
            MIN_ARG_FOR_EXP_IN_LOSS = -50.0 # To prevent exp(-large_num) becoming exactly 0.0 too quickly,
                                          # which might cause issues if log(rates) or 1/rates were used (though not directly here for S).
        
            clipped_M = np.clip(M, MIN_ARG_FOR_EXP_IN_LOSS, MAX_ARG_FOR_EXP_IN_LOSS)
            S = np.exp(clipped_M) # S represents the predicted rates (lambda)
        
            # Optional: Further clip S to ensure rates are strictly positive and bounded,
            # if downstream calculations are ultra-sensitive or if X*M could still cause overflow with large X and clipped M.
            S = np.clip(S, 1e-9, 1e9) # Clip rates to be between a tiny positive and a large number
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss
        

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    max_abs_w_val = 3.0  # Try this first. If warnings persist, you could try 2.5 or 2.0.

    bnds = []
    for _ in range(2):  # For w_pos and w_neg parts
        for i in range(d):
            for j in range(d):
                if i == j:
                    bnds.append((0, 0))  # W_ii = 0, no self-loops
                else:
                    # 0 <= individual_w_component <= max_abs_w_val
                    bnds.append((0, max_abs_w_val))     
    
    # bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


if __name__ == '__main__':
#     from notears import utils
    import utils
    utils.set_random_seed(1)

    n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    np.savetxt('W_true.csv', W_true, delimiter=',')

    X = utils.simulate_linear_sem(W_true, n, sem_type)
    np.savetxt('X.csv', X, delimiter=',')

    W_est = notears_linear(X, lambda1=0.1, loss_type='l2')
    assert utils.is_dag(W_est)
    np.savetxt('W_est.csv', W_est, delimiter=',')
    acc = utils.count_accuracy(B_true, W_est != 0)
    print(acc)

