# Minimize the loss following the same framework as the paper
from math import comb
import torch

def CUR_iter_solver(X, T, alpha, lamda, epsilon, set_seed=False):
    N = X.shape[0]
    if set_seed:
        torch.manual_seed(1234)
    # Initialize variables
    W = torch.zeros(N,N)
    W_tilde = torch.zeros(N,N)
    W_hat = torch.zeros(N,N)
    Lamda_1 = torch.zeros(N,N)
    Lamda_2 = torch.zeros(N,N)
    rho_1 = rho_2 = 1e-6

    # Record old W
    W_old = torch.zeros(N,N)

    # Computation for the first iter
    XXT = torch.matmul(X, X.t()) 
    max_error = torch.tensor([1])
    max_rho = 1e10
    tau = 1.1

    while max_error.item() > epsilon:
        A = XXT + (rho_1 + rho_2) * torch.diag(torch.ones(N))
        A_inverse = A.inverse()
        W = update_W(A_inverse, XXT, rho_1, rho_2, W_tilde, Lamda_1, W_hat, Lamda_2)
        W_tilde = update_W_tilde(W, alpha, rho_1, Lamda_1)
        W_hat_new = update_W_hat(T, lamda, W, rho_2, Lamda_2)
        Lamda_1 = Lamda_1 + rho_1*(W.t() - W_tilde)
        Lamda_2 = Lamda_2 + rho_2*(W-W_hat_new)
        # compute stopping criterkon
        error_1 = torch.max(abs(W.t()-W_tilde))
        error_2 = torch.max(abs(W-W_hat_new))
        loss_old, _, _, _ = comput_desired_loss(X, W_old, T, alpha, lamda)
        loss_new, _, _, _ = comput_desired_loss(X, W, T, alpha, lamda)
        error_3 = torch.max(abs(loss_new-loss_old))
        max_error = torch.max(torch.max(error_1, error_2), error_3)

        rho_1 = min(tau * rho_1, max_rho)
        rho_2 = min(tau * rho_2, max_rho)
        
        W_old = W
    return W, W_tilde, W_hat, loss_new

def comput_desired_loss(X, W, T, alpha, lamda):
    loss1 = torch.sum((X-torch.matmul(W,X)).pow(2))
    loss2 = alpha*torch.sum(torch.sum(W.pow(2), dim=1).sqrt())
    loss3 = lamda * torch.sum(torch.abs(T*W)) 
    loss =  loss1 + loss2 + loss3
    return loss, loss1, loss2, loss3

def update_W(A_inverse, XXT, rho_1, rho_2, W_tilde, Lamda_1, W_hat, Lamda_2):
    H = 2*XXT + rho_1 * (W_tilde-1/rho_1*Lamda_1).t() + rho_2 * (W_hat-1/rho_2*Lamda_2)
    return torch.matmul(H, A_inverse)

def update_W_tilde(W, alpha, rho_1, Lamda_1):
    V = W.t() + 1/rho_1 * Lamda_1
    eff = torch.ones((W.shape[0]))-alpha/(rho_1 * torch.sum(V.pow(2), dim=1).sqrt()) # [r] dimension 
    W_tilde = torch.matmul(torch.diag(eff), V) # a scalar for each row
    return W_tilde

def update_W_hat(T, lamda, W, rho_2, Lamda_2):
    t = W + 1/rho_2 * Lamda_2 
    sgn = (t>0).float() - (t<0).float()
    W_hat = torch.maximum(torch.abs(t)-lamda/rho_2*T, torch.zeros_like(t)) * sgn
    return W_hat



