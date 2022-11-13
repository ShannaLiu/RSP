# A constraint method for optimizing the Laplacian loss

import torch
from datetime import datetime

def op_const_lap(X, L, r, epsilon=1e-3, ratio=0.5):
    N = X.shape[0]
    W = torch.zeros(N,N)
    W_tilde = torch.zeros(N,N)
    error = torch.tensor([1])
    lamda = ratio / (2*torch.max(torch.matmul(X, X.t()).abs()) )
    loss_rec = []
    now = sum_time_compute_A = sum_time_compute_W = sum_time_compute_column_loss =sum_time_compute_column_selection  = datetime.now()
    

    while error.item() > epsilon:
        W_tilde = W 
        W, time_compute_A, time_compute_W, time_compute_column_loss, time_compute_column_selection = update_W(W_tilde, X, lamda, L, r)
        sum_time_compute_A = time_compute_A + sum_time_compute_A 
        sum_time_compute_W = time_compute_W + sum_time_compute_W 
        sum_time_compute_column_loss = time_compute_column_loss + sum_time_compute_column_loss 
        sum_time_compute_column_selection = time_compute_column_selection + sum_time_compute_column_selection
        error = torch.max(torch.abs(W-W_tilde))

        loss_rec.append(compute_loss(W, L, X))
    
    sum_time_compute_A = sum_time_compute_A - now 
    sum_time_compute_W = sum_time_compute_W - now 
    sum_time_compute_column_loss = sum_time_compute_column_loss - now 
    sum_time_compute_column_selection = sum_time_compute_column_selection - now
    return W, torch.tensor(loss_rec), sum_time_compute_A, sum_time_compute_W, sum_time_compute_column_loss, sum_time_compute_column_selection


def update_W(W_tilde, X, lamda, L, r):
    N = X.shape[0]

    time_compute_A_tilde_start = datetime.now()
    A_tilde = W_tilde - lamda * 2 * torch.matmul((W_tilde-torch.diag(torch.ones(N))), torch.matmul(X, X.t()))
    time_compute_A_tilde_end = datetime.now()
    time_compute_A = time_compute_A_tilde_end - time_compute_A_tilde_start

    time_compute_W_star_start = datetime.now()
    W_star = torch.linalg.solve(2*L*lamda+torch.diag(torch.ones(N)), A_tilde)
    time_compute_W_star_end = datetime.now()
    time_compute_W = time_compute_W_star_end - time_compute_W_star_start

    time_compute_column_loss_start = datetime.now()
    column_loss = 1/2*torch.sum((A_tilde).pow(2), dim=0) - 1/2*torch.sum((W_star-A_tilde).pow(2), dim=0) - lamda * torch.diag(torch.matmul(W_star.t(), torch.matmul(L, W_star)))

    time_compute_column_loss_end = datetime.now()
    time_compute_column_loss = time_compute_column_loss_end - time_compute_column_loss_start

    time_compute_column_selection_start = datetime.now()
    ind = torch.sort(column_loss, descending=True).indices # select max values

    W = torch.zeros_like(W_star)
    W[:,ind[:r]] = W_star[:, ind[:r]]
    time_compute_column_selection_end = datetime.now()
    time_compute_column_selection = time_compute_column_selection_end - time_compute_column_selection_start

    return W, time_compute_A, time_compute_W, time_compute_column_loss, time_compute_column_selection

def compute_loss(W, T, X):
    loss = torch.sum((X-torch.matmul(W,X)).pow(2)) + torch.sum(torch.abs(T*W))
    return loss