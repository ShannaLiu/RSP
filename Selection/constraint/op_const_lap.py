# A constraint method for optimizing the Laplacian loss

import torch
import time 

def op_const_lap(X, L, r, epsilon=1e-3, ratio=0.5, training_mask=False):
    N = X.shape[0]
    W = torch.zeros(N,N)
    W_tilde = torch.zeros(N,N)
    error = torch.tensor([1])
    L = ( 2*torch.max(torch.matmul(X, X.t()).abs()) )
    lamda = ratio / L
    loss_rec = []
    sum_time_compute_A = 0
    sum_time_compute_W = 0
    sum_time_compute_column_loss = 0
    sum_time_compute_column_selection  = 0

    # Compute unchanged matrix multiplication
    XXT = torch.matmul(X, X.t())
    

    while error.item() > epsilon:
        W_tilde = W 
        W, time_compute_A, time_compute_W, time_compute_column_loss, time_compute_column_selection = update_W(W_tilde, X, XXT, lamda, L, r)
        sum_time_compute_A = time_compute_A + sum_time_compute_A 
        sum_time_compute_W = time_compute_W + sum_time_compute_W 
        sum_time_compute_column_loss = time_compute_column_loss + sum_time_compute_column_loss 
        sum_time_compute_column_selection = time_compute_column_selection + sum_time_compute_column_selection
        error = torch.max(torch.abs(W-W_tilde))

        loss_rec.append(compute_loss(W, L, X))
    
    return W, torch.tensor(loss_rec), sum_time_compute_A, sum_time_compute_W, sum_time_compute_column_loss, sum_time_compute_column_selection


def update_W(W_tilde, X, XXT, lamda, L, r):
    N = X.shape[0]

    # Compute matrix A from last iteration
    start = time.time()
    A_tilde = W_tilde - lamda * 2 * torch.matmul((W_tilde-torch.diag(torch.ones(N))), XXT)
    time_compute_A = time.time() - start

    # Compute optimal W 
    time_compute_W_star_start = time.time()
    W_star = torch.linalg.solve(2*L*lamda+torch.diag(torch.ones(N)), A_tilde)
    time_compute_W_star_end = time.time()
    time_compute_W = time_compute_W_star_end - time_compute_W_star_start

    time_compute_column_loss_start = time.time()
    column_loss = 1/2*torch.sum((A_tilde).pow(2), dim=0) - 1/2*torch.sum((W_star-A_tilde).pow(2), dim=0) - lamda * torch.diag(torch.matmul(W_star.t(), torch.matmul(L, W_star)))

    time_compute_column_loss_end = time.time()
    time_compute_column_loss = time_compute_column_loss_end - time_compute_column_loss_start

    time_compute_column_selection_start = time.time()
    ind = torch.sort(column_loss, descending=True).indices # select max values

    W = torch.zeros_like(W_star)
    W[:,ind[:r]] = W_star[:, ind[:r]]
    time_compute_column_selection_end = time.time()
    time_compute_column_selection = time_compute_column_selection_end - time_compute_column_selection_start

    return W, time_compute_A, time_compute_W, time_compute_column_loss, time_compute_column_selection

def compute_loss(W, T, X):
    loss = torch.sum((X-torch.matmul(W,X)).pow(2)) + torch.sum(torch.abs(T*W))
    return loss