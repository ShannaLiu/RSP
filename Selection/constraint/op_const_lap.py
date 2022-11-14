# A constraint method for optimizing the Laplacian loss

import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import warnings

def op_const_lap(X, L, r, lambda1=1, epsilon=1e-3, ratio=0.5, max_iter=100, plot=False, training_mask=None):
    start = time.time()
    N = X.shape[0]
    W = torch.zeros(N,N)
    W_tilde = torch.zeros(N,N)
    error = torch.tensor([1])
    L0 = ( 2*torch.max(torch.matmul(X, X.t()).abs()) )
    L = lambda1 * L
    lamda = ratio / L0
    loss_rec = []
    error_rec = []
    sum_time_compute_A = []
    sum_time_compute_W = []
    sum_time_compute_column_loss = []
    sum_time_compute_column_selection  = []
    sum_time_compute_loss = []
    iter = 0
    time_prep = time.time() - start


    # Compute unchanged matrix multiplication
    XXT = torch.matmul(X, X.t())
    inverse = 2*L*lamda+torch.diag(torch.ones(N))
    
    while error.item() > epsilon:
        W_tilde = W 
        W, time_compute_A, time_compute_W, time_compute_column_loss, time_compute_column_selection = update_W(W_tilde, X, XXT, inverse, lamda, L, r, training_mask)
        sum_time_compute_A.append(time_compute_A)
        sum_time_compute_W.append(time_compute_W)
        sum_time_compute_column_loss.append(time_compute_column_loss)
        sum_time_compute_column_selection.append(time_compute_column_selection)
        error = torch.max(torch.abs(W-W_tilde))
        error = torch.maximum(error, (compute_loss(W, L, X) - compute_loss(W_tilde, L, X)).abs())
        error_rec.append(error)
        start = time.time()
        loss_rec.append(compute_loss(W, L, X))
        sum_time_compute_loss.append(time.time()-start)
        iter = iter + 1

        if iter > max_iter:
            break

    if plot:
        f = plt.figure(figsize=(15,5))
        ax0 = f.add_subplot(131)
        ax0.plot(loss_rec)
        ax0.set_title('loss in each iteration')

        ax1 = f.add_subplot(132)
        ax1.plot(np.cumsum(sum_time_compute_A), label='A tilde')
        ax1.plot(np.cumsum(sum_time_compute_W), label='optimal W')
        ax1.plot(np.cumsum(sum_time_compute_column_loss), label='each column loss')
        ax1.plot(np.cumsum(sum_time_compute_column_selection), label='column selection')
        ax1.plot(np.cumsum(sum_time_compute_loss), label='each iter loss (not necessay)')
        ax1.axhline(y=time_prep, label='preparation (fixed multiply)')
        ax1.legend()
        ax1.set_title('Computation time')

        ax2 = f.add_subplot(133)
        ax2.plot(error_rec)
        ax1.set_title('Error in each iteration')

    return W, loss_rec, sum_time_compute_A, sum_time_compute_W, sum_time_compute_column_loss, sum_time_compute_column_selection


def update_W(W_tilde, X, XXT, inverse, lamda, L, r, training_mask):
    N = X.shape[0]
    W = torch.zeros_like(W_tilde)

    # Compute matrix A from last iteration
    start = time.time()
    A_tilde = W_tilde - lamda * 2 * torch.matmul((W_tilde-torch.diag(torch.ones(N))), XXT)
    time_compute_A = time.time() - start

    # Compute optimal W 
    start = time.time()
    W_star = torch.linalg.solve(inverse, A_tilde)
    # W_star = torch.matmul(inverse, A_tilde)
    time_compute_W = time.time() - start

    # Compute loss decrease for each column
    start = time.time()
    column_loss = 1/2*torch.sum((A_tilde).pow(2), dim=0) - 1/2*torch.sum((W_star-A_tilde).pow(2), dim=0) - lamda * torch.diag(torch.matmul(W_star.t(), torch.matmul(L, W_star)))
    time_compute_column_loss = time.time() - start

    # Sort and generate new W
    start = time.time()
    column_loss[training_mask==0] = -float('inf')
    ind = torch.sort(column_loss, descending=True).indices # select max values
    W[:,ind[:r]] = W_star[:, ind[:r]]
    if column_loss.max() <0 :
        warnings.warn('The column is not decreasing the loss')
    time_compute_column_selection = time.time() - start

    return W, time_compute_A, time_compute_W, time_compute_column_loss, time_compute_column_selection

def compute_loss(W, L, X):
    loss = torch.sum((X-torch.matmul(W,X)).pow(2)) + torch.sum(torch.diag(torch.matmul(W.t(), torch.matmul(L, W))))
    return loss