# This files achieves the naive greedy method in the note

import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt 
import time 

def greedy_dist(X, T, r, lambda1=1, plot=False):
    N = X.shape[0]
    X_selected_ind = torch.LongTensor([])
    X_unselected_ind = torch.arange(N)
    A = X
    W = torch.zeros(N,N)
    T = lambda1 * T

    loss_rec = []
    sum_time_optimal_W = []
    sum_time_column_loss = []
    sum_time_selection = []
    sum_time_update_A = []
    sum_time_update_W = []
    sum_time_compute_loss = []


    for i in range(r):
        W, X_selected_ind, X_unselected_ind, A, time_optimal_W, time_column_loss, time_selection, time_update_A, time_update_W = \
            greedy_select_dist(X_selected_ind, X_unselected_ind, A, X, T, W)
        start = time.time()
        loss_rec.append(compute_loss(W, T, X))
        sum_time_compute_loss.append(time.time()-start)
        sum_time_optimal_W.append(time_optimal_W)
        sum_time_column_loss.append(time_column_loss)
        sum_time_selection.append(time_selection)
        sum_time_update_A.append(time_update_A)
        sum_time_update_W.append(time_update_W)
    
    if plot:
        f = plt.figure(figsize=(15,5))
        ax0 = f.add_subplot(121)
        ax0.plot(loss_rec)
        ax0.set_title('loss in each iteration')

        ax1 = f.add_subplot(122)
        ax1.plot(np.cumsum(sum_time_optimal_W), label='Compute optimal W')
        ax1.plot(np.cumsum(sum_time_column_loss), label='each column loss')
        ax1.plot(np.cumsum(sum_time_selection), label='column selection')
        ax1.plot(np.cumsum(sum_time_update_A), label='update A')
        ax1.plot(np.cumsum(sum_time_update_W), label='update W)')
        ax1.plot(np.cumsum(sum_time_compute_loss), label='each iter loss (not necessay)')

        ax1.legend()
        ax1.set_title('Computation time')


    return W 

def greedy_select_dist(X_selected_ind, X_unselected_ind, A, X, T, W):
    X_unselected = X[X_unselected_ind, :]
    T_unselected = T[:,X_unselected_ind]

    # Compute optimal W
    start = time.time()
    X_unselected_norm_inv = torch.diag(1/torch.sum(X_unselected.pow(2), dim=1)) # row sum
    AX = torch.matmul(A, X_unselected.t())

    W_star = (AX.abs() > T_unselected/2) * torch.matmul((AX - T_unselected/2*AX.sign()), X_unselected_norm_inv)
    time_optimal_W = time.time() - start

    # Compute each column loss
    start = time.time()
    column_loss = compute_column_loss(A, T_unselected, X_unselected, W_star)
    time_column_loss = time.time() - start

    # Select columns
    start = time.time()
    selected_ind_loc = torch.sort(column_loss, descending=False).indices[0] 
    selected_ind = X_unselected_ind[selected_ind_loc]
    time_selection = time.time() - start

    # Update A
    start = time.time()
    A = A - torch.matmul(W_star[:,selected_ind_loc].unsqueeze(0).t(), X_unselected[selected_ind_loc,:].unsqueeze(0))
    time_update_A = time.time() - start 

    # Update W
    start = time.time()
    X_selected_ind = torch.cat([X_selected_ind, selected_ind.unsqueeze(0)])
    X_unselected_ind = X_unselected_ind[X_unselected_ind!=selected_ind]
    W[:,selected_ind] = W_star[:,selected_ind_loc].t()
    time_update_W = time.time() - start

    return W, X_selected_ind, X_unselected_ind, A, time_optimal_W, time_column_loss, time_selection, time_update_A, time_update_W

def compute_column_loss(A, T_unselected, X_unselected, W_star):
    num_unselected = X_unselected.shape[0]
    column_loss = torch.zeros(num_unselected)
    for i in range(num_unselected):
        column_loss[i] = torch.sum( (A-torch.matmul(W_star[:,i].unsqueeze(0).t(), X_unselected[i,:].unsqueeze(0))).pow(2) ) + torch.sum( (T_unselected * W_star).abs() )
    return column_loss

def compute_loss(W, T, X):
    loss = torch.sum((X-torch.matmul(W,X)).pow(2)) + torch.sum(torch.abs(T*W))
    return loss
