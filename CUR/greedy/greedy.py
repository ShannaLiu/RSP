# This files achieves the naive greedy method in the note

import torch
from scipy.linalg import solve_sylvester
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np

def greedy_lap(X, L, r, lambda1 = 1, plot=False):
    N = X.shape[0]
    X_selected_ind = torch.LongTensor([])
    X_unselected_ind = torch.arange(N)
    A = X
    W = torch.zeros(N,N)
    L = L * lambda1

    total_loss_rec = []
    recon_loss_rec = []
    lap_loss_rec = []

    sum_time_solve_sylvester = []
    sum_time_column_loss = []
    sum_time_selection = []
    sum_time_update_A = []
    sum_time_update_W = []
    sum_time_compute_loss = []

    for i in range(r):
        W, X_selected_ind, X_unselected_ind, A, time_solve_sylvester, time_column_loss, \
            time_selection, time_update_A, time_update_W = greedy_select_lap(X_selected_ind, \
                X_unselected_ind, A, X, L, W)
        
        sum_time_solve_sylvester.append(time_solve_sylvester)
        sum_time_column_loss.append(time_column_loss)
        sum_time_selection.append(time_selection)
        sum_time_update_A.append(time_update_A)
        sum_time_update_W.append(time_update_W)

        start = time.time()
        loss, recon_loss, lap_loss = compute_loss(W, L, X)
        total_loss_rec.append(loss)
        recon_loss_rec.append(recon_loss)
        lap_loss_rec.append(lap_loss)
        sum_time_compute_loss.append(time.time()-start)
    
    if plot:
        f = plt.figure(figsize=(15,5))
        ax0 = f.add_subplot(131)
        ax0.plot(total_loss_rec, label='Total loss')
        ax0.plot(recon_loss_rec, label='reconstruction loss')
        ax0.plot(lap_loss_rec, label='lap loss')
        ax0.legend()
        ax0.set_title('loss in each iteration')

        ax1 = f.add_subplot(122)
        ax1.plot(np.cumsum(sum_time_solve_sylvester), label='sylvester equation')
        ax1.plot(np.cumsum(sum_time_column_loss), label='each column loss')
        ax1.plot(np.cumsum(sum_time_selection), label='column selection')
        ax1.plot(np.cumsum(sum_time_update_A), label='update A')
        ax1.plot(np.cumsum(sum_time_update_W), label='update W)')
        ax1.plot(np.cumsum(sum_time_compute_loss), label='each iter loss (not necessay)')

        ax1.legend()
        ax1.set_title('Computation time')
    return W, total_loss_rec, recon_loss_rec, lap_loss_rec, X_unselected_ind

def greedy_select_lap(X_selected_ind, X_unselected_ind, A, X, L, W):
    X_unselected = X[X_unselected_ind, :]
    X_unselected_norm = torch.diag(torch.sum(X_unselected.pow(2), dim=1)) # row sum

    # Solve Sylvester equation
    start = time.time()
    W_star = torch.tensor(solve_sylvester(L.numpy(), X_unselected_norm.numpy(), torch.matmul(A,X_unselected.t()).numpy()))
    time_solve_sylvester = time.time() - start

    # Compute each column loss
    start = time.time()
    column_loss = compute_column_loss(A, L, W_star, X_unselected)
    time_column_loss = time.time() - start

    # Select next column
    start = time.time()
    selected_ind_loc = torch.sort(column_loss, descending=False).indices[0] # From small to large
    selected_ind = X_unselected_ind[selected_ind_loc]
    time_selection = time.time() - start

    if column_loss.max() < 0:
        warnings.warn('The loss increases after adding a column')

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
    return W, X_selected_ind, X_unselected_ind, A, time_solve_sylvester, time_column_loss, time_selection, time_update_A, time_update_W

def compute_column_loss(A, L, W_star, X_unselected):
    num_unselected = X_unselected.shape[0]
    column_loss = torch.zeros(num_unselected)
    for i in range(num_unselected):
        column_loss[i] = torch.sum( (A-torch.matmul(W_star[:,i].unsqueeze(0).t(), X_unselected[i,:].unsqueeze(0))).pow(2) ) + torch.matmul(W_star[:,i].t(), torch.matmul(L, W_star[:,i]))
    return column_loss

def compute_loss(W, L, X):
    loss = torch.sum((X-torch.matmul(W,X)).pow(2)) + torch.sum(torch.diag(torch.matmul(W.t(), torch.matmul(L, W))))
    recon_loss = torch.sum((X-torch.matmul(W,X)).pow(2)) 
    lap_loss = torch.sum(torch.diag(torch.matmul(W.t(), torch.matmul(L, W))))
    return loss, recon_loss, lap_loss


