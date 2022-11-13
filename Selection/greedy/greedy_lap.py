# This files achieves the naive greedy method in the note

import torch
from scipy.linalg import solve_sylvester

def greedy_lap(X, L, r):
    N = X.shape[0]
    X_selected_ind = torch.LongTensor([])
    X_unselected_ind = torch.arange(N)
    A = X
    W = torch.zeros(N,N)
    for i in range(r):
        W, X_selected_ind, X_unselected_ind, A = greedy_select_lap(X_selected_ind, X_unselected_ind, A, X, L, W)
    return W 

def greedy_select_lap(X_selected_ind, X_unselected_ind, A, X, L, W):
    X_unselected = X[X_unselected_ind, :]
    X_unselected_norm = torch.diag(torch.sum(X_unselected.pow(2), dim=1)) # row sum
    W_star = torch.tensor(solve_sylvester(L.numpy(), X_unselected_norm.numpy(), torch.matmul(A,X_unselected.t()).numpy()))
    # Takes a lot to compute
    column_loss = compute_column_loss(A, L, W_star, X_unselected)
    selected_ind_loc = torch.sort(column_loss, descending=False).indices[0] # Think about it!
    selected_ind = X_unselected_ind[selected_ind_loc]
    A = A - torch.matmul(W_star[:,selected_ind_loc].unsqueeze(0).t(), X_unselected[selected_ind_loc,:].unsqueeze(0))
    X_selected_ind = torch.cat([X_selected_ind, selected_ind.unsqueeze(0)])
    X_unselected_ind = X_unselected_ind[X_unselected_ind!=selected_ind]
    W[:,selected_ind] = W_star[:,selected_ind_loc].t()
    return W, X_selected_ind, X_unselected_ind, A

def compute_column_loss(A, L, W_star, X_unselected):
    num_unselected = X_unselected.shape[0]
    column_loss = torch.zeros(num_unselected)
    for i in range(num_unselected):
        column_loss[i] = torch.sum( (A-torch.matmul(W_star[:,i].unsqueeze(0).t(), X_unselected[i,:].unsqueeze(0))).pow(2) ) + torch.matmul(W_star[:,i].t(), torch.matmul(L, W_star[:,i]))
    return column_loss


