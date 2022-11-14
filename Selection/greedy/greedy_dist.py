# This files achieves the naive greedy method in the note

import torch

def greedy_dist(X, T, r, lambda1):
    N = X.shape[0]
    X_selected_ind = torch.LongTensor([])
    X_unselected_ind = torch.arange(N)
    A = X
    W = torch.zeros(N,N)
    T = lambda1 * T
    for i in range(r):
        W, X_selected_ind, X_unselected_ind, A = greedy_select_dist(X_selected_ind, X_unselected_ind, A, X, T, W)
    return W 

def greedy_select_dist(X_selected_ind, X_unselected_ind, A, X, T, W):
    X_unselected = X[X_unselected_ind, :]
    T_unselected = T[:,X_unselected_ind]
    X_unselected_norm_inv = torch.diag(1/torch.sum(X_unselected.pow(2), dim=1)) # row sum
    W_star = torch.matmul( (torch.matmul(A, X_unselected.t()).abs() > T_unselected/2) * (torch.matmul(A, X_unselected.t()) - T_unselected/2),  X_unselected_norm_inv) * (torch.matmul(A, X_unselected.t())).sign()
    
    # Takes a lot to compute
    column_loss = compute_column_loss(A, T_unselected, X_unselected, W_star)
    selected_ind_loc = torch.sort(column_loss, descending=False).indices[0] 
    selected_ind = X_unselected_ind[selected_ind_loc]
    A = A - torch.matmul(W_star[:,selected_ind_loc].unsqueeze(0).t(), X_unselected[selected_ind_loc,:].unsqueeze(0))
    X_selected_ind = torch.cat([X_selected_ind, selected_ind.unsqueeze(0)])
    X_unselected_ind = X_unselected_ind[X_unselected_ind!=selected_ind]
    W[:,selected_ind] = W_star[:,selected_ind_loc].t()
    return W, X_selected_ind, X_unselected_ind, A

def compute_column_loss(A, T_unselected, X_unselected, W_star):
    num_unselected = X_unselected.shape[0]
    column_loss = torch.zeros(num_unselected)
    for i in range(num_unselected):
        column_loss[i] = torch.sum( (A-torch.matmul(W_star[:,i].unsqueeze(0).t(), X_unselected[i,:].unsqueeze(0))).pow(2) ) + torch.sum( (T_unselected * W_star).abs() )
    return column_loss