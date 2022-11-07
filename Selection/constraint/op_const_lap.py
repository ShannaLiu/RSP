# A constraint method for optimizing the Laplacian loss

import torch

def op_const_lap(X, L, r, epsilon=1e-3, ratio=0.5):
    N = X.shape[0]
    W = torch.zeros(N,N)
    W_tilde = torch.zeros(N,N)
    error = torch.tensor([1])
    lamda = ratio / (2*torch.max(torch.matmul(X, X.t()).abs()) )
    loss_rec = []

    while error.item() > epsilon:
        W_tilde = W 
        W = update_W(W_tilde, X, lamda, L, r)
        error = torch.max(torch.abs(W-W_tilde))
        loss_rec.append(compute_loss(W, L, X))
        print(error)
    return W, torch.tensor(loss_rec)


def update_W(W_tilde, X, lamda, L, r):
    N = X.shape[0]
    A_tilde = W_tilde - lamda * 2 * torch.matmul((W_tilde-torch.diag(torch.ones(N))), torch.matmul(X, X.t()))
    W_star = torch.linalg.solve(2*L*lamda+torch.diag(torch.ones(N)), A_tilde)
    column_loss = 1/2*torch.sum((W_star-A_tilde).pow(2), dim=0) + lamda * torch.diag(torch.matmul(W_star.t(), torch.matmul(L, W_star)))
    ind = torch.sort(column_loss, descending=True).indices
    W = torch.zeros_like(W_star)
    W[:,ind[:r]] = W_star[:, ind[:r]]
    return W

def compute_loss(W, T, X):
    loss = torch.sum((X-torch.matmul(W,X)).pow(2)) + torch.sum(torch.abs(T*W))
    return loss



    

