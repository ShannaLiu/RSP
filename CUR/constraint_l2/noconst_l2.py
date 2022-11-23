# optimize Laplacian grouping loss without any constraint

import torch
import time
import numpy as np
import matplotlib.pyplot as plt

def const_no(X, L, lambda1=1, epsilon=1e-3, ratio=0.5, max_iter=100, init_method='zero', plot=False):
    start = time.time()
    N = X.shape[0]
    W = torch.zeros(N,N)
    if init_method not in ['zero', 'rand']:
        raise Exception('Wrong method for init_method, please use ''zero'' or ''rand'' ')
    else : 
        if init_method == 'rand':
            W = torch.randn(N,N)
    W_tilde = torch.clone(W)
    error = torch.tensor([1])

    # Matrix product XXT for futuren use
    XXT = torch.matmul(X, X.t())

    # lip l and step size
    l = ( 2*torch.norm(XXT) ) # lip l
    lamda = ratio / l # step size

    # rescale L based on the value of lambda1
    L = lambda1 * L

    # matrix inverse for solving linear equation
    inverse = (2*L*lamda+torch.diag(torch.ones(N))).inverse()

    total_loss_rec = []
    recon_loss_rec = []
    lap_loss_rec = []
    error_rec = []
    sum_time_compute_A = []
    sum_time_compute_W = []
    sum_time_compute_column_loss = []
    sum_time_compute_column_selection  = []
    sum_time_compute_loss = []
    iter = 0
    time_prep = time.time() - start



    while error.item() > epsilon:
        W_tilde = W 
        W, time_compute_A, time_compute_W = update_W(W_tilde, X, XXT, inverse, lamda)
        sum_time_compute_A.append(time_compute_A)
        sum_time_compute_W.append(time_compute_W)

        
        start = time.time()
        tot_loss, recon_loss, lap_loss = compute_loss(W, L, X)
        total_loss_rec.append(tot_loss)
        recon_loss_rec.append(recon_loss)
        lap_loss_rec.append(lap_loss)

        sum_time_compute_loss.append(time.time()-start)
        error = torch.max(torch.abs(W-W_tilde))
        error = torch.maximum(error, (tot_loss - compute_loss(W_tilde, L, X)[0]).abs())
        error_rec.append(error)

        iter = iter + 1

        if iter > max_iter:
            break

    if plot:
        f = plt.figure(figsize=(15,5))
        ax0 = f.add_subplot(131)
        ax0.plot(total_loss_rec, label='Total loss')
        ax0.plot(recon_loss_rec, label='reconstruction loss')
        ax0.plot(lap_loss_rec, label='Laplacian loss')
        ax0.legend()
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
        ax2.set_title('Error in each iteration')

    return W, total_loss_rec, recon_loss_rec, lap_loss_rec, error_rec

def update_W(W_tilde, X, XXT, inverse, lamda):
    N = X.shape[0]

    # Compute matrix A from last iteration
    start = time.time()
    A_tilde = W_tilde - lamda * 2 * torch.matmul((W_tilde-torch.diag(torch.ones(N))), XXT)
    time_compute_A = time.time() - start

    # Compute optimal W 
    start = time.time()
    W_star = torch.matmul(inverse, A_tilde)
    # W_star = torch.matmul(inverse, A_tilde)
    time_compute_W = time.time() - start

    return W_star, time_compute_A, time_compute_W

def compute_loss(W, L, X):
    loss = torch.sum((X-torch.matmul(W,X)).pow(2)) + torch.sum(torch.diag(torch.matmul(W.t(), torch.matmul(L, W))))
    recon_loss = torch.sum((X-torch.matmul(W,X)).pow(2))
    lap_loss = torch.sum(torch.diag(torch.matmul(W.t(), torch.matmul(L, W))))
    return loss, recon_loss, lap_loss