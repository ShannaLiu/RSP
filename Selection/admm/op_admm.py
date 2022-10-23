# Optimization - iterative method
# Minimize the loss following the same framework as the paper
import torch

def CUR_iter_solver(X, T, alpha, lamda, epsilon, rho_1=10^2, rho_2=10^2, set_seed=False):
    N = X.shape[0]
    if set_seed:
        torch.manual_seed(1234)
    # Initialize variables
    W = torch.zeros(N,N)
    W_tilde = torch.zeros(N,N)
    W_hat = torch.zeros(N,N)
    Lamda_1 = torch.zeros(N,N)
    Lamda_2 = torch.zeros(N,N)

    # Record old loss
    loss_old = -99999

    # record the loss in each iteration if record = True

    loss1_W_rec = []
    loss2_Wt_rec = []
    loss3_W_rec = []
    loss0_Wt_rec = []
    loss2_W_tilde_rec = []
    loss3_W_hat_rec = []
    loss0_W_tilde_rec = []
        
    # Computation for the unchanged terms
    XXT = torch.matmul(X, X.t()) 
    A = 2*XXT + (rho_1 + rho_2) * torch.diag(torch.ones(N))
    A_inverse = A.inverse()

    max_error = torch.tensor([1])

    while max_error.item() > epsilon:
        W = update_W(A_inverse, XXT, rho_1, rho_2, W_tilde, Lamda_1, W_hat, Lamda_2)
        W_tilde = update_W_tilde(W, alpha, rho_1, Lamda_1)
        W_hat = update_W_hat(T, lamda, W, rho_2, Lamda_2)
        Lamda_1 = Lamda_1 + rho_1*(W.t() - W_tilde)
        Lamda_2 = Lamda_2 + rho_2*(W-W_hat)
        # Compute loss functions
        loss1_W = loss1(X, W)
        loss2_Wt = loss2(W.t())
        loss3_W = loss3(T, W)
        loss0_Wt = loss0(W.t())
        loss2_W_tilde = loss2(W_tilde)
        loss3_W_hat = loss3(T, W_hat)
        loss0_W_tilde = loss0(W_tilde)

        loss1_W_rec.append(loss1_W.item())
        loss2_Wt_rec.append(loss2_Wt.item())
        loss3_W_rec.append(loss3_W.item())
        loss0_Wt_rec.append(loss0_Wt)
        loss2_W_tilde_rec.append(loss2_W_tilde.item())
        loss3_W_hat_rec.append(loss3_W_hat.item())
        loss0_W_tilde_rec.append(loss0_W_tilde.item())
        
        # compute stopping criterkon
        error_1 = torch.max(abs(W.t()-W_tilde))
        error_2 = torch.max(abs(W-W_hat))
        loss_new = loss1_W + alpha*loss2_Wt + lamda*loss3_W
        error_3 = torch.max(abs(loss_new-loss_old))
        max_error = torch.max(torch.max(error_1, error_2), error_3)
        loss_old = loss_new

    params = {'W':W, 'W_tilde':W_tilde}
    rec = {'loss1_W': torch.tensor(loss1_W_rec), 'loss2_Wt': torch.tensor(loss2_Wt_rec), 'loss3_W': torch.tensor(loss3_W_rec), 'loss0_Wt_rec': torch.tensor(loss0_Wt_rec),\
            'loss2_W_tilde_rec': torch.tensor(loss2_W_tilde_rec), 'loss3_W_hat_rec': torch.tensor(loss3_W_hat_rec), 'loss0_W_tilde_rec': torch.tensor(loss0_W_tilde_rec)}
    return params, rec


def loss1(X, W):
    loss1 = torch.sum((X-torch.matmul(W,X)).pow(2))
    return loss1

def loss2(W):
    loss2 = torch.sum(torch.sum(W.pow(2), dim=1).sqrt())
    return loss2

def loss3(T, W):
    loss3 = torch.sum(torch.abs(T*W)) 
    return loss3

def loss0(W):
    loss0 = torch.sum(torch.max(torch.abs(W), dim=1).values >0)
    return loss0



def update_W(A_inverse, XXT, rho_1, rho_2, W_tilde, Lamda_1, W_hat, Lamda_2):
    H = 2*XXT + rho_1 * (W_tilde-1/rho_1*Lamda_1).t() + rho_2 * (W_hat-1/rho_2*Lamda_2)
    return torch.matmul(H, A_inverse)

def update_W_tilde(W, alpha, rho_1, Lamda_1):
    V = W.t() + 1/rho_1 * Lamda_1
    eff = torch.maximum(torch.ones((W.shape[0]))-alpha/(rho_1 * torch.sum(V.pow(2), dim=1).sqrt()), torch.zeros(W.shape[0])) # [r] dimension 
    W_tilde = torch.matmul(torch.diag(eff), V) # a scalar for each row
    return W_tilde

def update_W_hat(T, lamda, W, rho_2, Lamda_2):
    t = W + 1/rho_2 * Lamda_2 
    sgn = (t>0).float() - (t<0).float()
    W_hat = torch.maximum(torch.abs(t)-lamda/rho_2*T, torch.zeros_like(t)) * sgn
    return W_hat



