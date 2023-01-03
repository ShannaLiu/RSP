# To be explored
from scipy.linalg import solve_sylvester
import numpy as np

def l2_sylvester_solver(X, L, l2):
    a = (L*l2)
    b = X @ X.T
    q = X @ X.T
    W = solve_sylvester(a,b,q)
    return W

def admm_solver(Gamma, X, alpha, beta, gamma, epsilon=1e-8, maxiter=None):
    XXT = X @ X.T
    L = Gamma.T @ Gamma
    crt = 1
    e, n = Gamma.shape
    Lambda1, Lambda2, Lambda3 = np.zeros((e, n)), np.zeros((e, n)), np.zeros((n, n))
    W, What, M, N = np.zeros((n,n)), np.zeros((n,n)), np.zeros((e,n)), np.zeros((e,n))
    rho = 1.1
    mu_max = 1e5
    loss = -10
    mu = 1e-5
    loss_rec = []
    iter = 0

    while crt > epsilon:
        loss_old = loss
        W = update_W(XXT, L, Gamma, M, N, What, alpha, mu, Lambda1, Lambda2, Lambda3)
        What = update_What(W, gamma, mu, Lambda3)
        M = update_M(Gamma, W, beta, mu, Lambda1)
        N = update_N(Gamma, W, beta, mu, Lambda2)
        Lambda1, Lambda2, Lambda3 = update_Lambda(Gamma, W, What, M, N, mu, Lambda1, Lambda2, Lambda3)
        mu = update_mu(mu, rho, mu_max)
        crt, loss = check_cvg(Gamma, X, W, What, alpha, beta, gamma, M, N, loss_old)
        loss_rec.append(loss)
        iter += 1
        if maxiter is not None:
            if iter > maxiter:
                break
    return W, loss_rec

def update_W(XXT, L, Gamma, M, N, What, alpha, mu, Lambda1, Lambda2, Lambda3):
    n = L.shape[0]
    Sigma1, U1 = reduced_eigen(2*XXT + (alpha+mu)*L) 
    Sigma2, U2 = reduced_eigen(mu*np.eye(n) + (alpha+mu)*L)
    P = 2*XXT + mu*(Gamma.T@M + N.T@Gamma + What) + Gamma.T@Lambda1 + Lambda2.T@Gamma + Lambda3
    num = U2.T @ P @ U1 
    Sigma1 = np.repeat(Sigma1.reshape(1,-1), n, axis=0)
    Sigma2 = np.repeat(Sigma2.reshape(1,-1), n, axis=1).reshape(n,n)
    den = Sigma1  + Sigma2
    Z = num/den
    return U2 @ Z @ U1.T

def update_What(W, gamma, mu, Lambda3):
    U, Sigma, V = np.linalg.svd(W - 1/mu * Lambda3)
    Sigma_new = Sigma * (Sigma>(gamma/mu))
    return U @ np.diag(Sigma_new) @ V

def update_M(Gamma, W, beta, mu, Lambda1):
    A = Gamma @ W - 1/mu * Lambda1
    return np.squeeze(np.asarray(( np.abs(A) > (beta/(2*mu)))))  * np.squeeze(np.asarray( np.abs(A) - (beta/(2*mu)) ))  * np.squeeze(np.asarray(np.sign(A))) 

def update_N(Gamma, W, beta, mu, Lambda2):
    A = Gamma @ W.T - 1/mu * Lambda2
    return np.squeeze(np.asarray(( np.abs(A) > (beta/(2*mu)))))  * np.squeeze(np.asarray( np.abs(A) - (beta/(2*mu)) ))  * np.squeeze(np.asarray(np.sign(A))) 

def update_Lambda(Gamma, W, What, M, N, mu, Lambda1, Lambda2, Lambda3):
    Lambda1 = Lambda1 + mu * (M - Gamma @ W)
    Lambda2 = Lambda2 + mu * (N - Gamma @ W.T)
    Lambda3 = Lambda3 + mu * (What - W)
    return Lambda1, Lambda2, Lambda3

def update_mu(mu, rho, mu_max):
    return np.min([rho*mu, mu_max])

def check_cvg(Gamma, X, W, What, alpha, beta, gamma, M, N, loss_old):
    loss_new = loss_l(Gamma, X, W, alpha, beta, gamma)
    e1, e2, e3, e4 = np.max(np.abs(M-Gamma@W)), np.max(np.abs(N-Gamma@W.T)), np.max(np.abs(What-W)), np.abs(loss_new-loss_old)
    return np.max([e1, e2, e3, e4]), loss_new

def loss_l(Gamma, X, W, alpha, beta, gamma):
    return np.linalg.norm(X-W@X, ord='fro')  + (alpha/2) * (np.linalg.norm(Gamma@W, ord='fro') + np.linalg.norm(Gamma@W.T, ord='fro'))  \
         + (beta/2) * (np.sum(np.abs(Gamma@W)) + np.sum(np.abs(Gamma@W.T))) + gamma * np.linalg.norm(W, ord='nuc')
      
def reduced_eigen(A):
    Sigma, U = np.linalg.eigh(A)
    idx = Sigma.argsort()[::-1]   
    Sigma = Sigma[idx]
    U = U[:,idx]
    k = np.linalg.matrix_rank(A)
    return Sigma[:k], U[:,:k]
