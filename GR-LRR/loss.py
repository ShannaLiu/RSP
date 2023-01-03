import cvxpy as cp
import numpy as np
import scipy

# For using cvxpy
def recon_loss(X, W):
    return  cp.norm(X-W@X, p='fro') ** 2

def sym_recon_loss(X, W):
    return  cp.norm(X - W@X - W.T@X, p='fro') ** 2

def ee_penalty(Gamma, W, alpha, beta):
    return (alpha/2) * cp.pnorm(Gamma@W, p=1) + (beta/2) * cp.norm(Gamma@W, p='fro')**2

def sym_ee_penalty(Gamma, W, alpha, beta):
    return (alpha/2) * ( cp.pnorm(Gamma@W, p=1) + cp.pnorm(Gamma@W.T, p=1)) + (beta/2) * (cp.norm(Gamma@W, p='fro')**2 + cp.norm(Gamma@W.T, p='fro')**2 )

def nuclear_penalty(W, gamma):
    return gamma*cp.norm(W, p='nuc')

def g_penalty(W, X):
    return cp.norm(W-X, p='fro')
