import cvxpy as cp
import numpy as np
import scipy

# For using cvxpy
def recon_loss(X, W):
    return  cp.norm(X-W@X, p='fro') ** 2

def ee_penalty(Gamma, W, l1, l2):
    return l1 * cp.pnorm(Gamma@W, p=1) + l2 * cp.norm(Gamma@W, p='fro')**2

def l1_penalty(Gamma, l1, W):
    return l1 * cp.pnorm(Gamma@W, p=1)

def l2_penalty(Gamma, l2, W):
    return l2 * cp.norm(Gamma@W, p='fro')**2

def sp_l2_penalty(Gamma, W, l1, l2):
    '''
    sparse l1 + l2 penalty
    '''
    return l1 * cp.pnorm(W, p=1) + l2 * cp.norm(Gamma@W, p='fro')**2

def eec_penalty(Gamma, D, W, l1, l2):
    '''
    degree corrected loss
    '''
    normalized_Gamma = Gamma@scipy.linalg.sqrtm(np.linalg.pinv(D))
    return l1 * cp.pnorm(normalized_Gamma@W, p=1) + l2 * cp.norm(normalized_Gamma@W, p='fro')**2

def diag_penalty(W, l3):
    '''
    diagonal penlty loss
    '''
    return l3 * (cp.trace(cp.abs(W)))


def sym_recon_loss(X, W):
    return  cp.norm(X-W@X-W.T@X, p='fro') ** 2


def sym_ee_penalty(Gamma, W, l1, l2):
    return l1 * ( cp.pnorm(Gamma@W, p=1) + cp.pnorm(Gamma@W.T, p=1)) + l2 * (cp.norm(Gamma@W, p='fro')**2 + cp.norm(Gamma@W.T, p='fro')**2 )
