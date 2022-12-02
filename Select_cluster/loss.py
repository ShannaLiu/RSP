import cvxpy as cp
import numpy as np

def recon_loss(X, W):
    return  cp.norm(X-W@X, p='fro') ** 2

def ee_penalty(Gamma, W, l1, l2):
    return l1 * cp.sum(cp.abs(Gamma@W)) + l2 * cp.norm(Gamma @ W, p='fro')**2

def sparse_lap(Gamma, W, l1, l2):
    return l1 * cp.sum(cp.abs(W)) + l2 * cp.norm(Gamma @ W, p='fro')**2


def compute_lap_penalty(L, W, l2):
    return l2 * np.sum(np.diag(W.T @ L @ W))


