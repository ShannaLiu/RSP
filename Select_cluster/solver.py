import numpy as np
import cvxpy as cp
from scipy.linalg import solve_sylvester
from loss import *

def elast_cp_solver(X, Gamma, l1, l2, maxiter):
    n = X.shape[0]
    W = cp.Variable((n,n))
    prob = cp.Problem(cp.Minimize( recon_loss(X, W) + ee_penalty(Gamma, W, l1, l2) ))
    prob.solve(max_iters=maxiter)
    return W.value

def sparsel2_cp_solver(X, Gamma, l1, l2, maxiter):
    n = X.shape[0]
    W = cp.Variable((n,n))
    prob = cp.Problem(cp.Minimize( recon_loss(X, W) + sparse_lap(Gamma, W, l1, l2) ))
    prob.solve(max_iters=maxiter)
    return W.value

def lap_sylvester_solver(X, L, l2):
    a = (L*l2)
    b = X @ X.T
    q = X @ X.T
    W = solve_sylvester(a,b,q)
    return W









