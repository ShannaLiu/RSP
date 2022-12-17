import numpy as np
import cvxpy as cp
from scs import SCS
from osqp import OSQP
from sklearn.base import BaseEstimator
import seaborn as sb
from sklearn.utils.validation import check_is_fitted
from scipy.optimize import minimize

from loss import *
from solver import *

class Estimator(BaseEstimator):
    '''
    Gamma : edge incidence matrix
    D : degree matrix
    method : 'cp' or or 'sylv' (only for RidgeEstimator), or solver name
              complete list for cvxpy solvers https://www.cvxpy.org/tutorial/advanced/index.html
    solver : solver name for 'cp', method name for 'scipy.optimize'
    diag_pen : penalization for diagonal matrix of W
    '''
    def __init__(self, l1:float=0, l2:float=0, l3:float=0, Gamma=None, D=None, method=None, solver=None, diag_pen=False):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.Gamma = Gamma
        self.D = D
        self.method = method 
        self.solver = solver 
        self.diag_pen = diag_pen
        self.W = None

    def check_matrix(self):
        if self.Gamma is None:
            print('edge incidence matrix is null')
        if self.D is None:
            print('Degree matrix is null')

    def heatplot(self, xticklabels=True, yticklabels=True, cbar=True):
        check_is_fitted(self, "W")
        sb.heatmap(self.W.value, cmap='rainbow', center=0, xticklabels=xticklabels, yticklabels=yticklabels, cbar=cbar)

    def check_method(self):
        if self.method == 'cp':
            if self.solver not in ['SCS', 'ECOS', 'CVXOPT', 'ECOS_BB']:
                self.solver = 'SCS'
                print('Wrong solver name for cvxpy, SCS solver is used by default')
    
    def check_symmetric(self):
        check_is_fitted(self, "W")
        np.allclose(self.W.value, self.W.value.T)
    

class ElastEstimator(Estimator):
    def __init__(self, l1=0, l2=0, l3=0, Gamma=None, D=None, method=None, solver=None, diag_pen=False):
        Estimator.__init__(self, l1=l1, l2=l2, l3=l3, Gamma=Gamma, D=D, method=method, solver=solver, diag_pen=diag_pen)
    
    def fit(self, X, maxiter):
        if self.method == 'cp':
            n = X.shape[0]
            W1 = cp.Variable((n,n))
            if self.diag_pen:
                prob = cp.Problem(cp.Minimize( recon_loss(X, W1) + ee_penalty(self.Gamma, W1, self.l1, self.l2) + diag_penalty(W1, self.l3) ))
            else:
                prob = cp.Problem(cp.Minimize( recon_loss(X, W1) + ee_penalty(self.Gamma, W1, self.l1, self.l2) ))
            prob.solve(max_iters=maxiter, solver=self.solver)
            self.W = W1


class LassoEstimator(Estimator):
    def __init__(self, l1=0, l2=0, l3=0, Gamma=None, D=None, method=None, solver=None, diag_pen=False):
        Estimator.__init__(self, l1=l1, l2=l2, l3=l3, Gamma=Gamma, D=D, method=method, solver=solver, diag_pen=diag_pen)

    def fit(self, X, maxiter):
        if self.method == 'cp':
            n = X.shape[0]
            W1 = cp.Variable((n,n))
            if self.diag_pen:
                prob = cp.Problem(cp.Minimize( recon_loss(X, W1) + l1_penalty(self.Gamma, self.l1, W1) ))
            else:
                prob = cp.Problem(cp.Minimize( recon_loss(X, W1) + l1_penalty(self.Gamma, self.l1, W1) + diag_penalty(W1, self.l3) ))
            prob.solve(max_iters=maxiter, solver=self.solver)
            self.W = W1


class RidgeEstimator(Estimator):
    def __init__(self, l1=0, l2=0, l3=0, Gamma=None, D=None, method=None, solver=None, diag_pen=False):
        Estimator.__init__(self, l1=l1, l2=l2, l3=l3, Gamma=Gamma, D=D, method=method, solver=solver, diag_pen=diag_pen)

    def fit(self, X, maxiter):
        if self.method == 'cp':
            n = X.shape[0]
            W1 = cp.Variable((n,n))
            if self.diag_pen:
                prob = cp.Problem(cp.Minimize( recon_loss(X, W1) + l2_penalty(self.Gamma, self.l2, W1) + diag_penalty(W1, self.l3) ))
            else:
                prob = cp.Problem(cp.Minimize( recon_loss(X, W1) + l2_penalty(self.Gamma, self.l2, W1) ))
            prob.solve(max_iters=maxiter, solver=self.solver)
            self.W = W1
        if self.method == 'sylv':
            W1 = l2_sylvester_solver(X, self.L, self.l2)
            self.W = W1 


class sparseElastEstimator(Estimator):
    def __init__(self, l1=0, l2=0, l3=0, Gamma=None, D=None, method=None, solver=None, diag_pen=False):
        Estimator.__init__(self, l1=l1, l2=l2, l3=l3, Gamma=Gamma, D=D, method=method, solver=solver, diag_pen=diag_pen)
    
    def fit(self, X, maxiter):
        if self.method == 'cp':
            n = X.shape[0]
            W1 = cp.Variable((n,n))
            if self.diag_pen:
                prob = cp.Problem(cp.Minimize( recon_loss(X, W1) + sp_l2_penalty(self.Gamma, W1, self.l1, self.l2) + diag_penalty(W1, self.l3)))
            else :
                prob = cp.Problem(cp.Minimize( recon_loss(X, W1) + sp_l2_penalty(self.Gamma, W1, self.l1, self.l2) ))                
            prob.solve(max_iters=maxiter, solver=self.solver)
            self.W = W1


class correctedElastEstimator(Estimator):
    def __init__(self, l1=0, l2=0, l3=0, Gamma=None, D=None, method=None, solver=None, diag_pen=False):
        Estimator.__init__(self, l1=l1, l2=l2, l3=l3, Gamma=Gamma, D=D, method=method, solver=solver, diag_pen=diag_pen)
    
    def fit(self, X, maxiter):
        if self.method == 'cp':
            n = X.shape[0]
            W1 = cp.Variable((n,n))
            if self.diag_pen:
                prob = cp.Problem(cp.Minimize( recon_loss(X, W1) + eec_penalty(self.Gamma, self.D, W1, self.l1, self.l2) + diag_penalty(W1, self.l3)))
            else :
                prob = cp.Problem(cp.Minimize( recon_loss(X, W1) + eec_penalty(self.Gamma, self.D, W1, self.l1, self.l2) ))
            prob.solve(max_iters=maxiter, solver=self.solver)
            self.W = W1

class symElastEstimator(Estimator):
    def __init__(self, l1=0, l2=0, l3=0, Gamma=None, D=None, method=None, solver=None, diag_pen=False):
        Estimator.__init__(self, l1=l1, l2=l2, l3=l3, Gamma=Gamma, D=D, method=method, solver=solver, diag_pen=diag_pen)
    
    def fit(self, X, maxiter):
        if self.method == 'cp':
            n = X.shape[0]
            W1 = cp.Variable((n,n))
            if self.diag_pen:
                prob = cp.Problem(cp.Minimize( sym_recon_loss(X, W1) + sym_ee_penalty(self.Gamma, W1, self.l1, self.l2) + diag_penalty(W1, self.l3) ))
            else:
                prob = cp.Problem(cp.Minimize( sym_recon_loss(X, W1) + sym_ee_penalty(self.Gamma, W1, self.l1, self.l2) ))
            prob.solve(max_iters=maxiter, solver=self.solver)
            self.W = W1




# def elast_cp_solver(X, Gamma, l1, l2, maxiter, solver='SCS'):
#     n = X.shape[0]
#     W = cp.Variable((n,n))
#     prob = cp.Problem(cp.Minimize( recon_loss(X, W) + ee_penalty(Gamma, W, l1, l2) ))
#     prob.solve(max_iters=maxiter, solver=solver)
#     return W.value

# def sparsel2_cp_solver(X, Gamma, l1, l2, maxiter, solver='SCS'):
#     n = X.shape[0]
#     W = cp.Variable((n,n))
#     prob = cp.Problem(cp.Minimize( recon_loss(X, W) + sparse_lap(Gamma, W, l1, l2) ))
#     prob.solve(max_iters=maxiter, solver=solver)
#     return W.value

# def lap_sylvester_solver(X, L, l2):
#     a = (L*l2)
#     b = X @ X.T
#     q = X @ X.T
#     W = solve_sylvester(a,b,q)
#     return W










