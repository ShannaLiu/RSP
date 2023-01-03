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
from util import *
import matplotlib.pyplot as plt

class Estimator(BaseEstimator):
    '''
    Gamma : edge incidence matrix
    D : degree matrix
    method : 'cp' or or 'sylv' (only for RidgeEstimator), or solver name
              complete list for cvxpy solvers https://www.cvxpy.org/tutorial/advanced/index.html
    solver : solver name for 'cp', method name for 'scipy.optimize'
    diag_pen : penalization for diagonal matrix of W
    '''
    def __init__(self, alpha:float=0, beta:float=0, gamma:float=0, Gamma=None, D=None, method=None, solver=None, deg_crct=False):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if deg_crct:
            self.Gamma = Gamma@scipy.linalg.sqrtm(np.linalg.pinv(D))
        else:
            self.Gamma = Gamma
        self.D = D
        self.method = method 
        self.solver = solver 
        self.W = None
        if self.solver == 'admm':
            self.trace = None

    def check_matrix(self):
        if self.Gamma is None:
            print('edge incidence matrix is null')
        if self.D is None:
            print('Degree matrix is null')

    def heatplot(self, xticklabels=True, yticklabels=True, cbar=True):
        check_is_fitted(self, "W")
        sb.heatmap(np.abs(self.W), cmap='rainbow', center=0, xticklabels=xticklabels, yticklabels=yticklabels, cbar=cbar)

    def check_method(self):
        if self.method == 'cp':
            if self.solver not in ['SCS', 'ECOS', 'CVXOPT', 'ECOS_BB']:
                self.solver = 'SCS'
                print('Wrong solver name for cvxpy, SCS solver is used by default')
    
    def check_symmetric(self):
        check_is_fitted(self, "W")
        if not np.allclose(self.W, self.W.T):
            print('The fitted matrix W is not symmetric')
    
    def scaling(self, epsilon=1e-5, max_iter=1000, plot=True, return_value=False, xticklabels=True, yticklabels=True, cbar=True):
        W0 = np.abs(self.W)
        W0[W0==0] = 1e-8
        W_new = scaling(W0, epsilon=epsilon, max_iter=max_iter)
        if plot:
            sb.heatmap(W_new, cmap='rainbow', center=0, xticklabels=xticklabels, yticklabels=yticklabels, cbar=cbar)
        if return_value:
            return W_new
    
    def plot_trace(self):
        if self.method != 'admm':
            raise Exception('The trace is only recorded for admm')
        plt.plot(self.trace)
        
    

class LR_Estimator(Estimator):
    '''
    Low rank representation
    '''
    def __init__(self, gamma=0, method=None, solver=None):
        Estimator.__init__(self, gamma=gamma, method=method, solver=solver)
        
    def fit(self, X, maxiter):
        if self.method == 'cp':
            n = X.shape[0]
            W1 = cp.Variable((n,n))
            prob = cp.Problem(cp.Minimize( recon_loss(X, W1) + nuclear_penalty(W1, self.gamma)))
            prob.solve(max_iters=maxiter, solver=self.solver)
            self.W = W1.value
    

class ss_El_Estimator(Estimator):
    '''
    symmetric recon loss + symmetric elast loss
    '''
    def __init__(self, alpha=0, beta=0, gamma=0, Gamma=None, D=None, method=None, solver=None, deg_crct=False):
        Estimator.__init__(self, alpha=alpha, beta=beta, gamma=gamma, Gamma=Gamma, D=D, method=method, solver=solver, deg_crct=deg_crct)
    
    def fit(self, X, maxiter):
        if self.method == 'cp':
            n = X.shape[0]
            W1 = cp.Variable((n,n))
            prob = cp.Problem(cp.Minimize( sym_recon_loss(X, W1) + sym_ee_penalty(self.Gamma, W1, self.alpha, self.beta) + nuclear_penalty(W1, self.gamma)))
            prob.solve(max_iters=maxiter, solver=self.solver)
            self.W = W1.value

class as_El_Estimator(Estimator):
    '''
    recon loss + symmetric elast loss
    '''
    def __init__(self, alpha=0, beta=0, gamma=0, Gamma=None, D=None, method=None, solver=None, deg_crct=False):
        Estimator.__init__(self, alpha=alpha, beta=beta, gamma=gamma, Gamma=Gamma, D=D, method=method, solver=solver, deg_crct=deg_crct)
    
    def fit(self, X, maxiter):

        if self.method == 'cp':
            n = X.shape[0]
            W1 = cp.Variable((n,n))
            prob = cp.Problem(cp.Minimize( recon_loss(X, W1) + sym_ee_penalty(self.Gamma, W1, self.alpha, self.beta) + nuclear_penalty(W1, self.gamma)))
            prob.solve(max_iters=maxiter, solver=self.solver)
            self.W = W1.value
        elif self.method == 'admm':
            n = X.shape[0]
            self.W, self.trace = admm_solver(self.Gamma, X, self.alpha, self.beta, self.gamma, epsilon=1e-8, maxiter=maxiter)


class aa_El_Estimator(Estimator):
    '''
    recon loss + elast loss
    '''
    def __init__(self, alpha=0, beta=0, gamma=0, Gamma=None, D=None, method=None, solver=None, deg_crct=False):
        Estimator.__init__(self, alpha=alpha, beta=beta, gamma=gamma, Gamma=Gamma, D=D, method=method, solver=solver, deg_crct=deg_crct)
    
    def fit(self, X, maxiter):
        if self.method == 'cp':
            n = X.shape[0]
            W1 = cp.Variable((n,n))
            prob = cp.Problem(cp.Minimize( recon_loss(X, W1) + ee_penalty(self.Gamma, W1, self.alpha, self.beta) + nuclear_penalty(W1, self.gamma)))
            prob.solve(max_iters=maxiter, solver=self.solver)
            self.W = W1.value
        




