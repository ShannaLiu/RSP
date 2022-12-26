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

class Estimator(BaseEstimator):
    '''
    Gamma : edge incidence matrix
    D : degree matrix
    method : 'cp' or or 'sylv' (only for RidgeEstimator), or solver name
              complete list for cvxpy solvers https://www.cvxpy.org/tutorial/advanced/index.html
    solver : solver name for 'cp', method name for 'scipy.optimize'
    diag_pen : penalization for diagonal matrix of W
    '''
    def __init__(self, l1:float=0, l2:float=0, l3:float=0, Gamma=None, D=None, method=None, solver=None, deg_crct=False):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        if deg_crct:
            self.Gamma = Gamma@scipy.linalg.sqrtm(np.linalg.pinv(D))
        else:
            self.Gamma = Gamma
        self.D = D
        self.method = method 
        self.solver = solver 
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
        if not np.allclose(self.W.value, self.W.value.T):
            print('The fitted matrix W is not symmetric')
    
    def scaling(self, epsilon=1e-8, max_iter=1000, plot=True, return_value=False):
        W0 = self.W.value
        W0[W0<=0] = 1e-8
        W_new = symscaling(W0, epsilon=epsilon, max_iter=max_iter)
        if plot:
            sb.heatmap(W_new, cmap='rainbow', center=0)
        if return_value:
            return W_new
    

class LR_Estimator(Estimator):
    '''
    Low rank representation
    '''
    def __init__(self, l3=0, method=None, solver=None):
        Estimator.__init__(self, l3=l3, method=method, solver=solver)
        
    def fit(self, X, maxiter):
        if self.method == 'cp':
            n = X.shape[0]
            W1 = cp.Variable((n,n))
            prob = cp.Problem(cp.Minimize( recon_loss(X, W1) + nuclear_penalty(W1, self.l3)))
            prob.solve(max_iters=maxiter, solver=self.solver)
            self.W = W1
    

class ss_El_Estimator(Estimator):
    '''
    symmetric recon loss + symmetric elast loss
    '''
    def __init__(self, l1=0, l2=0, l3=0, Gamma=None, D=None, method=None, solver=None, deg_crct=False):
        Estimator.__init__(self, l1=l1, l2=l2, l3=l3, Gamma=Gamma, D=D, method=method, solver=solver, deg_crct=deg_crct)
    
    def fit(self, X, maxiter):
        if self.method == 'cp':
            n = X.shape[0]
            W1 = cp.Variable((n,n))
            prob = cp.Problem(cp.Minimize( sym_recon_loss(X, W1) + sym_ee_penalty(self.Gamma, W1, self.l1, self.l2) + nuclear_penalty(W1, self.l3)))
            prob.solve(max_iters=maxiter, solver=self.solver)
            self.W = W1

class as_El_Estimator(Estimator):
    '''
    recon loss + symmetric elast loss
    '''
    def __init__(self, l1=0, l2=0, l3=0, Gamma=None, D=None, method=None, solver=None, deg_crct=False):
        Estimator.__init__(self, l1=l1, l2=l2, l3=l3, Gamma=Gamma, D=D, method=method, solver=solver, deg_crct=deg_crct)
    
    def fit(self, X, maxiter):
        if self.method == 'cp':
            n = X.shape[0]
            W1 = cp.Variable((n,n))
            prob = cp.Problem(cp.Minimize( recon_loss(X, W1) + sym_ee_penalty(self.Gamma, W1, self.l1, self.l2) + nuclear_penalty(W1, self.l3)))
            prob.solve(max_iters=maxiter, solver=self.solver)
            self.W = W1

class aa_El_Estimator(Estimator):
    '''
    recon loss + elast loss
    '''
    def __init__(self, l1=0, l2=0, l3=0, Gamma=None, D=None, method=None, solver=None, deg_crct=False):
        Estimator.__init__(self, l1=l1, l2=l2, l3=l3, Gamma=Gamma, D=D, method=method, solver=solver, deg_crct=deg_crct)
    
    def fit(self, X, maxiter):
        if self.method == 'cp':
            n = X.shape[0]
            W1 = cp.Variable((n,n))
            prob = cp.Problem(cp.Minimize( recon_loss(X, W1) + ee_penalty(self.Gamma, W1, self.l1, self.l2) + nuclear_penalty(W1, self.l3)))
            prob.solve(max_iters=maxiter, solver=self.solver)
            self.W = W1



