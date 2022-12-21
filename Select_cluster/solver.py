# To be explored
from scipy.linalg import solve_sylvester

def l2_sylvester_solver(X, L, l2):
    a = (L*l2)
    b = X @ X.T
    q = X @ X.T
    W = solve_sylvester(a,b,q)
    return W
