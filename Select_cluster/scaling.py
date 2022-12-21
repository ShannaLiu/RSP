# Scaling a matrix to total 
import numpy as np

def symscaling(A, epsilon=1e-5, max_iter=1000):
    N = A.shape[0]
    A_new = A 
    D = np.eye(N)
    error = 1
    iter = 0
    while error > epsilon:
        if iter > max_iter:
            print('Maximum number of iteration is reached, use other method or change max_iter')
            break
        A_old = A_new
        R_inv = np.diag(1/np.sqrt(np.sum(A_old, axis=1)))
        A_new = R_inv @ A_old @ R_inv
        D = D @ R_inv
        iter += 1
        error = np.max(np.abs(A_old - A_new))
    return A_new