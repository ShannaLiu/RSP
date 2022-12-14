import networkx as nx 
import numpy as np

def graph_to_mat(G):
    '''
    Get Laplacian matrix, edge incidence matrix, degree matri
    '''
    L = np.float32(nx.laplacian_matrix(G).todense())
    D = np.diag(np.diag(L))
    Gamma = np.float32(nx.incidence_matrix(G).todense().T) # E * N
    for i in range(Gamma.shape[0]):
        for j in range(Gamma.shape[1]):
            if Gamma[i,j] !=0:
                Gamma[i,j] = - Gamma[i,j]
                break
    return L, D, Gamma
    
def scaling(A, epsilon=1e-5, max_iter=1000):
    N = A.shape[0]
    A_new = A 
    # D = np.eye(N)
    # E = np.eye(N)
    error = 1
    iter = 0
    while error > epsilon:
        if iter > max_iter:
            break
        A_old = A_new
        R_inv = np.diag(1/np.sqrt(np.sum(A_old, axis=1)))
        C_inv = np.diag(1/np.sqrt(np.sum(A_old, axis=0)))
        A_new = R_inv @ A_old @ C_inv
        # D = D @ R_inv
        # E = E @ C_inv
        iter += 1
        error = np.max(np.abs(A_old - A_new))
    return A_new

def reduced_eigen(A):
    Sigma, U = np.linalg.eigh(A)
    idx = Sigma.argsort()[::-1]   
    Sigma = Sigma[idx]
    U = U[:,idx]
    k = np.linalg.matrix_rank(A)
    return Sigma[:k], U[:,:k]