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
    