import numpy as np
from sklearn.metrics import confusion_matrix
import networkx as nx


# Information-based evaluation
def compute_NMI(X,Y):
    '''
    Input:
        X, Y : label for each node
    '''
    p = confusion_matrix(X,Y) / len(X)
    pi = np.sum(p, axis=0)
    pj = np.sum(p, axis=1)
    num = np.diag(1/pi) @ p @ np.diag(1/pj)
    den = ( np.sum(pi * np.log(pi)) + np.sum(pi * np.log(pj)) )
    return -2 * num / den

def compute_RI(X, Y):
    n = len(X)
    a, b, c, d = 0, 0, 0, 0
    for i in range(n):
        for j in range(n):
            if (X[i]==X[j]) & (Y[i]==Y[j]):
                a += 1
            elif (X[i]!=X[j]) & (Y[i]!=Y[j]):
                d += 1
            elif (X[i]!=X[j]) & (Y[i]==Y[j]):
                c += 1
            elif (X[i]==X[j]) & (Y[i]!=Y[j]):
                b += 1
    return (a+d) / (a+b+c+d)

# Quality-based 
def compute_Q(G, X):
    '''
    A : the adjacency matrix
    X : the label for each node
    '''
    n = G.number_of_nodes()
    e = G.number_of_edges()
    A = nx.adj_matrix(G).todense()
    K = np.sum(A, axis=0)
    Q = 0
    for i in range(n):
        for j in range(n):
            Q += ( A[i,j]- K[i]*K[j]/(2*e) ) * (X[i]==X[j])
    return Q / (2*e)

def compute_den(G,X):
    n = G.number_of_nodes()
    e = G.number_of_edges()
    A = nx.adj_matrix(G).todense()
    den = 0
    for i in range(n):
        for j in range(n):
            den += A[i,j] * (X[i]==X[j])
    return den/e






            