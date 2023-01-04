import numpy as np
from sklearn.metrics import confusion_matrix
import networkx as nx
from sklearn import metrics
from sklearn.metrics.cluster import rand_score

# Information-based evaluation
def compute_NMI(X,Y):
    '''
    Input:
        X, Y : label for each node
    '''
    return metrics.adjusted_mutual_info_score(X, Y) 
    
def compute_RI(X, Y):
    return rand_score(X,Y)

# Quality-based 
def compute_Q(G, X):
    '''
    G : the graph
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

def compute_den(G, X):
    n = G.number_of_nodes()
    e = G.number_of_edges()
    A = nx.adj_matrix(G).todense()
    den = 0
    for i in range(n):
        for j in range(n):
            den += A[i,j] * (X[i]==X[j])
    return den/e






            