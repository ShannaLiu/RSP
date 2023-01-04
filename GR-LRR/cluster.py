import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.tree import DecisionTreeRegressor
from util import *


def post_process(W, tau):
    '''
    W : the symmetric doubly stochastics scaled matrix
    '''
    n = W.shape[0]
    Sigma, U = reduced_eigen(W)
    if tau is not None:
        num_cluster = np.int0(np.sum(softf(Sigma, tau)))
    else :
        dt = DecisionTreeRegressor(max_leaf_nodes=2).fit(np.array([range(n)]).T , Sigma)
        pred = dt.predict(np.array([range(len(Sigma))]).T)
        num_cluster = len(pred[pred==pred[0]])
    trees_list = []
    pred_mat = np.zeros((num_cluster, n)) 
    for i in range(num_cluster):
        dt = DecisionTreeRegressor(max_leaf_nodes=num_cluster).fit(np.array([range(n)]).T, U.T[i])
        pred_mat[i,:] = dt.predict(np.array([range(n)]).T)
        trees_list.append(dt)
    dt_final = DecisionTreeRegressor(max_leaf_nodes=num_cluster).fit(np.array([range(n)]).T , np.mean(pred_mat, axis=0))
    pred_final = dt_final.predict(np.array([range(n)]).T)
    return pred_final
        
def softf(x, tau):
    y = np.zeros_like(x)
    y[x>=tau] = 1
    y[x<tau] = np.log(1+(x[x<tau]**2/tau**2))
    return y

        

            





        