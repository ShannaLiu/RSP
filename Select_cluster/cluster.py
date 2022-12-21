import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.tree import DecisionTreeRegressor

def kplusplus(W, k, random_seed=0):
    # select columns of W
    N = W.shape[0]
    dist = pairwise_distances(W.T) # since we are selecting columns
    np.random.seed(0)
    selected = [np.random.randint(N)]
    for i in range(1,k):
        remained = dist[:,selected]
        remained[selected,:] = 0
        remained = np.min(remained, axis=1)
        new_seleceted = np.argmax( remained )
        selected.append(new_seleceted)
    return selected

def eval_kplusplus(W, X, selected):
    N = W.shape[0]
    proj_matrix = np.zeros((N,2))
    recon_losses = []
    recon_losses.append(np.linalg.norm(X)**2)

    for i in range(1,len(selected)):
        X_selected = X[selected[:i], :] # K * N
        proj_matrix = X @ X_selected.T @ np.linalg.pinv(X_selected @ X_selected.T) # N * K
        recon_losses.append(np.linalg.norm(X-proj_matrix@X_selected)**2)
    recon_losses/recon_losses[0]
    return recon_losses/recon_losses[0]

def eigengap(W_new, method='dt'):
    if method == 'dt':
        res = np.linalg.svd(W_new)
        N = W_new.shape[0]
        dt = DecisionTreeRegressor(max_leaf_nodes=2).fit(np.array([range(N)]).T , res[1])
        pred = dt.predict(np.array([range(len(res[1]))]).T)
        num_cluster = len(pred[pred==pred[0]])
        trees_list = []
        pred_mat = np.zeros((num_cluster, N)) 
        for i in range(num_cluster):
            dt = DecisionTreeRegressor(max_leaf_nodes=num_cluster).fit(np.array([range(N)]).T, res[2][i])
            pred_mat[i,:] = dt.predict(np.array([range(N)]).T)
            trees_list.append(dt)
        dt_final = DecisionTreeRegressor(max_leaf_nodes=num_cluster).fit(np.array([range(N)]).T , np.mean(pred_mat, axis=0))
        pred_final = dt_final.predict(np.array([range(len(res[1]))]).T)
    return pred_final
        
        

            





        