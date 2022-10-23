# This files achieves the algorithm 1 in 
# https://uwaterloo.ca/data-analytics/sites/ca.data-analytics/files/uploads/files/farahat_feature_preprint.pdf
# Basic idea: maximize the decreased loss in each step
# disadvantage: Projection matrix is directly determined by the selected columns, no regulariztaion


# Notive that this is a columns subset selection algorithm, so we need to transpose the matrix
'''
    X: the matrix we want to perform column subset selection
    c: number of partitions (for computational efficiency)
    k: number of iteration / number of selected columns
'''

import torch

def greedy_CSS(A, k, c=None):
    m, n = A.shape
    # Generate a partitioning P if c is int
    if type(c) == int:
        cumsum = (torch.randperm(n)[:c-1]+1).sort().values
        # define an partition matirx R [n * c]
        R = torch.zeros(n, c)
        R[:cumsum[0], 0] = 1
        for i in range(1,c-1):
            R[cumsum[i-1]:cumsum[i], i] = 1
        R[cumsum[c-2]:, c-1] = 1
    else:
        # Each data is a partition
        R = torch.diag(torch.ones(n,)) 
        c = n
    
    # Step 2: Initialize
    B = torch.matmul(A, R) # m*c 
    BTA = torch.matmul(B.t(), A) # c*n
    f = torch.sum(BTA.pow(2), dim=0) # n
    ATA = torch.matmul(A.t(), A)
    g = torch.diagonal(ATA) # n

    # Step 3: Iteratively select
    W = torch.zeros(n, k)
    V = torch.zeros(c, k)
    L = torch.zeros(k)
    delta = torch.zeros(n, 1)
    w = torch.zeros(n,1)
    gamma = torch.zeros(k, 1)
    v = torch.zeros(k,1)

    for t in range(k):
        ratio = f/g # n
        ratio[torch.isinf(ratio)] = torch.min(ratio)
        ratio[torch.isnan(ratio)] = torch.min(ratio)
        if t >= 1:
            ratio[torch.tensor(L[:t], dtype=torch.long)] = 0 # remove previously selected nodes

        l = torch.argmax(ratio) # index of selected column
        delta = torch.matmul(A.t(), A[:,l]) - torch.matmul(W, W.t()[:,l]) # n*1
        gamma = torch.matmul(B.t(), A[:,l]) - torch.matmul(V, W.t()[:,l]) # c*1
        w = delta / delta[l].sqrt()
        v = gamma / delta[l].sqrt()

        f = f - 2*(w*(torch.matmul(BTA.t(), v)-torch.matmul(torch.matmul(W,V.t()),v))) + torch.sum(v.pow(2)) * (w*w)
        g = g-w*w 

        W[:,t] = w
        V[:,t] = v
        L[t] = l
    
    return L






    




    




    


