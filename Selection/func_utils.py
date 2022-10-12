# This file includes the utility functions
import torch 

def filter_W(W, bound):
    # mask columns that has || ||1 < bound
    col_mean = torch.mean(W.abs(), dim=0)
    col_mask = (col_mean > bound) 
    return col_mask


