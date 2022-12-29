import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import os
from sklearn.utils import resample
from scipy.linalg import block_diag

def synthetic_graph_generator(list_shapes, list_shapes_args, graph_type, graph_args, plot=False, plot_color='group_label', savefig=False, root=None, figname=None):
    '''
    list_shapes: list of shapes of each subgraph, should correspond with class names in networkx, examples in Vis_list_type
    graph_type: type of graph used to conect the subgraphs, should correspond with function names in netwokx
    graph_args: arguments for the graph used to connect subgraphs
    '''
    G = nx.Graph()
    shape_unique = np.unique(list_shapes)
    shape_dist = {}
    for i in range(len(shape_unique)):
        shape_dist[shape_unique[i]] = i
    shape_label = []
    group_label = []
    n_previous_nodes = 0

    # Generate the subgraphs
    for i in range(len(list_shapes)):
        shape = list_shapes[i]
        shape_arg = list_shapes_args[i]
        subgraph = eval(shape)(*shape_arg)
        subgraph = nx.relabel_nodes(subgraph, lambda x: x + n_previous_nodes)
        G.add_nodes_from(subgraph.nodes())
        G.add_edges_from(subgraph.edges())
        n_nodes = subgraph.number_of_nodes()
        n_previous_nodes += n_nodes
        shape_label += [shape_dist[shape]]*n_nodes 
        group_label += [i]*n_nodes
    
    # Generate the upper-level graph for connecting
    Gg=eval(graph_type)(*graph_args)
    if Gg.number_of_nodes() != len(list_shapes):
        raise Exception("Number of nodes in graph don't match number of subgraphs")

    elist = []
    for e in Gg.edges():
        if e not in elist:
            ii=np.random.choice(np.where(np.array(group_label)==(e[0]))[0],1)[0]
            jj=np.random.choice(np.where(np.array(group_label)==(e[1]))[0],1)[0]
            G.add_edges_from([(ii,jj)])
            elist+=[e]
            elist+=[(e[1],e[0])]

    if plot:
        if plot_color not in ['group_label', 'shape_label']:
            raise Exception("Wrong plot_color, please change to 'group_label' or 'shape_label'")
        nx.draw(G, node_color=eval(plot_color), node_size=50)
        if savefig:
            if root != None:
                root = root + 'Synthetic'
                if not os.path.exists(root):
                    os.makedirs(root)
                current = os.getcwd()
                os.chdir(root)
                plt.savefig(figname+'.png')
                nx.write_adjlist(G, 'G.adjlist')
                np.save('group_label.npy', group_label)
                np.save('shape_label.npy', shape_label)
                os.chdir(current) # Move back to current directory


    if (savefig == True) & (plot==False):
        raise Exception("Please save figure afer plot")
    
    return G, Gg, group_label, shape_label, shape_dist

    
def synthetic_feature_generator(group_label, num_features, std=1, save=False, root=None):
    num_unique_group = len(np.unique(group_label))
    group_mean = torch.randn(num_unique_group, num_features)
    node_features = torch.zeros(len(group_label), num_features)
    for i in range(node_features.shape[0]):
        node_mean = torch.tensor(group_mean[group_label[i],])
        node_features[i,] = torch.distributions.MultivariateNormal(node_mean, std*torch.eye(node_mean.shape[0])).sample()
    if save:
        if root != None:
            root = root + 'Synthetic' 
            if not os.path.exists(root):
                os.makedirs(root)
            current = os.getcwd()
            os.chdir(root)
            np.save('node_features.npy', node_features)
            np.save('group_mean.npy', group_mean)
            os.chdir(current) # Move back to current directory
    return node_features, group_mean

def repeated_feature_generator(group_label, num_repeated, num_features, std=1, save=False, root=None):
    new_group_label = np.int0(np.array(group_label))
    label_unique = np.unique(group_label)
    num_unique_group = len(label_unique) - num_repeated
    if num_repeated >= num_unique_group:
        raise Exception('number of repeated clusters can not be larger than real clusters')
    del_label = resample(label_unique, replace=False, n_samples=num_repeated) # delete label
    rep_label = resample(label_unique[label_unique!=del_label], replace=True, n_samples=num_repeated) # replace label
    for i in range(num_repeated):
        new_group_label[group_label==del_label[i]] = rep_label[i]
    for i in range(num_repeated):
        current_max = np.max(new_group_label)
        if (del_label[i] < current_max):
            new_group_label[new_group_label==current_max] = del_label[i]
    node_features, group_mean = synthetic_feature_generator(new_group_label, num_features, std, save, root)
    return node_features, group_mean, new_group_label

def separated_feature_generator(group_label, num_separated, num_features, std=1, save=False, root=None):
    new_group_label = np.int0(np.array(group_label))
    label_unique = np.unique(group_label)
    sep_label = resample(label_unique, replace=False, n_samples=num_separated) # seperated label
    for i in range(num_separated):
        sep_group_size = len(new_group_label[group_label==sep_label[i]])
        label_change = resample(range(sep_group_size), replace=False, n_samples=np.int0(np.floor(sep_group_size/2)))
        new_group_label[group_label==sep_label[i]] = np.concatenate(([np.max(new_group_label) + 1] * len(label_change), [sep_label[i]] * (len(new_group_label[group_label==sep_label[i]]) - len(label_change))))
    node_features, group_mean = synthetic_feature_generator(new_group_label, num_features, std, save, root)
    return node_features, group_mean, new_group_label

def clean_feature_generator(group_label, num_features):
    num_unique_group = len(np.unique(group_label))
    group_mean = torch.randn(num_unique_group, num_features)
    node_features = torch.zeros(len(group_label), num_features)
    for i in range(node_features.shape[0]):
        node_features[i,] = torch.tensor(group_mean[group_label[i],])
    return node_features
    
def sub_clean_feature_generator(group_label, num_features, num_latent_features, orthogonal=False):
    '''
    num_features : D, the dimension of features
    num_latent_features : r/d = sum d_i, the sum of dimension of all latent features
    num_features > num_latent_features > num_group
    orthogonal: whether the columns in U are orthogonal (not necessary)
    '''
    num_unique_group = len(np.unique(group_label))
    if num_latent_features < num_unique_group:
        raise Exception('number of latent features needs to be larger than number of unique groups to have independent subsapce')
    if num_latent_features > num_features:
        raise Exception('number of features needs to be larger than number of latent features')
    num_group = np.unique(group_label, return_counts=True)[1]
    sub_dim = random_partition_generator(num_latent_features, num_unique_group)
    cond = 1e10
    # To make sure the random matrix is of full rank
    while cond > 100:
        rand_mat = np.random.randn(num_features,num_features)
        cond = np.linalg.cond(rand_mat)
    if orthogonal:
        rand_mat, _ = np.linalg.qr(rand_mat)
    U = rand_mat[:,:num_latent_features]
    
    latent_features = random_latent_rep_generator(sub_dim, num_group)
    node_features = (U @ latent_features).T # N * d
    return U, latent_features, node_features

def sub_noisy_feature_generator(group_label, num_features, num_latent_features, std=1.0, orthogonal=False):
    '''
    std of Gaussian noise
    '''
    U, latent_features, node_features = sub_clean_feature_generator(group_label, num_features, num_latent_features, orthogonal)
    node_features = node_features + np.random.randn(node_features.shape[0], node_features.shape[1]) * std
    return U, latent_features, node_features


def random_partition_generator(sum, num):
    '''
    generate a positive vector of length num, with fixed sum
    '''
    if sum == num:
        nums = np.int0(np.ones(num))
    else:
        nums = []
        for i in range(num-1):
            nums.append(np.random.randint(sum-num-np.sum(nums)))
        nums.append(sum-num-np.sum(nums))
        nums = np.int0(nums + np.ones(num))
    return nums

def random_latent_rep_generator(sub_dim, num_group):
    '''
    generate random latent features
    num_group: number of nodes in each group
    '''
    S = np.random.rand(sub_dim[0], num_group[0])
    for i in range(1,len(sub_dim)):
        S = block_diag(S, np.random.rand(sub_dim[i],num_group[i]))
    return S



