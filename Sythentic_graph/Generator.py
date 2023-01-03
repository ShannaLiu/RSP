import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import os
from sklearn.utils import resample
from scipy.linalg import block_diag

def synthetic_graph_generator(list_shapes, list_shapes_args, graph_type, graph_args, plot=False, plot_color='group_label'):
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
    
    return G, Gg, group_label, shape_label, shape_dist

    
# Based on the subspace assumption
def clean_feature_generator(group_label, dim_features, dim_latent_features, orthogonal=False, graph_based=False, G0=None, k=None, rho=None):
    '''
    dim_features : D, the dimension of features
    dim_latent_features : r/d = sum d_i, the sum of dimension of all latent features
    dim_features > dim_latent_features > num_group
    orthogonal: whether the columns in U are orthogonal (not necessary)
    G0 : true correlation for generating the features
    '''
    num_unique_group = len(np.unique(group_label))
    if dim_latent_features < num_unique_group:
        raise Exception('number of latent features needs to be larger than number of unique groups to have independent subsapce')
    if dim_latent_features > dim_features:
        raise Exception('number of features needs to be larger than number of latent features')

    num_group = np.unique(group_label, return_counts=True)[1]
    sub_dim = random_partition_generator(dim_latent_features, num_unique_group)
    cond = 1e10
    # To make sure the random matrix is of full rank
    while cond > 100:
        rand_mat = np.random.randn(dim_features,dim_latent_features)
        cond = np.linalg.cond(rand_mat)
    if orthogonal:
        rand_mat, _ = np.linalg.qr(rand_mat, mode='reduced')
    S = rand_mat
    
    if graph_based:
        if (rho is None) | (G0 is None) | (k is None):
            raise Exception('rho and G are needed to generate features based on graph stucture')
        if sum(num_group) != G0.number_of_nodes():
            raise Exception('number of nodes does not match with the graph')

    Y = random_latent_rep_generator(sub_dim, num_group, graph_based, G0, k, rho)
    node_features = (S @ Y).T # N * d
    return S, Y, node_features

def noisy_feature_generator(group_label, dim_features, dim_latent_features, std=1.0, orthogonal=False, graph_based=False, G0=None, k=None, rho=None):
    '''
    std of Gaussian noise
    '''
    U, latent_features, node_features = clean_feature_generator(group_label, dim_features, dim_latent_features, orthogonal, graph_based, G0, k, rho)
    node_features = node_features + np.random.randn(node_features.shape[0], node_features.shape[1]) * std
    return U, latent_features, node_features


def random_latent_rep_generator(sub_dim, num_group, graph_based=False, G0=None, k=None, rho=None):
    '''
    generate random latent features
    input:
        num_group: number of nodes in each group
        sub_dim: demension for each subspace
        Y: sum_dim * num_nodes 
    output: 
    '''
    if graph_based:
        cov = cov_with_args(G0, k, rho) # N*N
        cov0 = cov[:num_group[0], :num_group[0]]
        Y = np.random.multivariate_normal(mean = np.zeros(cov0.shape[0]), cov=cov0, size=sub_dim[0])
        for i in range(1,len(sub_dim)):
            csum = np.cumsum(num_group[:i])[0]
            cov0 = cov[csum:csum+num_group[i], csum:csum+num_group[i]]
            Y = block_diag(Y, np.random.multivariate_normal(mean = np.zeros(cov0.shape[0]), cov=cov0, size=sub_dim[i]))
    else : 
        Y = np.random.rand(sub_dim[0], num_group[0])
        for i in range(1,len(sub_dim)):
            Y = block_diag(Y, np.random.rand(sub_dim[i],num_group[i]))
    return Y

def adj_to_dist(G):
    '''
    From adjacency matrix to minimum path length matrix
    '''
    length=dict(nx.all_pairs_dijkstra_path_length(G))
    N = G.number_of_nodes()
    Dist = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            try :
                Dist[i,j] = length[i][j]
            except:
                Dist[i,j] = np.inf 
    return Dist


def cov_with_args(G, k, rho):
    '''
    From adjacency matrix to covariance matrix
    '''
    sigmas = rho ** np.arange(k+1)
    N = G.number_of_nodes()
    Dist = adj_to_dist(G)
    cov = np.zeros((N,N))
    for i in range(k+1):
        cov[Dist==i] = sigmas[i]
    return cov


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


### Based on the center assumption (special case for the subspace assumption)

def clean_deg_feature_generator(group_label, num_features):
    num_unique_group = len(np.unique(group_label))
    group_mean = torch.randn(num_unique_group, num_features)
    node_features = torch.zeros(len(group_label), num_features)
    for i in range(node_features.shape[0]):
        node_features[i,] = torch.tensor(group_mean[group_label[i],])
    return node_features


def noisy_deg_feature_generator(group_label, num_features, std=1):
    num_unique_group = len(np.unique(group_label))
    group_mean = torch.randn(num_unique_group, num_features)
    node_features = torch.zeros(len(group_label), num_features)
    for i in range(node_features.shape[0]):
        node_mean = torch.tensor(group_mean[group_label[i],])
        node_features[i,] = torch.distributions.MultivariateNormal(node_mean, std*torch.eye(node_mean.shape[0])).sample()
    return node_features, group_mean








def repeated_feature_generator(group_label, num_repeated, num_features, std=1):
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
    node_features, group_mean = synthetic_feature_generator(new_group_label, num_features, std)
    return node_features, group_mean, new_group_label

def separated_feature_generator(group_label, num_separated, num_features, std=1):
    new_group_label = np.int0(np.array(group_label))
    label_unique = np.unique(group_label)
    sep_label = resample(label_unique, replace=False, n_samples=num_separated) # seperated label
    for i in range(num_separated):
        sep_group_size = len(new_group_label[group_label==sep_label[i]])
        label_change = resample(range(sep_group_size), replace=False, n_samples=np.int0(np.floor(sep_group_size/2)))
        new_group_label[group_label==sep_label[i]] = np.concatenate(([np.max(new_group_label) + 1] * len(label_change), [sep_label[i]] * (len(new_group_label[group_label==sep_label[i]]) - len(label_change))))
    node_features, group_mean = synthetic_feature_generator(new_group_label, num_features, std)
    return node_features, group_mean, new_group_label




