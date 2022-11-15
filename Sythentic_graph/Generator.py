import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch


def sythetic_graph_generator(list_shapes, list_shapes_args, graph_type, graph_args, plot=False, plot_color='group_label', savefig=True):
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
        nx.draw(G, node_color=eval(plot_color))

        if savefig:
            plt.savefig("plots/structure.png")

    if (savefig == True) & (plot==False):
        raise Exception("Please save figure afer plot")
    
    return G, Gg, group_label, shape_label, shape_dist

    
def sythetic_feature_generator(group_label, num_features, std=1):
    num_unique_group = len(np.unique(group_label))
    group_mean = torch.randn(num_unique_group, num_features)
    node_features = torch.zeros(len(group_label), num_features)
    for i in range(node_features.shape[0]):
        node_mean = torch.tensor(group_mean[group_label[i],])
        node_features[i,] = torch.normal(node_mean, std=std)
    return node_features, group_mean



