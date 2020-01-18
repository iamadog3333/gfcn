#import utils
import process
import numpy as np
import pickle

#adj, features, labels, idx_train, idx_val, idx_test = process.load_data()
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = process.load_data_with_label('citeseer')

idx_train = np.nonzero(train_mask)
idx_val = np.nonzero(val_mask)
idx_test = np.nonzero(test_mask)
labels = np.argmax(labels, 1)
adj = adj.todense()
features = features.todense()

print('adj.shape',adj.shape)
print('features.shape',features.shape)
print("labels ", labels.shape)


load_dict = {}
load_dict['adj'] = adj
load_dict['features'] = features
load_dict['labels'] = labels
load_dict['idx_train'] = idx_train
load_dict['idx_val'] = idx_val
load_dict['idx_test'] = idx_test

dump_file = 'citeseer_data.pkl'
pickle.dump(load_dict, open(dump_file, "wb"))

exit(0)

def gen_two_idx(used_path_idx_set, start, lens):
    # generat i j. start <= i < j < len , not in the set. if not find, one of ij will be -1 
    i = -1
    j = -1
    for idx in range(start, lens):
        if idx not in used_path_idx_set:
            i = idx
    for idx in range(i+1, lens):
        if idx not in used_path_idx_set:
            j = idx
    return i, j


def mirror_padding_paths(path_list):
    # 012  --> 21012
    ret_paths = []
    if len(path_list) == 0:
        return ret_paths
    for path in path_list:
        half_path = path[1::]
        #print('half_path', half_path)
        half_path.reverse()
        #print('half_path', half_path)
        #print('reverse path', path, path[1::], half_path)
        pad_path = half_path + path
        ret_paths.append(pad_path)
    return ret_paths

def add_paths(path_list):
    # path 3 2 5    paht 3 7 8 ---> 5 2 3 7 8
    used_path_idx_set = set()
    ret_paths = []
    for start in range(len(path_list)):
        idx1, idx2 = gen_two_idx(used_path_idx_set, start, len(path_list))
        if idx1 == -1 or idx2 == -1:
            break  #
        while idx2 < len(path_list):
            path1 = path_list[idx1]
            path2 = path_list[idx2]
            set1 = set(path1[1::])
            set2 = set(path2[1::])
            if len(set1 & set2) == 0:
                half_path = path1[1::]
                half_path.reverse()
                merge_path = half_path + path2
                used_path_idx_set.add(idx1)
                used_path_idx_set.add(idx2)
                ret_paths.append(merge_path)
                idx2 = len(path_list)
            idx2 += 1
    return ret_paths



def gen_paths(G, source_node=0):
    # generate some path in the graph
    edges_list = list(nx.dfs_edges(G, source=source_node, depth_limit=2))
    #print(edges_list)
    if len(edges_list) == 0:
        print(' this source_node find no neighbor ', source_node)
        print(edges_list)

    # creat a small graph
    G_small = nx.Graph()
    for edge in edges_list:
        G_small.add_edge(edge[0], edge[1])
    node_set = set()
    for node1, node2 in edges_list:
        node_set.add(node1)
        node_set.add(node2)
    if source_node in node_set:
        node_set.remove(source_node)

    #print(node_set)
    paths_len3 = []
    paths_len2 = []
    for node in node_set:
        paths = list(nx.shortest_simple_paths(G_small, source_node, node))
        for path in paths:
            if len(path) == 3:
                paths_len3.append(path)
            elif len(path) == 2:
                #print('len 2 path', path)
                paths_len2.append(path + [path[-1]])  #padding path len 2 to 3

    #print('paths_len3', paths_len3)
    #print('paths_len2', paths_len2)
    if len(paths_len3)==0 and len(paths_len2)==0:
        print('no edge of node:', source_node)
        print(edges_list)
        paths_len3 = [[source_node, source_node, source_node]]
    merge_paths = add_paths(paths_len3)
    # print('paths_len3', merge_paths)
    # print(len(merge_paths))
    if len(merge_paths) < 10:
        merge_paths = merge_paths + mirror_padding_paths(paths_len3)
        # print('paths_len3 mirror', merge_paths)
        # print(len(merge_paths))
    if len(merge_paths) < 10:

        merge_paths = merge_paths + add_paths(paths_len2)
        # print('paths_len2', merge_paths)
        # print(len(merge_paths))
    if len(merge_paths) < 10:
        merge_paths = merge_paths + mirror_padding_paths(paths_len2)
    while len(merge_paths) < 10:
        merge_paths = merge_paths + merge_paths

    merge_paths = merge_paths[0:10]
    # print('final merge_paths', merge_paths)
    # print(len(merge_paths))
    # exit(0)


    return merge_paths


# def gen_short_paths(G, source_node=0, expected_path_len=5):
#     # generate some path in the graph
#     edges_list = list(nx.dfs_edges(G, source=source_node, depth_limit=expected_path_len-1))
#     #print(edges_list)

#     # creat a small graph
#     G_small = nx.Graph()
#     for edge in edges_list:
#         G_small.add_edge(edge[0], edge[1])
#     node_set = set()
#     for node1, node2 in edges_list:
#         node_set.add(node1)
#         node_set.add(node2)
#     node_set.remove(source_node)

#     #print(node_set)
#     ret_paths = []
#     for node in node_set:
#         paths = list(nx.shortest_simple_paths(G_small, source_node, node))
#         for path in paths:
#             ret_paths.append(path)
#     return ret_paths


# create a graph
import networkx as nx
G = nx.Graph()
# print(adj) it is a sparse matrix format, transform needed.
adj = adj.todense()
adj = np.array(adj)
print('adj shape', adj.shape)
nonzero_idx = np.transpose(np.nonzero(adj))
for idx in nonzero_idx:
    G.add_edge(idx[0], idx[1])
print('graph node:', G.number_of_nodes())
print('graph edge:', G.number_of_edges())

short_path_nodes = []
decomposed_paths = []
for i in range(G.number_of_nodes()):  #G.number_of_nodes()
    print('decomposing node ', i)
    ret_paths = gen_paths(G, source_node=i)
    decomposed_paths.append(ret_paths) 
    if len(ret_paths) == 0:
        short_path_nodes.append(i)
print('decomposed_paths len ', len(decomposed_paths))
print('short_path_nodes ', short_path_nodes )

# short_paths = []
# for node in short_path_nodes:
#     print('decomposing node', node)
#     ret_paths = gen_short_paths(G, source_node=node)
#     short_paths = short_paths + ret_paths
# print('short_paths len ', len(short_paths))
# print('short_path_nodes num ', len(short_path_nodes) )

dump_file = 'decomposed_paths_central_rectangle_citeseer'
dump_dict = {'decomposed_paths':decomposed_paths}
pickle.dump(dump_dict, open(dump_file, "wb"))
load_dict = pickle.load(open(dump_file, "rb"))
print('decomposed_paths len ', len(load_dict['decomposed_paths'] ))
#print('short_paths ', load_dict['short_paths']  )




