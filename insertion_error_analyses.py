import os
import sys
import pickle
sys.path.append(os.getcwd())
import time
import numpy as np
import networkx as nx
from collections import defaultdict
from sparseNJ import decomposeTree

##### MAIN ######
num_of_seed = 5
data_real = np.load('data/virus_data.npy')

n_leaves = data_real.shape[0]
initial_n_leaves = int(np.floor(np.sqrt(n_leaves*np.log2(n_leaves))))
num_of_heldout = n_leaves-initial_n_leaves
dist_in_edge = np.zeros([num_of_seed, num_of_heldout])
seq_length = data_real.shape[1]
for seed in range(num_of_seed):
    np.random.seed(seed)

    MAXN = 2 * n_leaves - 2

    first_dfs_subtree_sizes = [0] * MAXN  # []#
    first_dfs_subtree_leaves = defaultdict(list)  # []#
    first_parent_array = [-1] * MAXN

    root_centroid_dict = {}
    parent_node_dict = defaultdict()
    parent_adj_dict = defaultdict()

    perm_idx = np.random.permutation(n_leaves)
    first_leaf_at_rand = perm_idx[0]

    # RANDOM OPTION
    initial_leaves = perm_idx[0:initial_n_leaves]
    held_out_leaves = perm_idx[initial_n_leaves:]

    NJ_tree = pickle.load(open('results/virus_NJ_tree.pickle', 'rb'))
    root = MAXN - 1
    for held_out in held_out_leaves:
        parent_held_out = nx.predecessor(NJ_tree, root)[held_out][0]
        parent_node_dict[held_out] = parent_held_out

        NJ_tree.remove_node(held_out)
        t = 0
        nodes_to_connect = []
        for adj in NJ_tree.adj[parent_held_out]:
            nodes_to_connect.append(adj)
            t += NJ_tree.edges[(adj, parent_held_out)]['t']

        NJ_tree.remove_node(parent_held_out)
        NJ_tree.add_edge(nodes_to_connect[0], nodes_to_connect[1], t=t)

        parent_adj_dict[held_out] = nodes_to_connect

        if parent_held_out == root:
            root -= 1
            while not NJ_tree.has_node(root):
                root -= 1


    data = data_real
    our_tic = time.time()
    tree = NJ_tree.copy()

    first_insertion = True
    held_out_idx = -1
    for held_out in reversed(held_out_leaves):
        held_out_idx += 1

        centroidMarked = [False] * MAXN
        stop_node_dict = defaultdict(list)

        if first_insertion: #case_idx % num_of_heldout == 0:
            selected_adj, other_selected_adj, current_degree = decomposeTree(root, tree, held_out,
                                                                             first_placement=True)
            first_insertion = False
        else:
            selected_adj, other_selected_adj, current_degree = decomposeTree(root, tree, held_out,
                                                                             subtree_leaves=first_dfs_subtree_leaves.copy(),
                                                                             subtree_size=first_dfs_subtree_sizes.copy(),
                                                                             parent_array=first_parent_array.copy(),
                                                                             )

        num_of_edges = np.zeros([4])
        num_of_edges[0] = nx.shortest_path_length(tree, source=selected_adj, target=parent_adj_dict[held_out][0])
        num_of_edges[1] = nx.shortest_path_length(tree, source=other_selected_adj,
                                                  target=parent_adj_dict[held_out][0])
        num_of_edges[2] = nx.shortest_path_length(tree, source=selected_adj, target=parent_adj_dict[held_out][1])
        num_of_edges[3] = nx.shortest_path_length(tree, source=other_selected_adj,
                                                  target=parent_adj_dict[held_out][1])
        dist_in_edge[seed, held_out_idx] = np.min(num_of_edges)
        #print('current dist for held out:', held_out, ' = ' + str(np.min(num_of_edges)))

        # PLACE IT TO THE CORRECT PLACE
        selected_adj = parent_adj_dict[held_out][0]
        other_selected_adj = parent_adj_dict[held_out][1]

        if other_selected_adj == first_parent_array[selected_adj]:
            upper_neighbour = other_selected_adj
            lower_neighbour = selected_adj
        elif selected_adj == first_parent_array[other_selected_adj]:
            upper_neighbour = selected_adj
            lower_neighbour = other_selected_adj

        tree.remove_edge(upper_neighbour, lower_neighbour)
        new_int_node = parent_node_dict[held_out]
        first_centroid = root_centroid_dict[(root, 0, tuple([]), -1)]
        # print('NEW INT NODE=' + str(new_int_node))

        tree.add_node(new_int_node, parent=upper_neighbour)
        tree.add_edge(upper_neighbour, new_int_node, t=np.random.exponential(0.1))
        tree.add_node(lower_neighbour, parent=new_int_node)
        tree.add_edge(new_int_node, lower_neighbour, t=np.random.exponential(0.1))
        first_dfs_subtree_leaves[upper_neighbour].append(held_out)
        first_dfs_subtree_sizes[upper_neighbour] += 2
        grand_parent = first_parent_array[upper_neighbour]
        while grand_parent != -1:
            first_dfs_subtree_leaves[grand_parent].append(held_out)
            first_dfs_subtree_sizes[grand_parent] += 2
            grand_parent = first_parent_array[grand_parent]

        first_dfs_subtree_leaves[held_out].append(held_out)
        first_dfs_subtree_sizes[held_out] = 1

        first_dfs_subtree_leaves[new_int_node] = first_dfs_subtree_leaves[lower_neighbour].copy()
        first_dfs_subtree_leaves[new_int_node].append(held_out)
        first_dfs_subtree_sizes[new_int_node] = 2 + first_dfs_subtree_sizes[lower_neighbour]

        first_parent_array[lower_neighbour] = new_int_node
        first_parent_array[held_out] = new_int_node
        first_parent_array[new_int_node] = upper_neighbour

        tree.add_node(held_out, parent=new_int_node)
        tree.add_edge(new_int_node, held_out, t=np.random.exponential(0.1))

        # CHECK CENTROID
        current_size = tree.number_of_nodes()
        adj_sizes = np.array([first_dfs_subtree_sizes[adj] for adj in tree.adj[first_centroid]])
        # print('ADJ SIZES' + str(adj_sizes))
        fc_adj_list = np.array([adj for adj in tree.adj[first_centroid]])
        # print(adj_sizes)
        if (adj_sizes > current_size / 2).any():
            # print('first centroid coktu')
            # DFS UPDATE
            new_first_centroid = fc_adj_list[np.argmax(adj_sizes)]
            all_leaves = first_dfs_subtree_leaves[first_centroid].copy()

            first_dfs_subtree_sizes[first_centroid] -= first_dfs_subtree_sizes[new_first_centroid]
            first_dfs_subtree_leaves[first_centroid] = list(
                [node for node in first_dfs_subtree_leaves[first_centroid] if
                 node not in first_dfs_subtree_leaves[new_first_centroid]])

            first_dfs_subtree_sizes[new_first_centroid] = current_size
            first_dfs_subtree_leaves[new_first_centroid] = all_leaves

            first_parent_array[first_centroid] = new_first_centroid
            first_parent_array[new_first_centroid] = -1
            # CENTROID UPDATE
            root_centroid_dict[(root, 0, tuple([]), -1)] = new_first_centroid

    our_toc = time.time()

    print('*******************')
    print('DONE=>>>>> ' + ' seed= ' + str(seed))
    print('*******************')

np.save('results/dist_in_edge', dist_in_edge)