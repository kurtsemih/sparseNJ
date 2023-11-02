import os
import sys
import pickle
sys.path.append(os.getcwd())
import time
import numpy as np
import networkx as nx
from collections import defaultdict
from skbio import DistanceMatrix
from skbio.tree import nj
import copy
import argparse
from utils import convert_data_str_to_onehot, newick2nx


def DFS(src, visited, subtree_size, subtree_leaves, parent_array, n, tree):
    visited[src] = True
    n[0] += 1
    subtree_size[src] = 1

    if src < n_leaves:
        subtree_leaves[src].append(src)

    for adj in tree.adj[src]:
        if not visited[adj] and not centroidMarked[adj]:
            DFS(adj, visited, subtree_size, subtree_leaves, parent_array, n, tree)
            subtree_size[src] += subtree_size[adj]
            parent_array[adj] = src
            for leaf in subtree_leaves[adj]:
                subtree_leaves[src].append(leaf)


def getCentroid(src, visited, subtree_size, n, tree):

    is_centroid = True
    visited[src] = True
    heaviest_child = 0

    for adj in tree.adj[src]:
        if not visited[adj] and not centroidMarked[adj]:
            if subtree_size[adj] > n / 2:
                is_centroid = False

            if heaviest_child == 0 or subtree_size[adj] > subtree_size[heaviest_child]:
                heaviest_child = adj

    if is_centroid and n - subtree_size[src] <= n / 2:
        return src

    return getCentroid(heaviest_child, visited, subtree_size, n, tree)


def getCentroidTree(src, tree, subtree_size=None):
    if subtree_size == None:
        visited = [False] * MAXN
        subtree_size = [0] * MAXN
        parent_array = [-1] * MAXN
        subtree_leaves = defaultdict(list)
        n = [0]

        DFS(src, visited, subtree_size, subtree_leaves, parent_array, n, tree)
    else:
        n = [subtree_size[src]]

    visited = [False] * MAXN
    centroid = getCentroid(src, visited, subtree_size, n[0], tree)
    centroidMarked[centroid] = True

    return centroid


def orient_pick(held_out, leaves_at_adj, num_of_ort):
    num_leaf_at_adj = len(leaves_at_adj)
    if num_leaf_at_adj < num_of_ort:
        ort_leaves = []
        for leaf in leaves_at_adj:
            ort_leaves.append(leaf)
        num_leaf_to_sample = num_of_ort - num_leaf_at_adj
        for k in range(num_leaf_to_sample):
            ort_leaves.append(np.random.choice(leaves_at_adj))
    else:
        upp_limit = int(np.log2(n_leaves))
        if num_leaf_at_adj > upp_limit:
            leaves_at_adj_samp = np.random.choice(leaves_at_adj, upp_limit)
        else:
            leaves_at_adj_samp = leaves_at_adj.copy()

        num_leaf_at_adj = len(leaves_at_adj_samp)

        data_leaves_onehot = convert_data_str_to_onehot(data[leaves_at_adj_samp,:])
        data_heldout_onehot = np.squeeze(convert_data_str_to_onehot(data[held_out, :]))

        dist_vector = np.zeros([num_leaf_at_adj])
        for kj in range(num_leaf_at_adj):
            for site in range(data.shape[1]):
                dist_vector[kj] += np.abs(
                    data_heldout_onehot[site, :] - data_leaves_onehot[kj, site, :]).sum()
        dist_vector = dist_vector / (2 * data.shape[1])
        dist_vector = (-3 / 4) * np.log(1 - dist_vector * 4 / 3)
        ort_leaves = np.array(leaves_at_adj_samp)[np.argsort(dist_vector)[0: num_of_ort]]
        ort_leaves = list(ort_leaves)

    return ort_leaves


def decomposeTree(root, tree, held_out, is_last=False, current_degree=0, first_placement=False, subtree_leaves=None,
                  subtree_size=None, parent_array=None, root_parent=-1, dec_thres=0):
    # print('root', root)
    if (
    root, current_degree, tuple(stop_node_dict[root]), root_parent) not in root_centroid_dict and current_degree == 0:
        cend_tree = getCentroidTree(root, tree)
    elif (
    root, current_degree, tuple(stop_node_dict[root]), root_parent) not in root_centroid_dict and current_degree != 0:
        cend_tree = getCentroidTree(root, tree, subtree_size)
        root_centroid_dict[(root, current_degree, tuple(stop_node_dict[root]), root_parent)] = cend_tree
    elif (root, current_degree, tuple(stop_node_dict[root]), root_parent) in root_centroid_dict:
        cend_tree = root_centroid_dict[(root, current_degree, tuple(stop_node_dict[root]), root_parent)]
        # check if the centroid is still up
        if current_degree != 0:
            curr_adj_sizes = []
            curr_adj_list = []
            for adj in tree.adj[cend_tree]:
                curr_adj_sizes.append(subtree_size[adj])
                curr_adj_list.append(adj)

            sorted_adj_sizes = np.sort(np.array(curr_adj_sizes.copy()))
            # print(sorted_adj_sizes)
            sorted_adj_list = np.array(curr_adj_list.copy())
            sorted_adj_list = sorted_adj_list[np.argsort(np.array(curr_adj_sizes))]
            sorted_adj_sizes[-1] = subtree_size[root] - sorted_adj_sizes[0] - sorted_adj_sizes[1] - 1

            if (sorted_adj_sizes > subtree_size[root] / 2).any():
                cend_tree = sorted_adj_list[np.argmax(sorted_adj_sizes)]
                root_centroid_dict[(root, current_degree, tuple(stop_node_dict[root]), root_parent)] = cend_tree

        centroidMarked[cend_tree] = True

    if first_placement and current_degree == 0:
        visited = [False] * MAXN
        subtree_size = [0] * MAXN
        n = [0]
        parent_array = [-1] * MAXN
        subtree_leaves = defaultdict(list)

        DFS(cend_tree, visited, subtree_size, subtree_leaves, parent_array, n, tree)
        root_centroid_dict[(root, current_degree, tuple(stop_node_dict[root]), root_parent)] = cend_tree
        global first_dfs_subtree_leaves
        global first_dfs_subtree_sizes
        global first_parent_array
        first_dfs_subtree_leaves = copy.deepcopy(subtree_leaves)
        first_dfs_subtree_sizes = copy.deepcopy(subtree_size)
        first_parent_array = copy.deepcopy(parent_array)


    orient_leaves_1 = []
    orient_leaves_2 = []
    orient_leaves_3 = []

    subtree_leaf_size = []
    active_adj = []
    if current_degree == 0:
        subtree_all_leaves = np.array(subtree_leaves[cend_tree])
    else:
        subtree_all_leaves = np.array(subtree_leaves[root])


    for adj in tree.adj[cend_tree]:
        # print('adj ' + str(adj))
        if adj != parent_array[cend_tree]:
            leaves_at_adj = np.array(subtree_leaves[adj])

            subtree_leaf_size.append(leaves_at_adj.shape[0])
            subtree_all_leaves = list([leaf for leaf in subtree_all_leaves if leaf not in leaves_at_adj])

            if leaves_at_adj.shape[0] > 0:
                ort_leaves = orient_pick(held_out, leaves_at_adj, 3)
                orient_leaves_1.append(ort_leaves[0])
                orient_leaves_2.append(ort_leaves[1])
                orient_leaves_3.append(ort_leaves[2])
                active_adj.append(adj)
            else:
                leaves_at_adj = np.array(first_dfs_subtree_leaves[adj].copy())
                ort_leaves = orient_pick(held_out, leaves_at_adj, 3)
                orient_leaves_1.append(ort_leaves[0])
                orient_leaves_2.append(ort_leaves[1])
                orient_leaves_3.append(ort_leaves[2])
                active_adj.append(adj)
        else:
            parent_adj = adj


    if len(subtree_all_leaves) > 0:
        subtree_leaf_size.append(len(subtree_all_leaves))
        ort_leaves = orient_pick(held_out, subtree_all_leaves, 3)
        orient_leaves_1.append(ort_leaves[0])
        orient_leaves_2.append(ort_leaves[1])
        orient_leaves_3.append(ort_leaves[2])
        active_adj.append(parent_adj)
    elif current_degree != 0:
        subtree_leaf_size.append(len(subtree_all_leaves))
        grand_parent = parent_array[root]
        leaves_at_grand_parent = np.array(first_dfs_subtree_leaves[grand_parent].copy())
        leaves_at_sibling = list([leaf for leaf in leaves_at_grand_parent if leaf not in first_dfs_subtree_leaves[root]])
        ort_leaves = orient_pick(held_out, leaves_at_sibling, 3)
        orient_leaves_1.append(ort_leaves[0])
        orient_leaves_2.append(ort_leaves[1])
        orient_leaves_3.append(ort_leaves[2])
        active_adj.append(parent_adj)

    orient_leaves_1.append(held_out)
    orient_leaves_2.append(held_out)
    orient_leaves_3.append(held_out)

    dist_matrix = np.zeros([3, len(orient_leaves_1), len(orient_leaves_1)])
    for i in range(3):
        if i == 0:
            orient_leaves = orient_leaves_1
        elif i == 1:
            orient_leaves = orient_leaves_2
        elif i == 2:
            orient_leaves = orient_leaves_3

        data_quartet = data[orient_leaves, :]
        data_quart_onehot = convert_data_str_to_onehot(data_quartet)

        for ki in range(len(orient_leaves)):
            for kj in range(ki+1, len(orient_leaves)):
                for site in range(data.shape[1]):
                    dist_matrix[i, ki, kj] += np.abs(data_quart_onehot[ki, site, :] - data_quart_onehot[kj, site, :]).sum()

        for ki in range(len(orient_leaves)):
            for kj in range(ki + 1, len(orient_leaves)):
                dist_matrix[i, kj, ki] = dist_matrix[i, ki, kj]

    dist_matrix = dist_matrix / (2 * data.shape[1])
    dist_matrix = (-3 / 4) * np.log(1 - (dist_matrix * 4 / 3))
    dist_matrix = dist_matrix.mean(axis=0)
    dm = DistanceMatrix(dist_matrix)
    NJ_tree = nj(dm)

    held_out_idx = len(orient_leaves_1) - 1
    selected_path = int(NJ_tree.find(str(held_out_idx)).siblings()[0].name)
    selected_adj = active_adj[selected_path]


    if not centroidMarked[selected_adj]:
        if selected_adj == parent_array[cend_tree]:
            grand_parent = parent_array[cend_tree]
            while grand_parent != -1:
                subtree_leaves[grand_parent] = list(
                    [leaf for leaf in subtree_leaves[grand_parent] if leaf not in subtree_leaves[cend_tree]])
                subtree_size[grand_parent] -= subtree_size[cend_tree]
                stop_node_dict[grand_parent].append(cend_tree)
                grand_parent = parent_array[grand_parent]

            subtree_leaves[cend_tree] = []
            subtree_size[cend_tree] = 0

            selected_adj = root

    if selected_adj != root:
        root_parent = cend_tree
    if selected_adj == root and current_degree == 0:
        root_parent = cend_tree

    current_degree += 1

    if selected_adj >= n_leaves and not centroidMarked[selected_adj]:
        return decomposeTree(selected_adj, tree, held_out, current_degree=current_degree, subtree_leaves=subtree_leaves,
                             subtree_size=subtree_size, parent_array=parent_array, root_parent=root_parent, dec_thres=dec_thres)
    else:
        if selected_adj < n_leaves:
            next_selected_adj = parent_array[selected_adj]
        elif centroidMarked[selected_adj]:
            next_selected_adj = cend_tree

        return selected_adj, next_selected_adj, current_degree

##### MAIN ######
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help=' ds1 | ds2 | ... | ds8 | virus ')
    parser.add_argument('--num_of_seed', required=True, help=' 5 | 10 | 20  ')
    args = parser.parse_args()

    data_real = np.load('data/'+args.dataset+'_data.npy')
    n_leaves = data_real.shape[0]
    initial_n_leaves = int(np.floor(np.sqrt(n_leaves*np.log2(n_leaves))))
    num_of_heldout = n_leaves-initial_n_leaves
    num_of_seed = int(args.num_of_seed)
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

        data = data_real
        our_tic = time.time()
        data_quartet = data[initial_leaves, :]  # data[0:4, :]
        data_onehot = convert_data_str_to_onehot(data_quartet)
        dist_matrix = np.zeros([initial_n_leaves, initial_n_leaves])  # np.zeros([4, 4])

        for ki in range(initial_n_leaves):  # 4
            for kj in range(ki, initial_n_leaves):
                for site in range(data.shape[1]):
                    dist_matrix[ki, kj] += np.abs(data_onehot[ki, site, :] - data_onehot[kj, site, :]).sum()
        for ki in range(initial_n_leaves):
            for kj in range(ki + 1, initial_n_leaves):
                dist_matrix[kj, ki] = dist_matrix[ki, kj]

        dist_matrix = dist_matrix / (2 * data.shape[1])
        dist_matrix = (-3 / 4) * np.log(1 - dist_matrix * 4 / 3)

        dm = DistanceMatrix(dist_matrix)
        tree_newick = nj(dm, result_constructor=str)
        tree = newick2nx(tree_newick, initial_n_leaves)

        mapping = {}
        for k in range(0, initial_n_leaves):
            mapping[k] = initial_leaves[k]  # 2 * initial_n_leaves  - 2 -
        new_int_node = n_leaves
        for k in range(initial_n_leaves, 2 * initial_n_leaves - 2):
            mapping[k] = new_int_node
            new_int_node += 1

        tree = nx.relabel_nodes(tree, mapping)
        root = new_int_node - 1

        first_insertion = True
        for held_out in held_out_leaves:
            centroidMarked = [False] * MAXN
            stop_node_dict = defaultdict(list)

            if first_insertion:  # case_idx % num_of_heldout == 0:
                selected_adj, other_selected_adj, current_degree = decomposeTree(root, tree, held_out,
                                                                                 first_placement=True)
                first_insertion = False
            else:
                selected_adj, other_selected_adj, current_degree = decomposeTree(root, tree, held_out,
                                                                                 subtree_leaves=first_dfs_subtree_leaves.copy(),
                                                                                 subtree_size=first_dfs_subtree_sizes.copy(),
                                                                                 parent_array=first_parent_array.copy(),
                                                                                 )

            if other_selected_adj == first_parent_array[selected_adj]:
                upper_neighbour = other_selected_adj
                lower_neighbour = selected_adj
            elif selected_adj == first_parent_array[other_selected_adj]:
                upper_neighbour = selected_adj
                lower_neighbour = other_selected_adj

            tree.remove_edge(upper_neighbour, lower_neighbour)
            first_centroid = root_centroid_dict[(root, 0, tuple([]), -1)]

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

            new_int_node += 1
            # CHECK CENTROID
            current_size = tree.number_of_nodes()
            adj_sizes = np.array([first_dfs_subtree_sizes[adj] for adj in tree.adj[first_centroid]])
            fc_adj_list = np.array([adj for adj in tree.adj[first_centroid]])
            if (adj_sizes > current_size / 2).any():
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
        pickle.dump(tree, open(
            'results/' + args.dataset + '_SNJ_tree_seed' + str(seed) + '.pickle',
            'wb'))
        
        print('*******************')
        print('DONE=>>>>> ' + 'DS ' + args.dataset + ' seed= ' + str(seed))
        print('*******************')
