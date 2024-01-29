import os
import sys
import pickle
sys.path.append(os.getcwd())
import time
import numpy as np
from utils import newick2nx, convert_data_str_to_onehot
from skbio import DistanceMatrix
from skbio.tree import nj
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help=' 16SE | 23SB2k | ... ')
args = parser.parse_args()

data = np.load('data/'+args.dataset+'_data.npy', allow_pickle=False)


n_leaves = data.shape[0]
seq_length = data.shape[1]

data_onehot = convert_data_str_to_onehot(data)


way_1_tic = time.process_time()
dist_matrix = np.zeros([n_leaves, n_leaves])
for ki in range(n_leaves):
    dist_matrix[ki, ki+1:] += np.abs(data_onehot[ki+1:, :, :] - data_onehot[ki, :, :]).sum(axis=-1).sum(axis=-1)
dist_matrix += dist_matrix.T
dist_matrix = dist_matrix / (2 * data.shape[1])
dist_matrix = (-3 / 4) * np.log(1 - dist_matrix * 4 / 3)
way_1_toc = time.process_time()

NJ_tic = time.process_time()
dm = DistanceMatrix(dist_matrix)
NJ_tree = nj(dm, result_constructor=str)
NJ_toc = time.process_time()
NJ_nx = newick2nx(NJ_tree, n_leaves)


pickle.dump(NJ_nx, open('results/'+ args.dataset +'_NJ_tree.pickle', 'wb'))

print('*******************')
print('DONE=>>>>> ' + args.dataset )
print('Distance matrix time= ', way_1_toc-way_1_tic)
print('NJ clustering time= ', NJ_toc-NJ_tic)
print('*******************')
