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
parser.add_argument('--dataset', required=True, help=' ds1 | ds2 | ... | ds8 | virus ')
args = parser.parse_args()

data = np.load('data/'+args.dataset+'_data.npy')

NJ_tic = time.time()
n_leaves = data.shape[0]
seq_length = data.shape[1]

data_onehot = convert_data_str_to_onehot(data)
dist_matrix = np.zeros([n_leaves, n_leaves])

for ki in range(n_leaves):
    for kj in range(ki+1, n_leaves):
        for site in range(data.shape[1]):
            dist_matrix[ki, kj] += np.abs(data_onehot[ki, site, :] - data_onehot[kj, site, :]).sum()
for ki in range(n_leaves):
    for kj in range(ki + 1, n_leaves):
        dist_matrix[kj, ki] = dist_matrix[ki, kj]

dist_matrix = dist_matrix / (2 * data.shape[1])
dist_matrix = (-3 / 4) * np.log(1 - dist_matrix * 4 / 3)


dm = DistanceMatrix(dist_matrix)
NJ_tree = nj(dm, result_constructor=str)
NJ_nx = newick2nx(NJ_tree, n_leaves)
NJ_toc = time.time()
pickle.dump(NJ_nx, open('results/'+ args.dataset +'_NJ_tree.pickle', 'wb'))
print('Total time elapsed NJ=')
print(NJ_toc-NJ_tic)
