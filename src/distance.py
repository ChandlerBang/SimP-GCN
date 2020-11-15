import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
import collections
import sklearn
from ssl_utils import encode_onehot
from utils import row_normalize, to_scipy
import numpy as np
import os
import scipy.sparse as sp
from tqdm import tqdm
from itertools import product

class AttrSim:

    def __init__(self, adj, features, labels=None, args=None, nclass=4):
        if args.dataset == 'nell':
            self.features = to_scipy(features)
        else:
            self.features = features.cpu().numpy()
        self.features[self.features!=0] = 1

        self.labels = labels
        self.adj = adj
        self.nclass = nclass
        self.args = args

    def get_label(self):
        args = self.args
        if not os.path.exists(f'saved/{args.dataset}_cosine_sims.npy'):
            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity(self.features)
            np.save(f'saved/{args.dataset}_cosine_sims.npy', sims)
        else:
            sims = np.load(f'saved/{args.dataset}_cosine_sims.npy')

        k = 5

        if not os.path.exists(f'saved/{args.dataset}_{k}_attrsim_sampled_idx.npy'):
            try:
                indices_sorted = sims.argsort(1)
                idx = np.arange(k, sims.shape[0]-k)
                selected = np.hstack((indices_sorted[:, :k],
                    indices_sorted[:, -k-1:]))

                selected_set = set()
                for i in range(len(sims)):
                    for pair in product([i], selected[i]):
                        if pair[0] > pair[1]:
                            pair = (pair[1], pair[0])
                        if  pair[0] == pair[1]:
                            continue
                        selected_set.add(pair)

            except MemoryError:
                selected_set = set()
                for ii, row in tqdm(enumerate(sims)):
                    row = row.argsort()
                    idx = np.arange(k, sims.shape[0]-k)
                    sampled = np.random.choice(idx, k, replace=False)
                    for node in np.hstack((row[:k], row[-k-1:], row[sampled])):
                        if ii > node:
                            pair = (node, ii)
                        else:
                            pair = (ii, node)
                        selected_set.add(pair)

            sampled = np.array(list(selected_set)).transpose()
            print(f'loading saved/{args.dataset}_{k}_attrsim_sampled_idx.npy')
            np.save(f'saved/{args.dataset}_{k}_attrsim_sampled_idx.npy', sampled)
        else:
            print(f'loading saved/{args.dataset}_{k}_attrsim_sampled_idx.npy')
            sampled = np.load(f'saved/{args.dataset}_{k}_attrsim_sampled_idx.npy')
        print('number of sampled:', len(sampled[0]))
        self.node_pairs = (sampled[0], sampled[1])

        self.sims = sims
        return torch.FloatTensor(sims[self.node_pairs]).reshape(-1,1)

    def get_class(self):
        args = self.args
        if not os.path.exists(f'saved/{args.dataset}_cosine_sims.npy'):
            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity(self.features)
            np.save(f'saved/{args.dataset}_cosine_sims.npy', sims)
        else:
            sims = np.load(f'saved/{args.dataset}_cosine_sims.npy')

        k = 5
        # if not os.path.exists(f'{args.dataset}_{k}_attrsim_sampled_idx.npy'):
        if not os.path.exists(f'saved/{args.dataset}_{k}_attrsim_pseudo_label.npy'):
            indices_sorted = sims.argsort(1)
            idx = np.arange(k, sims.shape[0]-k)
            selected = np.hstack((indices_sorted[:, :k],
                indices_sorted[:, -k-1:]))


            for ii in range(len(sims)):
                sims[ii, indices_sorted[ii, :k]] = 0
                sims[ii, indices_sorted[ii, -k-1:]] = 1


            from itertools import product
            selected_set = set()
            for i in range(len(sims)):
                for pair in product([i], selected[i]):
                    if pair[0] > pair[1]:
                        pair = (pair[1], pair[0])
                    if  pair[0] == pair[1]:
                        continue
                    selected_set.add(pair)

            sampled = np.array(list(selected_set)).transpose()
            node_pairs = (sampled[0], sampled[1])
            pseudo_labels = sims[node_pairs].reshape(-1)
            np.save(f'saved/{args.dataset}_{k}_attrsim_pseudo_label.npy', pseudo_labels)
            np.save(f'saved/{args.dataset}_{k}_attrsim_sampled_idx.npy', sampled)
        else:
            print(f'loading saved/{args.dataset}_{k}_attrsim_sampled_idx.npy')
            sampled = np.load(f'saved/{args.dataset}_{k}_attrsim_sampled_idx.npy')
            pseudo_labels = np.load(f'saved/{args.dataset}_{k}_attrsim_pseudo_label.npy')
        print('number of sampled:', len(sampled[0]))
        self.node_pairs = (sampled[0], sampled[1])
        return pseudo_labels
