import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
import numpy as np
import torch
import networkx as nx
from sklearn.cluster import KMeans
from ssl_utils import *
from distance import *
import os
import os.path as osp
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

class Base:

    def __init__(self, adj, features, device):
        self.adj = adj
        self.features = features.to(device)
        self.device = device
        self.cached_adj_norm = None

    def get_adj_norm(self):
        if self.cached_adj_norm is None:
            adj_norm = preprocess_adj(self.adj, self.device)
            self.cached_adj_norm= adj_norm
        return self.cached_adj_norm

    def make_loss(self, embeddings):
        return 0

    def transform_data(self):
        return self.get_adj_norm(), self.features


class PairwiseAttrSim(Base):

    def __init__(self, adj, features, nhid, device, idx_train, args, regression=True):
        args.idx_train = idx_train

        self.adj = adj

        self.args = args
        self.features = features.to(device)
        self.nfeat = features.shape[1]
        self.cached_adj_norm = None
        self.device = device

        self.labeled = idx_train.cpu().numpy()
        self.all = np.arange(adj.shape[0])
        self.unlabeled = np.array([n for n in self.all if n not in idx_train])

        self.regression = regression
        self.nclass = 1
        if regression:
            self.linear = nn.Linear(nhid, self.nclass).to(device)
        else:
            self.linear = nn.Linear(nhid, 2).to(device)

        self.pseudo_labels = None
        self.sims = None

    def transform_data(self):
        return self.get_adj_norm(), self.features

    def make_loss(self, embeddings):
        if self.regression:
            return self.regression_loss(embeddings)
        else:
            return self.classification_loss(embeddings)

    def regression_loss(self, embeddings):
        if self.pseudo_labels is None:
            agent = AttrSim(self.adj ,self.features, args=self.args)
            self.pseudo_labels = agent.get_label().to(self.device)
            node_pairs = agent.node_pairs
            self.node_pairs = node_pairs

        k = 10000
        node_pairs = self.node_pairs
        if len(self.node_pairs[0]) > k:
            sampled = np.random.choice(len(self.node_pairs[0]), k, replace=False)

            embeddings0 = embeddings[node_pairs[0][sampled]]
            embeddings1 = embeddings[node_pairs[1][sampled]]
            embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
            loss = F.mse_loss(embeddings, self.pseudo_labels[sampled], reduction='mean')
        else:
            embeddings0 = embeddings[node_pairs[0]]
            embeddings1 = embeddings[node_pairs[1]]
            embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
            loss = F.mse_loss(embeddings, self.pseudo_labels, reduction='mean')

        # print(loss)
        return loss

    def classification_loss(self, embeddings):
        if self.pseudo_labels is None:
            agent = AttrSim(self.adj ,self.features, self.labels, self.args)
            pseudo_labels = agent.get_class()
            self.pseudo_labels = torch.LongTensor(pseudo_labels).to(self.device)
            self.node_pairs = agent.node_pairs

        node_pairs = self.node_pairs
        embeddings0 = embeddings[node_pairs[0]]
        embeddings1 = embeddings[node_pairs[1]]
        embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
        output = F.log_softmax(embeddings, dim=1)

        loss = F.nll_loss(output, self.pseudo_labels)
        print(loss)
        from metric import accuracy
        acc = accuracy(output, self.pseudo_labels)
        print(acc)
        return loss

    def sample(self, labels, ratio=0.1, k=2000):
        node_pairs = []
        for i in range(1, labels.max()+1):
            tmp = np.array(np.where(labels==i)).transpose()
            indices = np.random.choice(np.arange(len(tmp)), k, replace=False)
            node_pairs.append(tmp[indices])
        node_pairs = np.array(node_pairs).reshape(-1, 2).transpose()
        return node_pairs[0], node_pairs[1]


class MergedKNNGraph(Base):

    def __init__(self, adj, features, nhid, device, idx_train, args):
        self.adj = adj
        self.args = args

        self.features = features.to(device)
        self.nfeat = features.shape[1]
        self.cached_adj_norm = None
        self.device = device

        self.labeled = idx_train.cpu().numpy()
        self.all = np.arange(adj.shape[0])
        self.unlabeled = np.array([n for n in self.all if n not in idx_train])
        self.nclass = 1
        self.pseudo_labels = None

        degree = self.adj.sum(0).A1
        k = args.k

        if not osp.exists('saved/'):
           os.mkdir('saved')
        if not os.path.exists(f'saved/{args.dataset}_sims_{k}.npz'):
            from sklearn.metrics.pairwise import cosine_similarity
            features = np.copy(features)
            features[features!=0] = 1
            sims = cosine_similarity(features)
            sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
            for i in range(len(sims)):
                indices_argsort = np.argsort(sims[i])
                sims[i, indices_argsort[: -k]] = 0

            self.A_feat = sp.csr_matrix(sims)
            sp.save_npz(f'saved/{args.dataset}_sims_{k}.npz', self.A_feat)
        else:
            print(f'loading saved/{args.dataset}_sims_{k}.npz')
            self.A_feat = sp.load_npz(f'saved/{args.dataset}_sims_{k}.npz')


    def transform_data(self, lambda_=None):
        if self.cached_adj_norm is None:
            if lambda_ is None:
                r_adj = self.adj + self.args.lambda_ * self.A_feat
            else:
                r_adj = self.adj + lambda_ * self.A_feat
            r_adj = preprocess_adj(r_adj, self.device)
            self.cached_adj_norm = r_adj
        return self.cached_adj_norm, self.features

    def make_loss(self, embeddings):
        return 0


class KNNGraphPrediction(Base):

    def __init__(self, adj, features, nhid, device, args):
        self.adj = adj
        self.device = device
        self.features = features.to(device)
        from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

        k = args.k
        if not osp.exists(f'{args.dataset}_knn_{k}.npz'):
            self.adj_knn = kneighbors_graph(features, k, mode='connectivity', include_self=False).tocoo()
            sp.save_npz(f'{args.dataset}_knn_{k}.npz', self.adj_knn)
        else:
            self.adj_knn = sp.load_npz(f'{args.dataset}_knn_{k}.npz')

        self.cached_adj_norm = None
        self.pseudo_labels = None
        self.linear = nn.Linear(2*nhid, 2).to(device)

    def transform_data(self):
        sample_ratio = 0.1
        self.pseudo_labels = None
        if self.pseudo_labels is None:
            edges = self.pos_sample(sample_ratio)
            self.pos_edges = edges
            self.neg_edges = self.neg_sample(k=len(edges[0]))
            self.pseudo_labels = np.zeros(2*len(edges[0]))
            self.pseudo_labels[: len(edges[0])] = 1
            self.pseudo_labels = torch.LongTensor(self.pseudo_labels).to(self.device)
            # self.neg_edges = self.neg_sample(k=len(edges[0]))
        return self.get_adj_norm(), self.features

    def make_loss(self, embeddings):
        '''link prediction loss'''
        pos_edges = self.pos_edges
        neg_edges = self.neg_edges
        node_pairs = np.hstack((np.array(pos_edges), np.array(neg_edges).transpose()))
        self.node_pairs = node_pairs
        embeddings0 = embeddings[node_pairs[0]]
        embeddings1 = embeddings[node_pairs[1]]

        embeddings = self.linear(torch.cat([embeddings0, embeddings1], dim=1))
        output = F.log_softmax(embeddings, dim=1)

        loss = F.nll_loss(output, self.pseudo_labels)
        # print(loss)
        return loss

    def pos_sample(self, sample_ratio):
        nnz = self.adj_knn.nnz
        perm = np.random.permutation(nnz)
        nnz_sampled = int(nnz*(sample_ratio))
        sampled = perm[: nnz_sampled]
        edges = (self.adj_knn.row[sampled], self.adj_knn.col[sampled])
        return edges

    def neg_sample(self, k):
        nonzero = set(zip(*self.adj.nonzero()))
        edges = self.random_sample_edges(self.adj, k, exclude=nonzero)
        return edges

    def random_sample_edges(self, adj, n, exclude):
        '''
            'exclude' is a set which contains the edges we do not want to sample
             and the edges already sampled
        '''
        itr = self.sample_forever(adj, exclude=exclude)
        return [next(itr) for _ in range(n)]

    def sample_forever(self, adj, exclude):
        '''
            'exclude' is a set which contains the edges we do not want to sample
             and the edges already sampled
        '''
        while True:
            # t = tuple(np.random.randint(0, adj.shape[0], 2))
            t = tuple(random.sample(range(0, adj.shape[0]), 2))
            if t not in exclude:
                yield t
                exclude.add(t)
                exclude.add((t[1], t[0]))


class KNNPlusPairAttr(Base):

    def __init__(self, adj, features, labels, nhid, device, idx_train, args, regression=True):
        self.agent1 = MergedKNNGraph(adj, features, idx_train=idx_train, nhid=args.hidden, args=args, device='cuda')
        self.agent2 = PairwiseAttrSim(adj, features, idx_train=idx_train, nhid=args.hidden, args=args, device='cuda')
        self.args = args

    def transform_data(self):
        # TODO incorporate feature information
        return self.agent1.transform_data(self.args.k_lambda_)

    def make_loss(self, embeddings):
        return self.agent2.make_loss(embeddings)



class OnlyKNN(Base):

    def __init__(self, adj, features, nhid, device, idx_train, args):
        self.adj = adj
        self.args = args

        self.features = features.to(device)
        self.nfeat = features.shape[1]
        self.cached_adj_norm = None
        self.device = device

        self.labeled = idx_train.cpu().numpy()
        self.all = np.arange(adj.shape[0])
        self.unlabeled = np.array([n for n in self.all if n not in idx_train])
        self.nclass = 1
        self.pseudo_labels = None

        degree = self.adj.sum(0).A1
        k = args.k
        if not os.path.exists(f'{args.dataset}_sims_{k}.npz'):
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.metrics import pairwise_distances
            from scipy.spatial.distance import jaccard
            # metric = jaccard
            metric = "cosine"
            # sims = pairwise_distances(self.features, self.features, metric=metric)
            features[features!=0] = 1

            sims = cosine_similarity(features)
            sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
            for i in range(len(sims)):
                indices_argsort = np.argsort(sims[i])
                sims[i, indices_argsort[: -k]] = 0

            self.A_feat = sp.csr_matrix(sims)
            sp.save_npz(f'{args.dataset}_sims_{k}.npz', self.A_feat)
        else:

            print(f'loading {args.dataset}_sims_{k}.npz')
            self.A_feat = sp.load_npz(f'{args.dataset}_sims_{k}.npz')


    def transform_data(self, lambda_=None):


        if self.cached_adj_norm is None:
            r_adj = self.A_feat
            r_adj = preprocess_adj(r_adj, self.device)
            self.cached_adj_norm = r_adj

            self.features[self.features!=0] = 1

        return self.cached_adj_norm, self.features

    def make_loss(self, embeddings):
        return 0

def preprocess_features(features, device):
    return features.to(device)

def preprocess_adj(adj, device):
    # adj_normalizer = fetch_normalization(normalization)
    adj_normalizer = aug_normalized_adjacency
    r_adj = adj_normalizer(adj)
    r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()
    r_adj = r_adj.to(device)
    return r_adj

def preprocess_adj_noloop(adj, device):
    # adj_normalizer = fetch_normalization(normalization)
    adj_normalizer = noaug_normalized_adjacency
    r_adj = adj_normalizer(adj)
    r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()
    r_adj = r_adj.to(device)
    return r_adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def noaug_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

