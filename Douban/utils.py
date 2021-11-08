import numpy as np
import scipy.io
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from itertools import chain
from collections import defaultdict


def dropout_edge(adj, keep_prob,seed):
    np.random.seed(seed)
    noise_shape = adj.nnz
    random_tensor = keep_prob
    random_tensor += np.random.uniform(size=noise_shape)
    mask_tensor = sp.csr_matrix((np.floor(random_tensor), adj.indices, adj.indptr), shape=adj.shape,dtype=int)
    retain_edge = adj.multiply(mask_tensor)
    return retain_edge 

def adversarial_noise(u_indices,v_indices,labels,probability = 1):
  size = int(len(labels)*probability)
  np.random.seed(42)
  selected_u_idx = np.random.choice(u_indices.max()-1,size = size)
  selected_v_idx = np.random.choice(v_indices.max()-1,size = size)
  selected_labels = np.random.choice(labels.max(),size = size)
  train_u_indices = np.append(u_indices,selected_u_idx)
  train_v_indices = np.append(v_indices,selected_v_idx)
  train_labels = np.append(labels,selected_labels)
  return train_u_indices,train_v_indices,train_labels

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    #return features
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rows, cols = adj.nonzero()
    adj[cols, rows] = adj[rows, cols]
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).tocoo()


def preprocess_gcn_adj(adj):
    adj_normalized = normalize_adj(adj + 5 * sp.eye(adj.shape[0]))
    sym_l = sp.eye(adj_normalized.shape[0]) - adj_normalized
    true_a = adj_normalized
    return sparse_to_tuple(true_a)

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    sym_l = sp.eye(adj_normalized.shape[0]) - adj_normalized
    true_a = 0.1*sp.eye(adj_normalized.shape[0]) + adj_normalized + 0.5 * sym_l.dot(sym_l) - 1.0/6 * sym_l.dot(sym_l).dot(sym_l) + 1.0/24 * sym_l.dot(sym_l).dot(sym_l).dot(sym_l)
    return sparse_to_tuple(true_a)


def preprocess_gcn_inverse_adj(adj):
    adj_normalized = normalize_adj(adj + 5 * sp.eye(adj.shape[0]))
    sym_l = sp.eye(adj_normalized.shape[0]) - adj_normalized
    true_a = sp.eye(adj_normalized.shape[0]) +sym_l 
    return sparse_to_tuple(true_a)

def preprocess_inverse_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    sym_l = sp.eye(adj_normalized.shape[0]) - adj_normalized
    true_a = 1.1*sp.eye(adj_normalized.shape[0]) +sym_l + 0.5 * sym_l.dot(sym_l) + 1.0/6 * sym_l.dot(sym_l).dot(sym_l) + 1.0/24 * sym_l.dot(sym_l).dot(sym_l).dot(sym_l)
    return sparse_to_tuple(true_a)


def construct_feed_dict(gcn,gcn_inverse,features, support,inverse_support, num_nodes,labels,u_indices,v_indices,placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['gcn_support']: gcn})
    feed_dict.update({placeholders['gcn_inverse_support']: gcn_inverse})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['inverse_support']: inverse_support})
    feed_dict.update({placeholders['num_nodes']: num_nodes})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['u_indices']: u_indices})
    feed_dict.update({placeholders['v_indices']: v_indices})
    return feed_dict
