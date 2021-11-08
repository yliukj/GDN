from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp
import pickle as pkl
#import cPickle as pkl
import os
import h5py
import pandas as pd


from data_utils import load_data, map_data, download_dataset


def normalize_features(feat):

    degree = np.asarray(feat.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf

    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    if feat_norm.nnz == 0:
        print('ERROR: normalized adjacency matrix has only zero entries!!!!!')
        exit

    return feat_norm


def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir = np.asarray(ds['ir'])
            jc = np.asarray(ds['jc'])
            out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T

    db.close()

    return out


def preprocess_user_item_features(u_features, v_features):
    """
    Creates one big feature matrix out of user features and item features.
    Stacks item features under the user features.
    """

    zero_csr_u = sp.csr_matrix((u_features.shape[0], v_features.shape[1]), dtype=u_features.dtype)
    zero_csr_v = sp.csr_matrix((v_features.shape[0], u_features.shape[1]), dtype=v_features.dtype)

    u_features = sp.hstack([u_features, zero_csr_u], format='csr')
    v_features = sp.hstack([zero_csr_v, v_features], format='csr')

    return u_features, v_features


def globally_normalize_bipartite_adjacency(adjacencies, verbose=False, symmetric=True):
    """ Globally Normalizes set of bipartite adjacency matrices """

    if verbose:
        print('Symmetrically normalizing bipartite adj')
    # degree_u and degree_v are row and column sums of adj+I

    adj_tot = np.sum(adj for adj in adjacencies)
    degree_u = np.asarray(adj_tot.sum(1)).flatten()
    degree_v = np.asarray(adj_tot.sum(0)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree_u[degree_u == 0.] = np.inf
    degree_v[degree_v == 0.] = np.inf

    degree_u_inv_sqrt = 1. / np.sqrt(degree_u)
    degree_v_inv_sqrt = 1. / np.sqrt(degree_v)
    degree_u_inv_sqrt_mat = sp.diags([degree_u_inv_sqrt], [0])
    degree_v_inv_sqrt_mat = sp.diags([degree_v_inv_sqrt], [0])

    degree_u_inv = degree_u_inv_sqrt_mat.dot(degree_u_inv_sqrt_mat)

    if symmetric:
        adj_norm = [degree_u_inv_sqrt_mat.dot(adj).dot(degree_v_inv_sqrt_mat) for adj in adjacencies]

    else:
        adj_norm = [degree_u_inv.dot(adj) for adj in adjacencies]

    return adj_norm


def sparse_to_tuple(sparse_mx):
    """ change of format for sparse matrix. This format is used
    for the feed_dict where sparse matrices need to be linked to placeholders
    representing sparse matrices. """

    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def load_data_monti(dataset, testing=False):
    """
    Loads data from Monti et al. paper.
    """

    path_dataset = 'data/' + dataset + '/training_test_dataset.mat'

    M = load_matlab_file(path_dataset, 'M')
    Otraining = load_matlab_file(path_dataset, 'Otraining')
    Otest = load_matlab_file(path_dataset, 'Otest')

    num_users = M.shape[0]
    num_items = M.shape[1]

    if dataset == 'flixster':
        Wrow = load_matlab_file(path_dataset, 'W_users')
        Wcol = load_matlab_file(path_dataset, 'W_movies')
        u_features = Wrow
        v_features = Wcol
        # print(num_items, v_features.shape)
        # v_features = np.eye(num_items)

    elif dataset == 'douban':
        Wrow = load_matlab_file(path_dataset, 'W_users')
        u_features = Wrow
        v_features = np.eye(num_items)
    elif dataset == 'yahoo_music':
        Wcol = load_matlab_file(path_dataset, 'W_tracks')
        u_features = np.eye(num_users)
        v_features = Wcol

    u_nodes_ratings = np.where(M)[0]
    v_nodes_ratings = np.where(M)[1]
    ratings = M[np.where(M)]

    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
    ratings = ratings.astype(np.float64)

    u_nodes = u_nodes_ratings
    v_nodes = v_nodes_ratings

    print('number of users = ', len(set(u_nodes)))
    print('number of item = ', len(set(v_nodes)))

    neutral_rating = -1  # int(np.ceil(np.float(num_classes)/2.)) - 1

    # assumes that ratings_train contains at least one example of every rating type
    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])

    for i in range(len(u_nodes)):
        assert(labels[u_nodes[i], v_nodes[i]] == rating_dict[ratings[i]])

    labels = labels.reshape([-1])

    # number of test and validation edges

    num_train = np.where(Otraining)[0].shape[0]
    num_test = np.where(Otest)[0].shape[0]
    num_val = int(np.ceil(num_train * 0.2))
    num_train = num_train - num_val

    pairs_nonzero_train = np.array([[u, v] for u, v in zip(np.where(Otraining)[0], np.where(Otraining)[1])])
    idx_nonzero_train = np.array([u * num_items + v for u, v in pairs_nonzero_train])

    pairs_nonzero_test = np.array([[u, v] for u, v in zip(np.where(Otest)[0], np.where(Otest)[1])])
    idx_nonzero_test = np.array([u * num_items + v for u, v in pairs_nonzero_test])

    # Internally shuffle training set (before splitting off validation set)
    rand_idx = range(len(idx_nonzero_train))
    np.random.seed(42)
    np.random.shuffle(list(rand_idx))
    #np.random.shuffle(rand_idx)
    idx_nonzero_train = idx_nonzero_train[rand_idx]
    pairs_nonzero_train = pairs_nonzero_train[rand_idx]

    idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
    pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test], axis=0)
    print(pairs_nonzero.shape)
    val_idx = idx_nonzero[0:num_val]
    train_idx = idx_nonzero[num_val:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]

    assert(len(test_idx) == num_test)

    val_pairs_idx = pairs_nonzero[0:num_val]
    train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

    class_values = np.sort(np.unique(ratings))

    if u_features is not None:
        u_features = sp.csr_matrix(u_features)
        print("User features shape: " + str(u_features.shape))

    if v_features is not None:
        v_features = sp.csr_matrix(v_features)
        print("Item features shape: " + str(v_features.shape))

    return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
        val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values
