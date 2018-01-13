#!/usr/bin/env python
"""
Preprocessing data
"""

import numpy as np
import sklearn.preprocessing as skpre
from sklearn.decomposition import TruncatedSVD

__author__ = 'Cong Bao'

def normalize(vector, scale_factor):
    """
    normalize a vector
    """
    return scale_factor * skpre.normalize(vector.reshape(1, -1))[0]

def normalize_embeddings(embedding_dict, scale_factor):
    """
    normalize the embeddings
    """
    for word, values in embedding_dict.items():
        embedding_dict[word] = normalize(values, scale_factor)
    return embedding_dict

def tsvd(embedding_dict, dim=300, niter=10):
    """
    perform truncated SVD on embedding set
    """
    words = list(embedding_dict.keys())
    arr = np.asarray([embedding_dict[word] for word in words])
    svd = TruncatedSVD(n_components=dim, n_iter=niter)
    reduced = svd.fit_transform(arr)
    res = {}
    for i, vec in enumerate(reduced):
        res[words[i]] = vec
    return res
