#!/usr/bin/env python
"""
Preprocessing data
"""

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

def normalize_emb(emb_dict, word_list=None):
    """
    normalize the embeddings
    """
    words = list(emb_dict.keys()) if word_list is None else word_list
    norm = skpre.normalize([emb_dict[word] for word in words])
    res = {}
    for i, vec in enumerate(norm):
        res[words[i]] = vec
    return res

def tsvd(embedding_dict, dim=300, niter=10):
    """
    perform truncated SVD on embedding set
    """
    words = list(embedding_dict.keys())
    svd = TruncatedSVD(n_components=dim, n_iter=niter)
    reduced = svd.fit_transform([embedding_dict[word] for word in words])
    res = {}
    for i, vec in enumerate(reduced):
        res[words[i]] = vec
    return res
