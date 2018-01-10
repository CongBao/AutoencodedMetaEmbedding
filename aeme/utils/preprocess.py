#!/usr/bin/env python
"""
Preprocessing data
"""

import sklearn.preprocessing as skpre

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
