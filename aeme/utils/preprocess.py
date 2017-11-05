#!/usr/bin/env python
"""
Preprocessing data
"""

import sklearn.preprocessing as skpre

__author__ = 'Cong Bao'

def normalize_embeddings(embedding_dict, scale_factor):
    """
    normalize the embeddings
    """
    for word, values in embedding_dict.items():
        embedding_dict[word] = scale_factor * skpre.normalize(values.reshape(1, -1))[0]
    return embedding_dict
