"""
Some useful functions
"""

import csv

import numpy as np
import pandas as pd
import sklearn.preprocessing as skpre
from tqdm import tqdm


def load_embeddings(file_path, header=None, index=0):
    """
    Load embeddings from file.
    """
    embeddings = {}
    data = pd.read_table(file_path, sep=' ', header=header, index_col=index, quoting=csv.QUOTE_NONE)
    fmt = 'Loading embeddings: {percentage:3.0f}% {r_bar}'
    for i in tqdm(range(len(data.index)), bar_format=fmt):
        embeddings[data.index[i]] = np.asarray(data.values[i], dtype='float32')
    return embeddings

def save_embeddings(embedding_dict, file_path):
    """
    write embeddings to file
    """
    data = pd.DataFrame.from_dict(embedding_dict, orient='index')
    data.to_csv(file_path, sep=" ", header=False, encoding='utf-8')

def normalize_embeddings(embedding_dict, scale_factor):
    """
    normalize the embeddings
    """
    for word, values in embedding_dict.items():
        embedding_dict[word] = scale_factor * skpre.normalize(values.reshape(1,-1))[0]
    return embedding_dict

def normalize_emb(emb_dict):
    """
    normalize the embeddings
    """
    words = list(emb_dict.keys())
    norm = skpre.normalize([emb_dict[word] for word in words])
    res = {}
    for i, vec in enumerate(norm):
        res[words[i]] = vec
    return res
