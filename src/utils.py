"""
Some useful functions
"""

import csv

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_embeddings(file_path, header=None, index=0):
    """
    Load embeddings from file.
    """
    embeddings = {}
    data = pd.read_table(file_path, sep=' ', header=header, index_col=index, quoting=csv.QUOTE_NONE)
    for i in tqdm(range(0, len(data.index))):
        embeddings[data.index[i]] = np.asarray(data.values[i], dtype='float32')
    return embeddings

def load_intersection_words(file_path):
    """
    load intersection words from file
    """
    return pd.read_table(file_path).values.T[0]