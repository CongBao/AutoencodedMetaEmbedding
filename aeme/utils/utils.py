# Utilities used for I/O
# File: utils.py
# Author: Cong Bao

from __future__ import print_function

import csv

import numpy as np
import pandas as pd
from tqdm import tqdm

__author__ = 'Cong Bao'

class Utils(object):
    """ Utilities for I/O.
        :param log: a log function, if None print will be used
    """

    def __init__(self, log=None):
        self.log = log if log else print

    def load_words(self, file_path):
        """ Load word list from file.
            :param file_path: path of word list file
            :return: a list of words
        """
        self.log('Loading %s' % file_path)
        with open(file_path) as f:
            word_list = f.readlines()
        return [x.strip() for x in word_list]

    def save_words(self, iterable, file_path):
        """ Save a list of words to file.
            :param iterable: a list of words
            :param file_path: path of target file
        """
        self.log('Saving to %s' % file_path)
        with open(file_path, 'w') as f:
            for item in iterable:
                f.write('%s\n' % str(item))

    def load_emb(self, file_path):
        """ Load word embeddings from file.
            :param file_path: path of word embedding file
            :return: a dict {word:embedding}
        """
        embed = {}
        self.log('Loading %s' % file_path)
        data = pd.read_table(file_path, sep=' ', header=None, index_col=0, quoting=csv.QUOTE_NONE)
        for i in tqdm(range(len(data.index)), bar_format='Progress: {percentage:3.0f}% {r_bar}'):
            embed[str(data.index[i])] = np.asarray(data.values[i], dtype='float32')
        return embed

    def save_emb(self, emb_dict, file_path):
        """ Save word embedding dict to file.
            :param emb_dict: a dict {word:embedding}
            :param file_path: path of output file
        """
        self.log('Saving embedding to %s' % file_path)
        data = pd.DataFrame.from_dict(emb_dict, orient='index')
        data.to_csv(file_path, sep=' ', header=False, encoding='utf-8')
