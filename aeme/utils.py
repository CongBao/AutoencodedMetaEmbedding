# utils

from __future__ import print_function

import csv

import numpy as np
import pandas as pd
from tqdm import tqdm

__author__ = 'Cong Bao'

class Utils(object):

    def __init__(self, log=None):
        self.log = log if log else print

    def load_emb(self, file_path):
        embed = {}
        self.log('loading %s' % file_path)
        data = pd.read_table(file_path, sep=' ', header=None, index_col=0, quoting=csv.QUOTE_NONE)
        for i in tqdm(range(len(data.index)), bar_format='Progress: {percentage:3.0f}% {r_bar}'):
            embed[str(data.index[i])] = np.asarray(data.values[i], dtype='float32')
        return embed

    def save_emb(self, emb_dict, file_path):
        self.log('Saving embedding to %s' % file_path)
        data = pd.DataFrame.from_dict(emb_dict, orient='index')
        data.to_csv(file_path, sep=' ', header=False, encoding='utf-8')
