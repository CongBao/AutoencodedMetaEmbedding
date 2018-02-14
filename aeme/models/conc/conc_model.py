#!/usr/bin/env python
"""
concatenation model
"""

from __future__ import division

import numpy as np

from aeme.models.core.model import Model
from aeme.utils import data_process, embed_io

__author__ = 'Cong Bao'

class ConcModel(Model):
    """A simple model using concatenation"""

    def __init__(self, log_path):
        Model.__init__(self, self.__class__.__name__, log_path)

    def _generate_meta_embedding(self):
        meta_embedding = {}
        self.logger.log('Generating meta embeddings...')
        for word in self.inter_words:
            meta_embedding[word] = np.concatenate([self.source_dict['cbow'][word], self.source_dict['glove'][word]])
        meta_embedding = data_process.normalize_emb(meta_embedding)
        self.logger.log('Saving data into output file: %s' % self.output_path)
        embed_io.save_embeddings(meta_embedding, self.output_path)

    def build_model(self):
        pass

    def run(self, params):
        self._configure(params)
        self._load_data()
        self._generate_meta_embedding()
        self.logger.log('Complete.')

class AvgModel(Model):

    def __init__(self, log_path):
        Model.__init__(self, self.__class__.__name__, log_path)

    def _generate_meta_embedding(self):
        meta_embedding = {}
        self.logger.log('Generating meta embeddings...')
        for word in self.inter_words:
            meta_embedding[word] = np.add(self.source_dict['cbow'][word], self.source_dict['glove'][word])
        meta_embedding = data_process.normalize_emb(meta_embedding)
        self.logger.log('Saving data into output file: %s' % self.output_path)
        embed_io.save_embeddings(meta_embedding, self.output_path)

    def build_model(self):
        pass

    def run(self, params):
        self._configure(params)
        self._load_data()
        self._generate_meta_embedding()
        self.logger.log('Complete.')
