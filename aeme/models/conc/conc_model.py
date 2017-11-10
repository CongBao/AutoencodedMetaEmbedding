#!/usr/bin/env python
"""
concatenation model
"""

from __future__ import division

import json

import numpy as np

from aeme.models.core.model import Model
from aeme.utils import io

__author__ = 'Cong Bao'

class ConcModel(Model):

    def __init__(self,
                 input_path,
                 output_path,
                 log_path,
                 graph_path,
                 learning_rate,
                 batch_size,
                 epoch,
                 noise_type,
                 noise_ratio):
        Model.__init__(self, self.__class__.__name__, log_path)
        self.input_path = json.loads(input_path)
        self.output_path = output_path
        self.log_path = log_path
        self.graph_path = graph_path + self.name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio

    def _generate_meta_embedding(self):
        meta_embedding = {}
        self.logger.log('Generating meta embeddings...')
        for word in self.inter_words:
            meta_embedding[word] = np.concatenate([self.source_dict['cbow'][word], self.source_dict['glove'][word]])
        self.logger.log('Saving data into output file: %s' % self.output_path)
        io.save_embeddings(meta_embedding, self.output_path)

    def build_model(self):
        pass

    def run(self):
        self._load_data()
        self._generate_meta_embedding()
        self.logger.log('Complete.')
