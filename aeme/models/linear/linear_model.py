#!/usr/bin/env python
"""
linear model
"""

from __future__ import division

import json
import os

from aeme.models.core.model import Model

__author__ = 'Cong Bao'

class LinearModel(Model):

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
        self.graph_path = graph_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio

        # init source data
        self._load_data()
        self._init_source()
        self._build_model()

    def _build_model(self):
        self.encoder['cbow'] = self._add_layer(self.input['cbow'], (300, 300), None, 'cbow_encoder')
        self.decoder['cbow'] = self._add_layer(self.encoder['cbow'], (300, 300), None, 'cbow_decoder')
        self.encoder['glove'] = self._add_layer(self.input['glove'], (300, 300), None, 'glove_encoder')
        self.decoder['glove'] = self._add_layer(self.encoder['glove'], (300, 300), None, 'glove_decoder')
