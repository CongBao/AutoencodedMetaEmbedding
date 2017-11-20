#!/usr/bin/env python
"""
linear model
"""

from __future__ import division

import tensorflow as tf

from aeme.models.core.model import Model

__author__ = 'Cong Bao'

class AESigmoidModel(Model):

    def __init__(self, log_path):
        Model.__init__(self, self.__class__.__name__, log_path)

    def build_model(self):
        self.encoder['cbow'] = self.add_layer(self.input['cbow'], (300, 300), lambda x:tf.nn.sigmoid(x) - 0.5, 'cbow_encoder')
        self.decoder['cbow'] = self.add_layer(self.encoder['cbow'], (300, 300), None, 'cbow_decoder')
        self.encoder['glove'] = self.add_layer(self.input['glove'], (300, 300), lambda x:tf.nn.sigmoid(x) - 0.5, 'glove_encoder')
        self.decoder['glove'] = self.add_layer(self.encoder['glove'], (300, 300), None, 'glove_decoder')

class AETanHModel(Model):

    def __init__(self, log_path):
        Model.__init__(self, self.__class__.__name__, log_path)

    def build_model(self):
        self.encoder['cbow'] = self.add_layer(self.input['cbow'], (300, 300), tf.nn.tanh, 'cbow_encoder')
        self.decoder['cbow'] = self.add_layer(self.encoder['cbow'], (300, 300), None, 'cbow_decoder')
        self.encoder['glove'] = self.add_layer(self.input['glove'], (300, 300), tf.nn.tanh, 'glove_encoder')
        self.decoder['glove'] = self.add_layer(self.encoder['glove'], (300, 300), None, 'glove_decoder')

class AEReluModel(Model):

    def __init__(self, log_path):
        Model.__init__(self, self.__class__.__name__, log_path)

    def build_model(self):
        self.encoder['cbow'] = self.add_layer(self.input['cbow'], (300, 300), tf.nn.relu, 'cbow_encoder')
        self.decoder['cbow'] = self.add_layer(self.encoder['cbow'], (300, 300), None, 'cbow_decoder')
        self.encoder['glove'] = self.add_layer(self.input['glove'], (300, 300), tf.nn.relu, 'glove_encoder')
        self.decoder['glove'] = self.add_layer(self.encoder['glove'], (300, 300), None, 'glove_decoder')
