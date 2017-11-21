#!/usr/bin/env python
"""
linear model
"""

from __future__ import division

import tensorflow as tf

from aeme.models.core.model import Model

__author__ = 'Cong Bao'

class LinearModel(Model):
    """A regular linear model"""

    def __init__(self, log_path):
        Model.__init__(self, self.__class__.__name__, log_path)

    def build_model(self):
        self.encoder['cbow'] = self.add_layer(self.input['cbow'], (300, 300), None, 'cbow_encoder')
        self.decoder['cbow'] = self.add_layer(self.encoder['cbow'], (300, 300), None, 'cbow_decoder')
        self.encoder['glove'] = self.add_layer(self.input['glove'], (300, 300), None, 'glove_encoder')
        self.decoder['glove'] = self.add_layer(self.encoder['glove'], (300, 300), None, 'glove_decoder')

class TiedLinearModel(Model):
    """A linear model with tied weight: W_D = W_E.T"""
    
    def __init__(self, log_path):
        Model.__init__(self, self.__class__.__name__, log_path)

    def build_model(self):
        with tf.name_scope('cbow_encoder'):
            w_cbow_en = tf.Variable(tf.random_normal(shape=(300, 300), stddev=0.01), name='w_cbow_en')
            b_cbow_en = tf.Variable(tf.zeros(shape=(1, 300)), name='b_cbow_en')
            tf.summary.histogram('w_cbow_en', w_cbow_en)
            tf.summary.histogram('b_cbow_en', b_cbow_en)
        with tf.name_scope('cbow_decoder'):
            w_cbow_de = tf.transpose(w_cbow_en, name='w_cbow_de')
            b_cbow_de = tf.Variable(tf.zeros(shape=(1, 300)), name='b_cbow_de')
            tf.summary.histogram('w_cbow_de', w_cbow_de)
            tf.summary.histogram('b_cbow_de', b_cbow_de)
        with tf.name_scope('glove_encoder'):
            w_glove_en = tf.Variable(tf.random_normal(shape=(300, 300), stddev=0.01), name='w_glove_en')
            b_glove_en = tf.Variable(tf.zeros(shape=(1, 300)), name='b_glove_en')
            tf.summary.histogram('w_glove_en', w_glove_en)
            tf.summary.histogram('b_glove_en', b_glove_en)
        with tf.name_scope('glove_decoder'):
            w_glove_de = tf.transpose(w_glove_en, name='w_glove_de')
            b_glove_de = tf.Variable(tf.zeros(shape=(1, 300)), name='b_glove_de')
            tf.summary.histogram('w_glove_de', w_glove_de)
            tf.summary.histogram('b_glove_de', b_glove_de)
        self.encoder['cbow'] = tf.matmul(self.input['cbow'], w_cbow_en) + b_cbow_en
        self.decoder['cbow'] = tf.matmul(self.encoder['cbow'], w_cbow_de) + b_cbow_de
        self.encoder['glove'] = tf.matmul(self.input['glove'], w_glove_en) + b_glove_en
        self.decoder['glove'] = tf.matmul(self.encoder['glove'], w_glove_de) + b_glove_de
