#!/usr/bin/env python
"""
regular model
"""

from __future__ import division

import numpy as np
import tensorflow as tf

from aeme.models.core.model import Model
from aeme.utils import embed_io

__author__ = 'Cong Bao'

class AEModel(Model):
    """A regular autoencoder model"""

    def __init__(self, log_path):
        Model.__init__(self, self.__class__.__name__, log_path)

    def build_model(self):
        self.encoder['cbow'] = self.add_layer(self.input['cbow'], (300, 300), self.activ_func, 'cbow_encoder')
        self.decoder['cbow'] = self.add_layer(self.encoder['cbow'], (300, 300), None, 'cbow_decoder')
        self.encoder['glove'] = self.add_layer(self.input['glove'], (300, 300), self.activ_func, 'glove_encoder')
        self.decoder['glove'] = self.add_layer(self.encoder['glove'], (300, 300), None, 'glove_decoder')

class TiedAEModel(Model):
    """An autoencoder model with tied weights: W_D = W_E.T"""
    
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
        if self.reg_ratio is not None:
            self.reg_var.append(w_cbow_en)
            self.reg_var.append(w_cbow_de)
            self.reg_var.append(w_glove_en)
            self.reg_var.append(w_glove_de)
        self.encoder['cbow'] = self.activ_func(tf.matmul(self.input['cbow'], w_cbow_en) + b_cbow_en)
        self.decoder['cbow'] = tf.matmul(self.encoder['cbow'], w_cbow_de) + b_cbow_de
        self.encoder['glove'] = self.activ_func(tf.matmul(self.input['glove'], w_glove_en) + b_glove_en)
        self.decoder['glove'] = tf.matmul(self.encoder['glove'], w_glove_de) + b_glove_de

class ZipAEModel(Model):
    """An autoencoder model with hidden layer size half as input layer size"""

    def __init__(self, log_path):
        Model.__init__(self, self.__class__.__name__, log_path)

    def build_model(self):
        self.encoder['cbow'] = self.add_layer(self.input['cbow'], (300, 150), self.activ_func, 'cbow_encoder')
        self.decoder['cbow'] = self.add_layer(self.encoder['cbow'], (150, 300), None, 'cbow_decoder')
        self.encoder['glove'] = self.add_layer(self.input['glove'], (300, 150), self.activ_func, 'glove_encoder')
        self.decoder['glove'] = self.add_layer(self.encoder['glove'], (150, 300), None, 'glove_decoder')

class DActivAEModel(Model):
    
    def __init__(self, log_path):
        Model.__init__(self, self.__class__.__name__, log_path)

    def build_model(self):
        self.encoder['cbow'] = self.add_layer(self.input['cbow'], (300, 300), self.activ_func, 'cbow_encoder')
        self.decoder['cbow'] = self.add_layer(self.encoder['cbow'], (300, 300), self.activ_func, 'cbow_decoder')
        self.encoder['glove'] = self.add_layer(self.input['glove'], (300, 300), self.activ_func, 'glove_encoder')
        self.decoder['glove'] = self.add_layer(self.encoder['glove'], (300, 300), self.activ_func, 'glove_decoder')

class DeepAEModel(Model):
    """A deep autoencoder model"""

    def __init__(self, log_path):
        Model.__init__(self, self.__class__.__name__, log_path)

    def build_model(self):
        cbow_en_h = self.add_layer(self.input['cbow'], (300, 300), self.activ_func, 'cbow_encoder_h')
        self.encoder['cbow'] = self.add_layer(cbow_en_h, (300, 300), self.activ_func, 'cbow_encoder')
        cbow_de_h = self.add_layer(self.input['cbow'], (300, 300), self.activ_func, 'cbow_decoder_h')
        self.decoder['cbow'] = self.add_layer(cbow_de_h, (300, 300), None, 'cbow_decoder')
        glove_en_h = self.add_layer(self.input['glove'], (300, 300), self.activ_func, 'glove_encoder_h')
        self.encoder['glove'] = self.add_layer(glove_en_h, (300, 300), self.activ_func, 'glove_encoder')
        glove_de_h = self.add_layer(self.input['glove'], (300, 300), self.activ_func, 'glove_decoder_h')
        self.decoder['glove'] = self.add_layer(glove_de_h, (300, 300), None, 'glove_decoder')

class AvgAEModel(Model):

    def __init__(self, log_path):
        Model.__init__(self, self.__class__.__name__, log_path)

    def _def_loss(self):
        with tf.name_scope('loss'):
            part1 = tf.squared_difference(self.decoder['cbow'], self.source['cbow'])
            part2 = tf.squared_difference(self.decoder['glove'], self.source['glove'])
            f1, f2, _ = self.factors
            self.loss = f1 * tf.reduce_mean(part1) + f2 * tf.reduce_mean(part2)
            if self.reg_ratio is not None:
                self.loss += tf.reduce_mean(tf.add_n([tf.nn.l2_loss(v) for v in self.reg_var]) * self.reg_ratio)
            tf.summary.scalar('loss', self.loss)
            self.valid = f1 * tf.reduce_mean(part1) + f2 * tf.reduce_mean(part2)
            if self.reg_ratio is not None:
                self.valid += tf.reduce_mean(tf.add_n([tf.nn.l2_loss(v) for v in self.reg_var]) * self.reg_ratio)
            tf.summary.scalar('valid', self.valid)

    def build_model(self):
        self.encoder['cbow'] = self.add_layer(self.input['cbow'], (300, 300), self.activ_func, 'cbow_encoder')
        self.encoder['glove'] = self.add_layer(self.input['glove'], (300, 300), self.activ_func, 'glove_encoder')
        meta_emb = tf.nn.l2_normalize(tf.add(self.encoder['cbow'], self.encoder['glove']), 1)
        self.decoder['cbow'] = self.add_layer(meta_emb, (300, 300), None, 'cbow_decoder')
        self.decoder['glove'] = self.add_layer(meta_emb, (300, 300), None, 'glove_decoder')
