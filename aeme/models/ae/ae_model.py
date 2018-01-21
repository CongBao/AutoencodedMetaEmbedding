#!/usr/bin/env python
"""
regular model
"""

from __future__ import division

import numpy as np
import tensorflow as tf

from aeme.models.core.model import Model
from aeme.utils import io

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

class SpecialAEModel(Model):

    def __init__(self, log_path):
        Model.__init__(self, self.__class__.__name__, log_path)

    def _def_inputs(self):
        with tf.name_scope('inputs'):
            self.source['cbow'] = tf.placeholder(tf.float32, (None, 200), 's_cbow')
            self.source['glove'] = tf.placeholder(tf.float32, (None, 200), 's_glove')
            self.input['cbow'] = tf.placeholder(tf.float32, (None, 200), 'i_cbow')
            self.input['glove'] = tf.placeholder(tf.float32, (None, 200), 'i_glove')

    def build_model(self):
        self.encoder['cbow'] = self.add_layer(self.input['cbow'], (200, 200), self.activ_func, 'cbow_encoder')
        self.decoder['cbow'] = self.add_layer(self.encoder['cbow'], (200, 200), None, 'cbow_decoder')
        self.encoder['glove'] = self.add_layer(self.input['glove'], (200, 200), self.activ_func, 'glove_encoder')
        self.decoder['glove'] = self.add_layer(self.encoder['glove'], (200, 200), None, 'glove_decoder')

    def _generate_meta_embedding(self):
        meta_embedding = {}
        self.logger.log('Generating meta embeddings...')
        for word in self.inter_words:
            i1_cbow = self.source_dict['cbow'][word].reshape((1, 200))
            i2_glove = self.source_dict['glove'][word].reshape((1, 200))
            embed_cbow, embed_glove = self.session.run([self.encoder['cbow'], self.encoder['glove']],
                                                       {self.input['cbow']: i1_cbow,
                                                        self.input['glove']: i2_glove})
            embed_cbow = embed_cbow.reshape((self.encoder['cbow'].shape[1]))
            embed_glove = embed_glove.reshape((self.encoder['glove'].shape[1]))
            meta_embedding[word] = np.concatenate([embed_cbow, embed_glove])
        self.logger.log('Saving data into output file: %s' % self.output_path)
        io.save_embeddings(meta_embedding, self.output_path)
