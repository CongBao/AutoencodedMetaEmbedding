#!/usr/bin/env python
"""
model base class
"""

from __future__ import division

import random

import numpy as np
import tensorflow as tf

from aeme.utils import io, preprocess
from aeme.utils.logger import Logger

__author__ = 'Cong Bao'

class Model(object):
    """the base class of all models"""

    def __init__(self, name, log_path):
        # basic info
        self.name = name

        # file path (required)
        self.input_path = {}
        self.output_path = None
        self.log_path = log_path
        self.graph_path = None

        # init logger
        self.logger = Logger(self.name, self.log_path)

        # source info
        self.source_dict = {}
        self.inter_words = set()
        self.source_groups = []

        # training parameters (required)
        self.learning_rate = None
        self.batch_size = None
        self.epoch = None
        self.noise_type = None
        self.noise_ratio = None

        # training data
        self.source = {}
        self.input = {}
        self.encoder = {} # required
        self.decoder = {} # required
        self.loss = None
        self.optimizer = None

        # tensorflow config
        self.graph = tf.Graph()
        self.session = None
        self.saver = None
        self.merged_summaries = None
        self.summary_writer = None

    def _load_data(self):
        self.logger.log('Loading file: %s' % self.input_path['cbow'])
        self.source_dict['cbow'] = io.load_embeddings(self.input_path['cbow'])
        self.logger.log('Normalizing source embeddings: cbow')
        self.source_dict['cbow'] = preprocess.normalize_embeddings(self.source_dict['cbow'], 1.0)
        self.logger.log('Loading cbow complete')
        self.logger.log('Loading file: %s' % self.input_path['glove'])
        self.source_dict['glove'] = io.load_embeddings(self.input_path['glove'])
        self.logger.log('Normalizing source embeddings: glove')
        self.source_dict['glove'] = preprocess.normalize_embeddings(self.source_dict['glove'], 1.0)
        self.logger.log('Loading glove complete')
        self.inter_words = set(self.source_dict['cbow'].keys()) & set(self.source_dict['glove'].keys())
        self.logger.log('Number of intersection words: %s' % len(self.inter_words))
        self.source_groups = [[self.source_dict['cbow'][i], self.source_dict['glove'][i]] for i in self.inter_words]

    def _def_inputs(self):
        with tf.name_scope('inputs'):
            self.source['cbow'] = tf.placeholder(tf.float32, (None, 300), 's_cbow')
            self.source['glove'] = tf.placeholder(tf.float32, (None, 300), 's_glove')
            self.input['cbow'] = tf.placeholder(tf.float32, (None, 300), 'i_cbow')
            self.input['glove'] = tf.placeholder(tf.float32, (None, 300), 'i_glove')

    def _def_loss(self):
        with tf.name_scope('loss'):
            part1 = tf.squared_difference(self.encoder['cbow'], self.encoder['glove'])
            part2 = tf.squared_difference(self.decoder['cbow'], self.source['cbow'])
            part3 = tf.squared_difference(self.decoder['glove'], self.source['glove'])
            self.loss = tf.reduce_sum(part1) + tf.reduce_sum(part2) + tf.reduce_sum(part3)
            tf.summary.scalar('loss', self.loss)

    def _def_optimizer(self):
        with tf.name_scope('train'):
            step = tf.Variable(0, trainable=False)
            rate = tf.train.exponential_decay(self.learning_rate, step, 50, 0.999)
            self.optimizer = tf.train.AdamOptimizer(rate).minimize(self.loss, global_step=step)

    def _next_batch(self):
        if self.batch_size == 1:
            for cbow_item, glove_item in self.source_groups:
                yield (np.asarray([cbow_item]), np.asarray([glove_item]))
        elif self.batch_size == len(self.source_groups):
            cbow_batch = []
            glove_batch = []
            for cbow_item, glove_item in self.source_groups:
                cbow_batch.append(cbow_item)
                glove_batch.append(glove_item)
            yield (np.asarray(cbow_batch), np.asarray(glove_batch))
        else:
            cbow_batch = []
            glove_batch = []
            for cbow_item, glove_item in self.source_groups:
                cbow_batch.append(cbow_item)
                glove_batch.append(glove_item)
                if len(cbow_batch) == self.batch_size:
                    yield (np.asarray(cbow_batch), np.asarray(glove_batch))
                    cbow_batch = []
                    glove_batch = []
            for cbow_item, glove_item in random.sample(self.source_groups, self.batch_size - len(cbow_batch)):
                cbow_batch.append(cbow_item)
                glove_batch.append(glove_item)
            yield (np.asarray(cbow_batch), np.asarray(glove_batch))

    def _corrupt_input(self, input_batch):
        noisy_batch = np.copy(input_batch)
        batch_size, feature_size = input_batch.shape
        if self.noise_type is None:
            pass
        elif self.noise_type == 'GS':
            noisy_batch += np.random.normal(0.0, self.noise_ratio, input_batch.shape)
        elif self.noise_type == 'MN':
            for i in range(batch_size):
                mask = np.random.randint(0, feature_size, int(feature_size * self.noise_ratio))
                for m in mask:
                    noisy_batch[i][m] = 0.
        elif self.noise_type == 'SP':
            for i in range(batch_size):
                mask = np.random.randint(0, feature_size, int(feature_size * self.noise_ratio))
                for m in mask:
                    noisy_batch[i][m] = 0. if np.random.random() < 0.5 else 1.
        else:
            pass
        return noisy_batch

    def _train_model(self):
        self.session = tf.Session()
        with self.session.as_default():
            self.graph = self.session.graph
            self.merged_summaries = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.graph_path, self.graph)
            self.session.run(tf.global_variables_initializer())
            for itr in range(self.epoch):
                np.random.shuffle(self.source_groups)
                total_loss = 0.
                for s1_batch, s2_batch in self._next_batch():
                    i1_batch = self._corrupt_input(s1_batch)
                    i2_batch = self._corrupt_input(s2_batch)
                    _, batch_loss = self.session.run([self.optimizer, self.loss],
                                                     {self.source['cbow']: s1_batch,
                                                      self.source['glove']: s2_batch,
                                                      self.input['cbow']: i1_batch,
                                                      self.input['glove']: i2_batch})
                    total_loss += batch_loss
                self.logger.log('Epoch {0}: {1}'.format(itr, total_loss / self.batch_size))

                if itr % 100 == 0:
                    cbow_test = []
                    glove_test = []
                    for cbow_item, glove_item in random.sample(self.source_groups, self.batch_size):
                        cbow_test.append(cbow_item)
                        glove_test.append(glove_item)
                    result = self.session.run(self.merged_summaries,
                                              {self.source['cbow']: cbow_test,
                                               self.source['glove']: glove_test,
                                               self.input['cbow']: cbow_test,
                                               self.input['glove']: glove_test})
                    self.summary_writer.add_summary(result, itr)
            self.summary_writer.close()

    def _generate_meta_embedding(self):
        meta_embedding = {}
        self.logger.log('Generating meta embeddings...')
        for word in self.inter_words:
            i1_cbow = self.source_dict['cbow'][word].reshape((1, 300))
            i2_glove = self.source_dict['glove'][word].reshape((1, 300))
            embed_cbow, embed_glove = self.session.run([self.encoder['cbow'], self.decoder['glove']],
                                                       {self.input['cbow']: i1_cbow,
                                                        self.input['glove']: i2_glove})
            embed_cbow = embed_cbow.reshape((self.encoder['cbow'].shape[1]))
            embed_glove = embed_glove.reshape((self.encoder['glove'].shape[1]))
            meta_embedding[word] = np.concatenate([embed_cbow, embed_glove])
        self.logger.log('Saving data into output file: %s' % self.output_path)
        io.save_embeddings(meta_embedding, self.output_path)

    def add_layer(self, pre_layer, shape=None, activation_func=None, name=None):
        """ Function used to add a layer
            :param pre_layer: the previous layer
            :param shape: the shape of this layer, default None
            :param activation_func: the activation function of this layer, None if linear, default None
            :param name: the name of this layer, default None
            :return: a new layer
        """
        with tf.name_scope(name):
            weight = tf.Variable(tf.random_normal(shape=shape, stddev=0.01), name='w_' + name)
            bias = tf.Variable(tf.zeros(shape=(1, shape[1])), name='b_' + name)
            tf.summary.histogram('w_' + name, weight)
            tf.summary.histogram('b_' + name, bias)
        if activation_func is None:
            return tf.matmul(pre_layer, weight) + bias
        else:
            return activation_func(tf.matmul(pre_layer, weight) + bias)

    def build_model(self):
        """Define the encoder and decoder here"""
        raise NotImplementedError('Model Not Defined')

    def run(self):
        """Use this function to run and train model"""
        self._load_data()
        self._def_inputs()
        self.build_model()
        self._def_loss()
        self._def_optimizer()
        self._train_model()
        self._generate_meta_embedding()
        self.logger.log('Complete.')
