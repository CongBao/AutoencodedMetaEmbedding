#!/usr/bin/env python
"""
model base class
"""

from __future__ import division

import math
import random

import numpy as np
import tensorflow as tf

from aeme.utils import io, data_process
from aeme.utils.logger import Logger

__author__ = 'Cong Bao'

class Model(object):
    """The base class of all models"""

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
        self.train_words = set()
        self.valid_words = set()
        self.source_groups = []
        self.train_groups = []
        self.valid_groups = []

        # training parameters (required)
        self.learning_rate = None
        self.batch_size = None
        self.epoch = None
        self.valid_ratio = None
        self.reg_ratio = None
        self.activ_func = None
        self.factors = None
        self.noise_type = None
        self.noise_ratio = None
        self.meta_type = None

        # training data
        self.source = {}
        self.input = {}
        self.encoder = {} # required
        self.decoder = {} # required
        self.reg_var = []
        self.loss = None
        self.valid = None
        self.optimizer = None

        # tensorflow config
        self.graph = tf.Graph()
        self.session = tf.Session()
        self.saver = None
        self.merged_summaries = None
        self.summary_writer = None

    def _configure(self, params):
        """ Configure parameters or hyperparameters
            :param params: a dict of parameters
        """
        self.input_path = params.get('input_path')
        self.output_path = params.get('output_path')
        self.graph_path = params.get('graph_path', './graphs/') + self.name
        self.learning_rate = params.get('learning_rate', 0.001)
        self.batch_size = params.get('batch_size', 64)
        self.epoch = params.get('epoch', 1000)
        self.valid_ratio = params.get('valid_ratio', 0.1)
        self.reg_ratio = params.get('reg_ratio', None)
        activ_type = params.get('activ_func')
        if activ_type == 'sigmoid_m':
            self.activ_func = lambda x: tf.nn.sigmoid(x) - 0.5
        elif activ_type == 'sigmoid':
            self.activ_func = tf.nn.sigmoid
        elif activ_type == 'tanh':
            self.activ_func = tf.nn.tanh
        elif activ_type == 'relu':
            self.activ_func = tf.nn.relu
        else:
            self.activ_func = None
        self.factors = params.get('factors', (1.0, 1.0, 1.0))
        self.noise_type = params.get('noise_type', 'MN')
        self.noise_ratio = params.get('noise_ratio', 0.2)
        self.meta_type = params.get('meta_type', 'conc')

    def _load_data(self):
        self.logger.log('Loading file: %s' % self.input_path['cbow'])
        self.source_dict['cbow'] = io.load_embeddings(self.input_path['cbow'])
        self.logger.log('Normalizing source embeddings: cbow')
        self.source_dict['cbow'] = data_process.normalize_embeddings(self.source_dict['cbow'], 1.0)
        self.logger.log('Loading cbow complete')
        self.logger.log('Loading file: %s' % self.input_path['glove'])
        self.source_dict['glove'] = io.load_embeddings(self.input_path['glove'])
        self.logger.log('Normalizing source embeddings: glove')
        self.source_dict['glove'] = data_process.normalize_embeddings(self.source_dict['glove'], 1.0)
        self.logger.log('Loading glove complete')
        self.inter_words = set(self.source_dict['cbow'].keys()) & set(self.source_dict['glove'].keys())
        self.logger.log('Number of intersection words: %s' % len(self.inter_words))
        #self.source_groups = [[self.source_dict['cbow'][w], self.source_dict['glove'][w]] for w in self.inter_words]
        self.valid_words = set(random.sample(self.inter_words, int(self.valid_ratio * len(self.inter_words))))
        self.train_words = self.inter_words - self.valid_words
        self.logger.log('Number of training words: %s' % len(self.train_words))
        self.logger.log('Number of validation words: %s' % len(self.valid_words))
        self.train_groups = [[self.source_dict['cbow'][w], self.source_dict['glove'][w]] for w in self.train_words]
        self.valid_groups = [[self.source_dict['cbow'][w], self.source_dict['glove'][w]] for w in self.valid_words]

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
            f1, f2, f3 = self.factors
            self.loss = f1 * tf.reduce_mean(part1) + f2 * tf.reduce_mean(part2) + f3 * tf.reduce_mean(part3)
            if self.reg_ratio is not None:
                self.loss += tf.reduce_mean(tf.add_n([tf.nn.l2_loss(v) for v in self.reg_var]) * self.reg_ratio)
            tf.summary.scalar('loss', self.loss)
            self.valid = f1 * tf.reduce_mean(part1) + f2 * tf.reduce_mean(part2) + f3 * tf.reduce_mean(part3)
            if self.reg_ratio is not None:
                self.valid += tf.reduce_mean(tf.add_n([tf.nn.l2_loss(v) for v in self.reg_var]) * self.reg_ratio)
            tf.summary.scalar('valid', self.valid)

    def _def_optimizer(self):
        with tf.name_scope('train'):
            step = tf.Variable(0, trainable=False)
            rate = tf.train.exponential_decay(self.learning_rate, step, 50, 0.999)
            self.optimizer = tf.train.AdamOptimizer(rate).minimize(self.loss, global_step=step)

    def _next_batch(self, source_group):
        if self.batch_size == 1:
            for cbow_item, glove_item in source_group:
                yield (np.asarray([cbow_item]), np.asarray([glove_item]))
        elif self.batch_size == len(source_group):
            cbow_batch = []
            glove_batch = []
            for cbow_item, glove_item in source_group:
                cbow_batch.append(cbow_item)
                glove_batch.append(glove_item)
            yield (np.asarray(cbow_batch), np.asarray(glove_batch))
        else:
            cbow_batch = []
            glove_batch = []
            for cbow_item, glove_item in source_group:
                cbow_batch.append(cbow_item)
                glove_batch.append(glove_item)
                if len(cbow_batch) == self.batch_size:
                    yield (np.asarray(cbow_batch), np.asarray(glove_batch))
                    cbow_batch = []
                    glove_batch = []
            for cbow_item, glove_item in random.sample(source_group, self.batch_size - len(cbow_batch)):
                cbow_batch.append(cbow_item)
                glove_batch.append(glove_item)
            yield (np.asarray(cbow_batch), np.asarray(glove_batch))

    def _corrupt_input(self, input_batch):
        noisy_batch = np.copy(input_batch)
        batch_size, feature_size = input_batch.shape
        if self.noise_type is None:
            pass
        elif self.noise_type == 'GS':
            noisy_batch += np.random.normal(0.0, self.noise_ratio ** 0.5, input_batch.shape)
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
        with self.session.as_default():
            self.graph = self.session.graph
            self.merged_summaries = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.graph_path, self.graph)
            self.session.run(tf.global_variables_initializer())
            n_train = math.ceil(len(self.train_groups) / self.batch_size)
            n_valid = math.ceil(len(self.valid_groups) / self.batch_size)
            for itr in range(self.epoch):
                np.random.shuffle(self.train_groups)
                train_loss = 0.
                for t1_batch, t2_batch in self._next_batch(self.train_groups):
                    i1_batch = self._corrupt_input(t1_batch)
                    i2_batch = self._corrupt_input(t2_batch)
                    _, batch_loss = self.session.run([self.optimizer, self.loss],
                                                     {self.source['cbow']: t1_batch,
                                                      self.source['glove']: t2_batch,
                                                      self.input['cbow']: i1_batch,
                                                      self.input['glove']: i2_batch})
                    train_loss += batch_loss
                valid_loss = 0.
                for v1_batch, v2_batch in self._next_batch(self.valid_groups):
                    i1_batch = self._corrupt_input(v1_batch)
                    i2_batch = self._corrupt_input(v2_batch)
                    batch_loss = self.session.run(self.valid,
                                                  {self.source['cbow']: v1_batch,
                                                   self.source['glove']: v2_batch,
                                                   self.input['cbow']: i1_batch,
                                                   self.input['glove']: i2_batch})
                    valid_loss += batch_loss
                self.logger.log('[Epoch {0}] loss: {1}, validation: {2}'.format(itr, train_loss / n_train, valid_loss / n_valid))

                """
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
                """
            self.summary_writer.close()

    def _generate_meta_embedding(self):
        meta_embedding = {}
        self.logger.log('Generating meta embeddings...')
        for word in self.inter_words:
            i1_cbow = self.source_dict['cbow'][word].reshape((1, 300))
            i2_glove = self.source_dict['glove'][word].reshape((1, 300))
            embed_cbow, embed_glove = self.session.run([self.encoder['cbow'], self.encoder['glove']],
                                                       {self.input['cbow']: i1_cbow,
                                                        self.input['glove']: i2_glove})
            embed_cbow = embed_cbow.reshape((self.encoder['cbow'].shape[1]))
            embed_glove = embed_glove.reshape((self.encoder['glove'].shape[1]))
            if self.meta_type == 'src1':
                meta_embedding[word] = embed_cbow
            elif self.meta_type == 'src2':
                meta_embedding[word] = embed_glove
            elif self.meta_type == 'avg':
                meta_embedding[word] = data_process.normalize(np.add(embed_cbow, embed_glove), 1.0)
            else:
                meta_embedding[word] = np.concatenate([embed_cbow, embed_glove])
        if self.meta_type == 'svd':
            meta_embedding = data_process.tsvd(meta_embedding)
        self.logger.log('Saving data into output file: %s' % self.output_path)
        io.save_embeddings(meta_embedding, self.output_path)

    def add_layer(self, pre_layer, shape=None, activ_func=None, name=None):
        """ Function used to add a layer
            :param pre_layer: the previous layer
            :param shape: the shape of this layer, default None
            :param activ_func: the activation function of this layer, None if linear, default None
            :param name: the name of this layer, default None
            :return: a new layer
        """
        with tf.name_scope(name):
            weight = tf.Variable(tf.random_normal(shape=shape, stddev=0.01), name='w_' + name)
            bias = tf.Variable(tf.zeros(shape=(1, shape[1])), name='b_' + name)
            tf.summary.histogram('w_' + name, weight)
            tf.summary.histogram('b_' + name, bias)
        if self.reg_ratio is not None:
            self.reg_var.append(weight)
        if activ_func is None:
            return tf.matmul(pre_layer, weight) + bias
        else:
            return activ_func(tf.matmul(pre_layer, weight) + bias)

    def build_model(self):
        """Define the encoder and decoder here"""
        raise NotImplementedError('Model Not Defined')

    def run(self, params):
        """ Use this function to run and train model
            :param params: a dict of parameters
        """
        self._configure(params)
        self._load_data()
        self._def_inputs()
        self.build_model()
        self._def_loss()
        self._def_optimizer()
        self._train_model()
        self._generate_meta_embedding()
        self.logger.log('Complete.')
