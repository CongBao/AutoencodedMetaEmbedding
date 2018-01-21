#!/usr/bin/env python
"""
stacked model
"""

from __future__ import division

import math
import random

import numpy as np
import tensorflow as tf

from aeme.models.core.model import Model
from aeme.utils import embed_io

__author__ = 'Cong Bao'

class SAEModel(Model):
    """A stacked autoencoder model"""

    def __init__(self, log_path):
        Model.__init__(self, self.__class__.__name__, log_path)
        self.stacked_train = {}
        self.base_graph_path = None

    def _configure(self, params):
        super(SAEModel, self)._configure(params)
        self.stacked_train = params.get('stacked_train')
        self.base_graph_path = self.graph_path

    def _redef_data(self):
        self.train_groups = []
        self.valid_groups = []
        self.logger.log('Calculating new source data for next iteration...')
        for word in self.inter_words:
            i1_cbow = self.source_dict['cbow'][word].reshape((1, 300))
            i2_glove = self.source_dict['glove'][word].reshape((1, 300))
            new_cbow, new_glove = self.session.run([self.encoder['cbow'], self.encoder['glove']],
                                                   {self.input['cbow']: i1_cbow,
                                                    self.input['glove']: i2_glove})
            new_cbow = new_cbow.reshape((self.encoder['cbow'].shape[1]))
            new_glove = new_glove.reshape((self.encoder['glove'].shape[1]))
            self.source_dict['cbow'][word] = new_cbow
            self.source_dict['glove'][word] = new_glove
        self.train_groups = [[self.source_dict['cbow'][w], self.source_dict['glove'][w]] for w in self.train_words]
        self.valid_groups = [[self.source_dict['cbow'][w], self.source_dict['glove'][w]] for w in self.valid_words]

    def _def_combine_data(self):
        self.source_dict['meta'] = {}
        for word in self.inter_words:
            self.source_dict['meta'][word] = np.concatenate([self.source_dict['cbow'][word], self.source_dict['glove'][word]])
        self.train_groups = [self.source_dict['meta'][w] for w in self.train_words]
        self.valid_groups = [self.source_dict['meta'][w] for w in self.valid_words]

    def _redef_combine_data(self):
        self.train_groups = []
        self.valid_groups = []
        self.logger.log('Calculating new source data for next iteration...')
        for word in self.inter_words:
            i_meta = self.source_dict['meta'][word].reshape((1, 600))
            new_meta = self.session.run(self.encoder['meta'], {self.input['meta']: i_meta})
            new_meta = new_meta.reshape((self.encoder['meta'].shape[1]))
            self.source_dict['meta'][word] = new_meta
        self.train_groups = [self.source_dict['meta'][w] for w in self.train_words]
        self.valid_groups = [self.source_dict['meta'][w] for w in self.valid_words]

    def _def_regular_ae(self):
        with tf.name_scope('inputs'):
            self.source['meta'] = tf.placeholder(tf.float32, (None, 600), 's_meta')
            self.input['meta'] = tf.placeholder(tf.float32, (None, 600), 'i_meta')
        with tf.name_scope('meta_ae'):
            self.reg_var = []
            self.encoder['meta'] = self.add_layer(self.input['meta'], (600, 600), self.activ_func, 'meta_encoder')
            self.decoder['meta'] = self.add_layer(self.encoder['meta'], (600, 600), None, 'meta_decoder')
        with tf.name_scope('loss'):
            diff = tf.squared_difference(self.decoder['meta'], self.source['meta'])
            self.loss = tf.reduce_mean(diff)
            if self.reg_ratio is not None:
                self.loss += tf.reduce_mean(tf.add_n([tf.nn.l2_loss(v) for v in self.reg_var]) * self.reg_ratio)
            tf.summary.scalar('loss', self.loss)
            self.valid = tf.reduce_mean(diff)
            if self.reg_ratio is not None:
                self.valid += tf.reduce_mean(tf.add_n([tf.nn.l2_loss(v) for v in self.reg_var]) * self.reg_ratio)
            tf.summary.scalar('valid', self.valid)
        with tf.name_scope('train'):
            step = tf.Variable(0, trainable=False)
            rate = tf.train.exponential_decay(self.learning_rate, step, 50, 0.999)
            self.optimizer = tf.train.AdamOptimizer(rate).minimize(self.loss, global_step=step)

    def _ae_next_batch(self, source_group):
        if self.batch_size == 1:
            for item in source_group:
                yield np.asarray([item])
        elif self.batch_size == len(source_group):
            batch = []
            for item in source_group:
                batch.append(item)
            yield np.asarray(batch)
        else:
            batch = []
            for item in source_group:
                batch.append(item)
                if len(batch) == self.batch_size:
                    yield np.asarray(batch)
                    batch = []
            for item in random.sample(source_group, self.batch_size - len(batch)):
                batch.append(item)
            yield np.asarray(batch)

    def _train_regular_ae(self):
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
                for t_batch in self._ae_next_batch(self.train_groups):
                    i_batch = self._corrupt_input(t_batch)
                    _, batch_loss = self.session.run([self.optimizer, self.loss],
                                                     {self.source['meta']: t_batch,
                                                      self.input['meta']: i_batch})
                    train_loss += batch_loss
                valid_loss = 0.
                for v_batch in self._ae_next_batch(self.valid_groups):
                    i_batch = self._corrupt_input(v_batch)
                    batch_loss = self.session.run(self.valid,
                                                  {self.source['meta']: v_batch,
                                                   self.input['meta']: i_batch})
                    valid_loss += batch_loss
                self.logger.log('[Epoch {0}] loss: {1}, validation: {2}'.format(itr, train_loss / n_train, valid_loss / n_valid))

                """
                if itr % 100 == 0:
                    test = []
                    for item in random.sample(self.source_groups, self.batch_size):
                        test.append(item)
                    result = self.session.run(self.merged_summaries,
                                              {self.source['meta']: test,
                                               self.input['meta']: test})
                    self.summary_writer.add_summary(result, itr)
                """
            self.summary_writer.close()

    def _generate_meta_embedding(self):
        self.logger.log('Saving data into output file: %s' % self.output_path)
        embed_io.save_embeddings(self.source_dict['meta'], self.output_path)
    
    def build_model(self):
        self.encoder['cbow'] = self.add_layer(self.input['cbow'], (300, 300), self.activ_func, 'cbow_encoder')
        self.decoder['cbow'] = self.add_layer(self.encoder['cbow'], (300, 300), None, 'cbow_decoder')
        self.encoder['glove'] = self.add_layer(self.input['glove'], (300, 300), self.activ_func, 'glove_encoder')
        self.decoder['glove'] = self.add_layer(self.encoder['glove'], (300, 300), None, 'glove_decoder')

    def run(self, params):
        self._configure(params)
        self._load_data()
        self._def_inputs()
        self.build_model()
        self._def_loss()
        self._def_optimizer()
        for itr in range(self.stacked_train['separate']):
            self.logger.log('Separated training: ' + str(itr + 1))
            self.graph_path = self.base_graph_path + '_sep_' + str(itr + 1)
            self._train_model()
            self._redef_data()
            self.session = tf.Session()
        self.session.close()
        tf.reset_default_graph()
        self.session = tf.Session()
        self._def_combine_data()
        self._def_regular_ae()
        for itr in range(self.stacked_train['combine']):
            self.logger.log('Combined training: ' + str(itr + 1))
            self.graph_path = self.base_graph_path + '_com_' + str(itr + 1)
            self._train_regular_ae()
            self._redef_combine_data()
            self.session = tf.Session()
        self._generate_meta_embedding()
        self.logger.log('Complete.')
