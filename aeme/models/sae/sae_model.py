#!/usr/bin/env python
"""
stacked model
"""

from __future__ import division

import random

import numpy as np
import tensorflow as tf

from aeme.models.core.model import Model
from aeme.utils import io

__author__ = 'Cong Bao'

class SAEModel(Model):

    def __init__(self, log_path):
        Model.__init__(self, self.__class__.__name__, log_path)
        self.stacked_train = {}
        self.base_graph_path = None

    def _configure(self, params):
        super(SAEModel, self)._configure(params)
        self.stacked_train = params.get('stacked_train')
        self.base_graph_path = self.graph_path

    def _redef_data(self):
        self.source_groups = []
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
            self.source_groups.append([new_cbow, new_glove])

    def _def_combine_data(self):
        self.source_dict['meta'] = {}
        for word in self.inter_words:
            self.source_dict['meta'][word] = np.concatenate([self.source_dict['cbow'][word], self.source_dict['glove'][word]])
        self.source_groups = list(self.source_dict['meta'].values())

    def _redef_combine_data(self):
        self.source_groups = []
        self.logger.log('Calculating new source data for next iteration...')
        for word in self.inter_words:
            i_meta = self.source_dict['meta'][word].reshape((1, 600))
            new_meta = self.session.run(self.encoder['meta'], {self.input['meta']: i_meta})
            new_meta = new_meta.reshape((self.encoder['meta'].shape[1]))
            self.source_dict['meta'][word] = new_meta
            self.source_groups.append(new_meta)

    def _def_regular_ae(self):
        with tf.name_scope('inputs'):
            self.source['meta'] = tf.placeholder(tf.float32, (None, 600), 's_meta')
            self.input['meta'] = tf.placeholder(tf.float32, (None, 600), 'i_meta')
        with tf.name_scope('meta_ae'):
            self.encoder['meta'] = self.add_layer(self.input['meta'], (600, 600), self.activ_func, 'meta_encoder')
            self.decoder['meta'] = self.add_layer(self.encoder['meta'], (600, 600), None, 'meta_decoder')
        with tf.name_scope('loss'):
            diff = tf.squared_difference(self.decoder['meta'], self.source['meta'])
            self.loss = tf.reduce_sum(diff)
            tf.summary.scalar('loss', self.loss)
        with tf.name_scope('train'):
            step = tf.Variable(0, trainable=False)
            rate = tf.train.exponential_decay(self.learning_rate, step, 50, 0.999)
            self.optimizer = tf.train.AdamOptimizer(rate).minimize(self.loss, global_step=step)

    def _ae_next_batch(self):
        if self.batch_size == 1:
            for item in self.source_groups:
                yield np.asarray([item])
        elif self.batch_size == len(self.source_groups):
            batch = []
            for item in self.source_groups:
                batch.append(item)
            yield np.asarray(batch)
        else:
            batch = []
            for item in self.source_groups:
                batch.append(item)
                if len(batch) == self.batch_size:
                    yield np.asarray(batch)
                    batch = []
            for item in random.sample(self.source_groups, self.batch_size - len(batch)):
                batch.append(item)
            yield np.asarray(batch)

    def _train_regular_ae(self):
        with self.session.as_default():
            self.graph = self.session.graph
            self.merged_summaries = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.graph_path, self.graph)
            self.session.run(tf.global_variables_initializer())
            for itr in range(self.epoch):
                np.random.shuffle(self.source_groups)
                total_loss = 0.
                for s_batch in self._ae_next_batch():
                    i_batch = self._corrupt_input(s_batch)
                    _, batch_loss = self.session.run([self.optimizer, self.loss],
                                                     {self.source['meta']: s_batch,
                                                      self.input['meta']: i_batch})
                    total_loss += batch_loss
                self.logger.log('Epoch {0}: {1}'.format(itr, total_loss / self.batch_size))

                if itr % 100 == 0:
                    test = []
                    for item in random.sample(self.source_groups, self.batch_size):
                        test.append(item)
                    result = self.session.run(self.merged_summaries,
                                              {self.source['meta']: test,
                                               self.input['meta']: test})
                    self.summary_writer.add_summary(result, itr)
            self.summary_writer.close()

    def _generate_meta_embedding(self):
        self.logger.log('Saving data into output file: %s' % self.output_path)
        io.save_embeddings(self.source_dict['meta'], self.output_path)
    
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
