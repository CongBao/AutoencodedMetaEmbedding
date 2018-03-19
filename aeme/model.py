# Model of AEME

from __future__ import division

import numpy as np
import sklearn.preprocessing as skpre
import tensorflow as tf

from utils import load_emb, save_emb
from logger import Logger

__author__ = 'Cong Bao'

class AEME(object):

    def __init__(self, **kwargs):
        self.input_list = kwargs['input'] # [path, ...]
        self.output_path = kwargs['output']
        self.log_path = kwargs['log']
        self.graph_path = kwargs['graph']
        self.checkpoint_path = kwargs['checkpoint']
        self.model_type = kwargs['model']
        self.dims = kwargs['dims']
        self.learning_rate = kwargs['learning_rate']
        self.batch_size = kwargs['batch']
        self.epoch = kwargs['epoch']
        self.activ = kwargs['activ']
        self.factors = kwargs['factors']
        self.noise = kwargs['noise']

        self.logger = Logger(self.model_type, self.log_path)

        self.inter_words = []
        self.sources = []

        self.sess = tf.Session()

    def load_data(self):
        src_dict_list = [load_emb(path, self.logger.log) for path in self.input_list]
        self.inter_words = list(set.intersection(*[set(src_dict.keys()) for src_dict in src_dict_list]))
        self.logger.log('Intersection Words: %s' % len(self.inter_words))
        self.sources = list(zip(*[skpre.normalize([src_dict[word] for word in self.inter_words]) for src_dict in src_dict_list]))
        del src_dict_list
        self.origin = self.sources[:]

    def build_model(self):
        self.srcs = [tf.placeholder(tf.float32, (None, dim)) for dim in self.dims]
        self.ipts = [tf.placeholder(tf.float32, (None, dim)) for dim in self.dims]
        params = [self.dims, self.activ, self.noise, self.factors]
        self.aeme = AbsModel(*params)
        if self.model_type == 'DAEME':
            self.aeme = DAEME(*params)
        elif self.model_type == 'CAEME':
            self.aeme = CAEME(*params)
        elif self.model_type == 'AAEME':
            self.aeme = AAEME(*params)
        self.aeme.build(self.srcs, self.ipts)

    def train_model(self):
        step = tf.Variable(0, trainable=False)
        rate = tf.train.exponential_decay(self.learning_rate, step, 50, 0.999)
        loss = self.aeme.loss()
        opti = tf.train.AdamOptimizer(rate).minimize(loss, global_step=step)
        self.sess.run(tf.global_variables_initializer())
        num = len(self.sources) // self.batch_size + 1
        try:
            for itr in range(self.epoch):
                np.random.shuffle(self.sources)
                train_loss = 0.
                for batches in self._next_batch(self.sources):
                    feed = {k:v for k in self.srcs for v in batches}
                    feed.update({k:self._corrupt(v) for k in self.ipts for v in batches})
                    _, batch_loss = self.sess.run([opti, loss], feed)
                    train_loss += batch_loss
                self.logger.log('[Epoch{0}]: loss: {1}'.format(itr, train_loss / num))
        except (KeyboardInterrupt, SystemExit):
            self.sess.close()
        finally:
            del self.sources

    def generate_meta_embed(self):
        embed= {}
        self.batch_size = 1
        try:
            for word, batch in zip(self.inter_words, self._next_batch(self.origin)):
                meta = self.sess.run(self.aeme.extract(), {k:v for k in self.ipts for v in batch})
                embed[word] = np.reshape(meta, (np.shape(meta)[1],))
        except (KeyboardInterrupt, SystemExit):
            raise
        finally:
            self.sess.close()
            del self.origin
        save_emb(embed, self.output_path, self.logger.log)

    def _next_batch(self, source):
        if self.batch_size <= 1:
            for items in source:
                yield [np.asarray([x]) for x in items]
        elif self.batch_size >= len(source):
            yield [np.asarray(x) for x in zip(*source)]
        else:
            for n in range(0, len(source), self.batch_size):
                yield [np.asarray(x) for x in zip(*source[n:n+self.batch_size])]

    def _corrupt(self, batch):
        noised = np.copy(batch)
        batch_size, feature_size = np.shape(batch)
        for i in range(batch_size):
            mask = np.random.randint(0, feature_size, int(feature_size * self.noise))
            for m in mask:
                noised[i][m] = 0.
        return noised

class AbsModel(object):

    def __init__(self, dims, activ, noise, factors):
        self.dims = dims # [dim, ...]
        self.activ = tf.keras.layers.Activation(activ)
        self.noise = noise
        self.factors = factors

        self.meta = None
        self.outs = None
        self.ipts = None
        self.encoders = None

    @staticmethod
    def mse(x, y, f):
        return f * tf.reduce_mean(tf.squared_difference(x, y))

    def extract(self):
        return self.meta

    def build(self, srcs, ipts):
        self.srcs = srcs
        self.ipts = ipts
        self.encoders = [tf.layers.Dense(min(self.dims), activation=self.activ)(ipt) for ipt in self.ipts]

    def loss(self):
        raise NotImplementedError('Loss Function Undefined')

class DAEME(AbsModel):

    def build(self, srcs, ipts):
        AbsModel.build(self, srcs, ipts)
        self.meta = tf.nn.l2_normalize(tf.concat(self.encoders, 1), axis=1)
        self.outs = [tf.layers.Dense(dim)(encoder) for dim, encoder in zip(self.dims, self.encoders)]

    def loss(self):
        ael = sum([self.mse(x, y, f) for x, y, f in zip(self.ipts, self.outs, self.factors[:-1])])
        mtl = 0.
        for i in range(len(self.encoders)):
            for j in range(i + 1, len(self.encoders)):
                mtl += self.mse(self.encoders[i], self.encoders[j], self.factors[-1])
        return ael + mtl

class CAEME(AbsModel):

    def build(self, srcs, ipts):
        AbsModel.build(self, srcs, ipts)
        self.meta = tf.nn.l2_normalize(tf.concat(self.encoders, 1), axis=1)
        self.outs = [tf.layers.Dense(dim)(self.meta) for dim in self.dims]

    def loss(self):
        return sum([self.mse(x, y, f) for x, y, f in zip(self.srcs, self.outs, self.factors)])

class AAEME(AbsModel):

    def build(self, srcs, ipts):
        AbsModel.build(self, srcs, ipts)
        self.meta = tf.nn.l2_normalize(tf.add_n(self.encoders), axis=1)
        self.outs = [tf.layers.Dense(dim)(self.meta) for dim in self.dims]

    def loss(self):
        return sum([self.mse(x, y, f) for x, y, f in zip(self.srcs, self.outs, self.factors)])
