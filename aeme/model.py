# Model of AEME

from __future__ import division

from abc import ABCMeta, abstractmethod
from itertools import combinations

import numpy as np
import sklearn.preprocessing as skpre
import tensorflow as tf

from utils import Logger, Utils

__author__ = 'Cong Bao'

class AEME(object):

    def __init__(self, **kwargs):
        self.input_list = kwargs['input'] # [path, ...]
        self.output_path = kwargs['output']
        self.log_path = kwargs['log']
        self.ckpt_path = kwargs['ckpt'] + 'model.ckpt'
        self.model_type = kwargs['model']
        self.dims = kwargs['dims']
        self.learning_rate = kwargs['learning_rate']
        self.batch_size = kwargs['batch']
        self.epoch = kwargs['epoch']
        self.activ = kwargs['activ']
        self.factors = kwargs['factors']
        self.noise = kwargs['noise']
        self.emb_dim = kwargs['emb']
        self.oov = kwargs['oov']
        self.restore = kwargs['restore']

        self.logger = Logger(self.model_type, self.log_path)
        self.utils = Utils(self.logger.log)

        self.sess = tf.Session()
        self.ckpt = None

    def load_data(self):
        src_dict_list = [self.utils.load_emb(path) for path in self.input_list]
        if self.oov:
            self.union_words = list(set.union(*[set(src_dict.keys()) for src_dict in src_dict_list]))
            self.logger.log('Union Words: %s' % len(self.union_words))
            source = []
            for i, src_dict in enumerate(src_dict_list):
                embed_mat = []
                for word in self.union_words:
                    embed = src_dict.get(word)
                    if embed is not None:
                        embed_mat.append(embed)
                    else:
                        embed_mat.append(np.zeros(self.dims[i]))
                source.append(skpre.normalize(embed_mat))
            self.sources = np.asarray(list(zip(*source)))
        else:
            self.inter_words = list(set.intersection(*[set(src_dict.keys()) for src_dict in src_dict_list]))
            self.logger.log('Intersection Words: %s' % len(self.inter_words))
            self.sources = np.asarray(list(zip(*[skpre.normalize([src_dict[word] for word in self.inter_words]) for src_dict in src_dict_list])))
        del src_dict_list

    def build_model(self):
        self.srcs = [tf.placeholder(tf.float32, (None, dim)) for dim in self.dims]
        self.ipts = [tf.placeholder(tf.float32, (None, dim)) for dim in self.dims]
        params = [self.dims, self.activ, self.noise, self.factors]
        if self.model_type == 'DAEME':
            self.aeme = DAEME(*params)
        elif self.model_type == 'CAEME':
            self.aeme = CAEME(*params)
        elif self.model_type == 'AAEME':
            self.aeme = AAEME(*params, emb_dim=self.emb_dim)
        self.aeme.build(self.srcs, self.ipts)

    def train_model(self):
        step = tf.Variable(0, trainable=False)
        rate = tf.train.exponential_decay(self.learning_rate, step, 50, 0.99)
        loss = self.aeme.loss()
        opti = tf.train.AdamOptimizer(rate).minimize(loss, global_step=step)
        self.ckpt = tf.train.Saver(tf.global_variables())
        if self.restore:
            self.ckpt.restore(self.sess, self.ckpt_path)
        else:
            self.sess.run(tf.global_variables_initializer())
        size = len(self.sources) // self.batch_size
        best = float('inf')
        for itr in range(self.epoch):
            indexes = np.random.permutation(len(self.sources))
            train_loss = 0.
            for idx in range(size):
                batch_idx = indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
                batches = list(zip(*self.sources[batch_idx]))
                feed = {k:v for k, v in zip(self.srcs, batches)}
                feed.update({k:self._corrupt(v) for k, v in zip(self.ipts, batches)})
                _, batch_loss = self.sess.run([opti, loss], feed)
                train_loss += batch_loss
            epoch_loss = train_loss / size
            if epoch_loss < best:
                self.ckpt.save(self.sess, self.ckpt_path)
                best = epoch_loss
            self.logger.log('[Epoch{0}] loss: {1}'.format(itr, epoch_loss))

    def generate_meta_embed(self):
        embed= {}
        self.logger.log('Generating meta embeddings...')
        self.ckpt.restore(self.sess, self.ckpt_path)
        if self.oov:
            vocabulary = self.union_words
        else:
            vocabulary = self.inter_words
        for i, word in enumerate(vocabulary):
            meta = self.sess.run(self.aeme.extract(), {k:[v] for k, v in zip(self.ipts, self.sources[i])})
            embed[word] = np.reshape(meta, (np.shape(meta)[1],))
        self.sess.close()
        del self.sources
        self.utils.save_emb(embed, self.output_path)

    def _corrupt(self, batch):
        noised = np.copy(batch)
        batch_size, feature_size = np.shape(batch)
        for i in range(batch_size):
            mask = np.random.randint(0, feature_size, int(feature_size * self.noise))
            for m in mask:
                noised[i][m] = 0.
        return noised

class AbsModel(object):

    __metaclass__ = ABCMeta

    def __init__(self, dims, activ, noise, factors):
        self.dims = dims # [dim, ...]
        self.noise = noise
        self.factors = factors

        if activ == 'lrelu':
            self.activ = tf.keras.layers.LeakyReLU(0.2)
        elif activ == 'prelu':
            self.activ = tf.keras.layers.PReLU()
        else:
            self.activ = tf.keras.layers.Activation(activ)

        self.meta = None

    @staticmethod
    def mse(x, y, f):
        x_d = x.get_shape().as_list()[1]
        y_d = y.get_shape().as_list()[1]
        if x_d != y_d:
            smaller = min(x_d, y_d)
            x = tf.slice(x, [0, 0], [tf.shape(x)[0], smaller])
            y = tf.slice(y, [0, 0], [tf.shape(y)[0], smaller])
        return tf.scalar_mul(f, tf.reduce_mean(tf.squared_difference(x, y)))

    def extract(self):
        return self.meta

    @abstractmethod
    def build(self, srcs, ipts):
        self.srcs = srcs
        self.ipts = ipts

    @abstractmethod
    def loss(self):
        pass

class DAEME(AbsModel):

    def build(self, srcs, ipts):
        AbsModel.build(self, srcs, ipts)
        self.encoders = [tf.layers.dense(ipt, dim, self.activ) for ipt, dim in zip(self.ipts, self.dims)]
        self.meta = tf.nn.l2_normalize(tf.concat(self.encoders, 1), 1)
        self.outs = [tf.layers.dense(encoder, dim) for encoder, dim in zip(self.encoders, self.dims)]

    def loss(self):
        los = tf.add_n([self.mse(x, y, f) for x, y, f in zip(self.srcs, self.outs, self.factors[:-1])])
        for x, y in combinations(self.encoders, 2):
            los = tf.add(los, self.mse(x, y, self.factors[-1]))
        return los

class CAEME(AbsModel):

    def build(self, srcs, ipts):
        AbsModel.build(self, srcs, ipts)
        self.encoders = [tf.layers.dense(ipt, dim, self.activ) for ipt, dim in zip(self.ipts, self.dims)]
        self.meta = tf.nn.l2_normalize(tf.concat(self.encoders, 1), 1)
        self.outs = [tf.layers.dense(self.meta, dim) for dim in self.dims]

    def loss(self):
        return tf.add_n([self.mse(x, y, f) for x, y, f in zip(self.srcs, self.outs, self.factors)])

class AAEME(AbsModel):

    def __init__(self, *args, **kwargs):
        AbsModel.__init__(self, *args)
        self.emb_dim = kwargs['emb_dim']

    def build(self, srcs, ipts):
        AbsModel.build(self, srcs, ipts)
        self.encoders = [tf.layers.dense(ipt, self.emb_dim, self.activ) for ipt in self.ipts]
        self.meta = tf.nn.l2_normalize(tf.add_n(self.encoders), 1)
        self.outs = [tf.layers.dense(self.meta, dim) for dim in self.dims]

    def loss(self):
        return tf.add_n([self.mse(x, y, f) for x, y, f in zip(self.srcs, self.outs, self.factors)])
