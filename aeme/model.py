# Model of AEME

from __future__ import print_function

import numpy as np
import sklearn.preprocessing as skpre
import tensorflow as tf

from utils import load_emb, save_emb

__author__ = 'Cong Bao'

class AEME(object):

    def __init__(self, **kwargs):
        self.input_list = kwargs['input'] # [path, ...]
        self.output_path = kwargs['output']
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

        self.inter_words = []
        self.sources = []

        self.aeme = None
        self.model = None

    def load_data(self):
        src_dict_list = [load_emb(path) for path in self.input_list]
        self.inter_words = set.intersection(*[set(src_dict.keys()) for src_dict in src_dict_list])
        print('Intersection Words: %s' % len(self.inter_words))
        self.sources = [skpre.normalize([src_dict[word] for word in self.inter_words]) for src_dict in src_dict_list]
        del src_dict_list

    def build_model(self):
        params = [self.dims, self.activ, self.noise, self.factors]
        self.aeme = AbsModel(*params)
        if self.model_type == 'DAEME':
            self.aeme = DAEME(*params)
        elif self.model_type == 'CAEME':
            self.aeme = CAEME(*params)
        elif self.model_type == 'AAEME':
            self.aeme = AAEME(*params)
        self.model = self.aeme.build()

    def train_model(self):
        step = tf.Variable(0, trainable=False)
        rate = tf.train.exponential_decay(self.learning_rate, step, 50, 0.999)
        opti = tf.train.AdamOptimizer(rate).minimize(self.aeme.loss(), global_step=step)

    def _next_batch(self):
        if self.batch_size == 1:
            yield from zip(*self.sources)
        elif self.batch_size == len(self.sources[0]):
            pass

    def generate_meta_embed(self):
        generator = self.aeme.extract()
        generated = generator.predict(self.sources, batch_size=self.batch_size, verbose=1)
        save_emb({k: v for k, v in zip(self.inter_words, generated)}, self.output_path)

class AbsModel(object):

    def __init__(self, dims, activ, noise, factors):
        self.dims = dims # [dim, ...]
        self.activ = activ
        self.noise = noise
        self.factors = factors

        self.meta = None
        self.outs = None
        self.srcs = [tf.placeholder(tf.float32, (None, dim)) for dim in self.dims]
        self.ipts = [tf.placeholder(tf.float32, (None, dim)) for dim in self.dims]
        self.encoders = [tf.layers.Dense(min(self.dims), activation=self.activ)(ipt) for ipt in self.ipts]

    def extract(self):
        #return Model(self.srcs, self.meta)
        pass

    def build(self):
        raise NotImplementedError('Model Undefined')

    def loss(self):
        raise NotImplementedError('Loss Function Undefined')

class DAEME(AbsModel):

    def build(self):
        self.meta = tf.nn.l2_normalize(tf.concat(self.encoders, 1), axis=1)
        self.outs = [tf.layers.Dense(dim)(encoder) for dim, encoder in zip(self.dims, self.encoders)]
        return self.srcs, self.ipts

    def loss(self):
        mse = lambda x, y, f: f * tf.reduce_mean(tf.squared_difference(x, y))
        ael = sum([mse(x, y, f) for x, y, f in zip(self.srcs, self.outs, self.factors[:-1])])
        mtl = 0.
        for i in range(len(self.encoders)):
            for j in range(i + 1, len(self.encoders)):
                mtl += mse(self.encoders[i], self.encoders[j], self.factors[-1])
        return ael + mtl

class CAEME(AbsModel):

    def build(self):
        self.meta = tf.nn.l2_normalize(tf.concat(self.encoders, 1), axis=1)
        self.outs = [tf.layers.Dense(dim)(self.meta) for dim in self.dims]
        return self.srcs, self.ipts

    def loss(self):
        mse = lambda x, y, f: f * tf.reduce_mean(tf.squared_difference(x, y))
        return sum([mse(x, y, f) for x, y, f in zip(self.srcs, self.outs, self.factors)])

class AAEME(AbsModel):

    def build(self):
        self.meta = tf.nn.l2_normalize(tf.add_n(self.encoders), axis=1)
        self.outs = [tf.layers.Dense(dim)(self.meta) for dim in self.dims]
        return self.srcs, self.ipts

    def loss(self):
        mse = lambda x, y, f: f * tf.reduce_mean(tf.squared_difference(x, y))
        return sum([mse(x, y, f) for x, y, f in zip(self.srcs, self.outs, self.factors)])
