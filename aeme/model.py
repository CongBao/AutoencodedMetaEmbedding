# Model of AEME

import numpy as np
from keras import backend as K
from keras import layers, utils
from keras.callbacks import LearningRateScheduler
from keras.layers import Dense, Dropout, Input, Merge
from keras.models import Model
from keras.optimizers import Adam

from utils import load_emb

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

        self.src_dict_list = []
        self.inter_words = []
        self.sources = []

        self.model = None

    def load_data(self):
        self.src_dict_list = [load_emb(path) for path in self.input_list]
        self.inter_words = sorted(list(set.intersection(*[set(src_dict.keys()) for src_dict in self.src_dict_list])))
        self.sources = [utils.normalize([src_dict[word] for word in self.inter_words]) for src_dict in self.src_dict_list]

    def build_model(self):
        params = [self.dims, self.activ, self.noise, self.factors]
        _model = AbsModel(*params)
        if self.model_type == 'DAEME':
            _model = DAEME(*params)
        elif self.model_type == 'CAEME':
            _model = CAEME(*params)
        elif self.model_type == 'AAEME':
            _model = AAEME(*params)
        self.model = _model.build()
        self.model.add_loss(_model.loss())

    def train_model(self):
        self.model.compile(optimizer=Adam(lr=self.learning_rate))
        self.model.summary()
        self.model.fit(self.sources, self.sources,
                       batch_size=self.batch_size,
                       epochs=self.epoch,
                       callbacks=LearningRateScheduler(lambda e: self.learning_rate * 0.999 ** (e / 50)))

    def generate_meta_embed(self):
        pass

class AbsModel(object):

    def __init__(self, dims, activ, noise, factors):
        self.dims = dims # [dim, ...]
        self.activ = activ
        self.noise = noise
        self.factors = factors

        self.meta = None
        self.outs = None
        self.srcs = [Input(shape=(dim,)) for dim in self.dims]
        self.encoders = [Dense(min(self.dims), activation=self.activ)(Dropout(self.noise)(src)) for src in self.srcs]

    def extract(self):
        return Model(self.srcs, self.meta)

    def build(self):
        raise NotImplementedError('Model Undefined')

    def loss(self):
        raise NotImplementedError('Loss Function Undefined')

class DAEME(AbsModel):

    def build(self):
        self.meta = utils.normalize(layers.concatenate(self.encoders))
        self.outs = [Dense(dim)(encoder) for dim, encoder in zip(self.dims, self.encoders)]
        return Model(self.srcs, self.outs)

    def loss(self):
        mse = lambda x, y, f: f * K.mean(K.square(y - x), axis=-1)
        ael = sum([mse(x, y, f) for x, y, f in zip(self.srcs, self.outs, self.factors[:-1])])
        mtl = 0.
        for i in range(len(self.encoders)):
            for j in range(i + 1, len(self.encoders)):
                mtl += mse(self.encoders[i], self.encoders[j], self.factors[-1])
        return ael + mtl

class CAEME(AbsModel):

    def build(self):
        self.meta = utils.normalize(layers.concatenate(self.encoders))
        self.outs = [Dense(dim)(self.meta) for dim in self.dims]
        return Model(self.srcs, self.outs)

    def loss(self):
        mse = lambda x, y, f: f * K.mean(K.square(y - x), axis=-1)
        return sum([mse(x, y, f) for x, y, f in zip(self.srcs, self.outs, self.factors)])

class AAEME(AbsModel):

    def build(self):
        self.meta = utils.normalize(layers.add(self.encoders))
        self.outs = [Dense(dim)(self.meta) for dim in self.dims]
        return Model(self.srcs, self.outs)

    def loss(self):
        mse = lambda x, y, f: f * K.mean(K.square(y - x), axis=-1)
        return sum([mse(x, y, f) for x, y, f in zip(self.srcs, self.outs, self.factors)])
