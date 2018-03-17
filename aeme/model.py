# Model of AEME

import numpy as np
from keras import backend as K
from keras import layers, utils
from keras.layers import Dense, Input, Merge
from keras.models import Model

from utils import load_emb

__author__ = 'Cong Bao'

class AEME(object):

    def __init__(self, **kwargs):
        self.input_list = kwargs['input'] # [path, ...]
        self.output_path = kwargs['output']
        self.graph_path = kwargs['graph']
        self.checkpoint_path = kwargs['checkpoint']
        self.learning_rate = kwargs['learning_rate']
        self.batch_size = kwargs['batch']
        self.epoch = kwargs['epoch']
        self.activ = kwargs['activ']
        self.factors = kwargs['factors']
        self.noise = kwargs['noise']

    def load_data(self):
        src_dict_list = [load_emb(path) for path in self.input_list]
        inter_words = set.intersection(*[set(src_dict.keys()) for src_dict in src_dict_list])
        inter_words = sorted(list(inter_words))
        sources = [utils.normalize([src_dict[word] for word in inter_words]) for src_dict in src_dict_list]

class AbsModel(object):

    def __init__(self, dims, activ):
        self.dims = dims # [dim, ...]
        self.activ = activ
        self.meta = None

    def extract(self):
        return K.eval(self.meta)

    def build(self):
        raise NotImplementedError('Model Undefined')

class DAEME(AbsModel):

    def build(self):
        srcs = [Input(shape=(dim,)) for dim in self.dims]
        enco = [Dense(min(self.dims), activation=self.activ)(src) for src in srcs]
        self.meta = utils.normalize(layers.concatenate(enco))
        outs = [Dense(dim)(ecd) for dim, ecd in zip(self.dims, enco)]
        return Model(srcs, outs)

class CAEME(AbsModel):

    def build(self):
        srcs = [Input(shape=(dim,)) for dim in self.dims]
        self.meta = utils.normalize(layers.concatenate([Dense(min(self.dims), activation=self.activ)(src) for src in srcs]))
        outs = [Dense(dim)(self.meta) for dim in self.dims]
        return Model(srcs, outs)

class AAEME(AbsModel):

    def build(self):
        srcs = [Input(shape=(dim,)) for dim in self.dims]
        self.meta = utils.normalize(layers.add([Dense(min(self.dims), activation=self.activ)(src) for src in srcs]))
        outs = [Dense(dim)(self.meta) for dim in self.dims]
        return Model(srcs, outs)
