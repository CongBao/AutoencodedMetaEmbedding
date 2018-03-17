# Model of AEME

import numpy as np
from keras import backend as K
from keras import layers, utils
from keras.layers import Activation, Dense, Input, Merge
from keras.models import Model

class AEME(object):

    def __init__(self, **kwargs):
        self.input_dict = kwargs['input'] # {'path':(dim)}
        self.output_path = kwargs['output']
        self.graph_path = kwargs['graph']
        self.checkpoint_path = kwargs['checkpoint']
        self.learning_rate = kwargs['learning_rate']
        self.batch_size = kwargs['batch']
        self.epoch = kwargs['epoch']
        self.activ = kwargs['activ']
        self.factors = kwargs['factors']
        self.noise = kwargs['noise']

class AbstractModel(object):

    def __init__(self, dims, activ):
        self.dims = dims # [dim, ...]
        self.activ = activ
        self.meta = None

    def extract(self):
        return K.eval(self.meta)

    def build(self):
        raise NotImplementedError('Model Undefined')

class DAEME(AbstractModel):

    def build(self):
        srcs = [Input(shape=(dim,)) for dim in self.dims]
        enco = [Dense(min(self.dims), activation=self.activ)(src) for src in srcs]
        self.meta = utils.normalize(layers.concatenate(enco))
        outs = [Dense(dim)(ecd) for dim, ecd in zip(self.dims, enco)]
        return Model(srcs, outs)

class CAEME(AbstractModel):

    def build(self):
        srcs = [Input(shape=(dim,)) for dim in self.dims]
        self.meta = utils.normalize(layers.concatenate([Dense(min(self.dims), activation=self.activ)(src) for src in srcs]))
        outs = [Dense(dim)(self.meta) for dim in self.dims]
        return Model(srcs, outs)

class AAEME(AbstractModel):

    def build(self):
        srcs = [Input(shape=(dim,)) for dim in self.dims]
        self.meta = utils.normalize(layers.add([Dense(min(self.dims), activation=self.activ)(src) for src in srcs]))
        outs = [Dense(dim)(self.meta) for dim in self.dims]
        return Model(srcs, outs)
