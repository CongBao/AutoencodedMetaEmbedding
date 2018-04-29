# Model of AEME
# File: model.py
# Author: Cong Bao

from __future__ import division

from abc import ABCMeta, abstractmethod
from itertools import combinations

import numpy as np
import sklearn.preprocessing as skpre
import tensorflow as tf

from utils import Logger, Utils

__author__ = 'Cong Bao'

class AEME(object):
    """ Autoencoded Meta-Embedding.
        :param input_list: a list of source embedding path
        :param output_path: a string path of output file
        :param log_path: a string path of log file
        :param ckpt_path: a string path of checkpoint file
        :param model_type: the type of model, among DAEME, CAEME, AAEME
        :param dims: a list of dimensionalities of each source embedding
        :param learning_rate: a float number of the learning rate
        :param batch_size: a int number of the batch size
        :param epoch: a int number of the epoch
        :param activ: a string name of activation function
        :param factors: a list of coefficients of each loss part
        :param noise: a float number between 0 and 1 of the masking noise rate
        :param emb_dim: a int number of meta-embedding dimensionality, only used in AAEME model
        :param oov: a boolean value whether to initialize inputs with oov or not
        :param restore: a boolean value whether to restore checkpoint from local file or 
        :property logger: a logger to record log information
        :property utils: a utility tool for I/O
        :property sess: a tensorflow session
        :property ckpt: a tensorflow saver 
    """

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
        """ Load individual source embeddings and store intersection/union words and embeddings in separate lists.
            If oov is True, word list will be the union of individual vocabularies.
            If oov is False, word list will be the intersection of individual vocabularies.
        """
        # a list of source embedding dict {word:embedding}
        src_dict_list = [self.utils.load_emb(path) for path in self.input_list]
        # if consider out-of-vocabulary words, initialize embeddings of them with zero vectors
        # the final vocabulary is the union of source vocabularies
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
        # if do not consider out-of-vocabulary, just load the embeddings from dict
        # the final vocabulary is the intersection of source vocabularies 
        else:
            self.inter_words = list(set.intersection(*[set(src_dict.keys()) for src_dict in src_dict_list]))
            self.logger.log('Intersection Words: %s' % len(self.inter_words))
            self.sources = np.asarray(list(zip(*[skpre.normalize([src_dict[word] for word in self.inter_words]) for src_dict in src_dict_list])))
        # delete the list of dicts to release memory
        del src_dict_list

    def build_model(self):
        """ Build the model of AEME.
            The model to build will be one of DAEME, CAEME, or AAEME.
        """
        # initialize sources and inputs
        self.srcs = [tf.placeholder(tf.float32, (None, dim)) for dim in self.dims]
        self.ipts = [tf.placeholder(tf.float32, (None, dim)) for dim in self.dims]
        # select and build models
        params = [self.dims, self.activ, self.factors]
        if self.model_type == 'DAEME':
            self.aeme = DAEME(*params)
        elif self.model_type == 'CAEME':
            self.aeme = CAEME(*params)
        elif self.model_type == 'AAEME':
            self.aeme = AAEME(*params, emb_dim=self.emb_dim)
        self.aeme.build(self.srcs, self.ipts)

    def train_model(self):
        """ Train the model.
            Variables with least losses will be stored in checkpoint file.
        """
        step = tf.Variable(0, trainable=False)
        rate = tf.train.exponential_decay(self.learning_rate, step, 50, 0.99)
        loss = self.aeme.loss()
        opti = tf.train.AdamOptimizer(rate).minimize(loss, global_step=step)
        self.ckpt = tf.train.Saver(tf.global_variables())
        if self.restore:
            self.ckpt.restore(self.sess, self.ckpt_path)
        else:
            self.sess.run(tf.global_variables_initializer())
        size = len(self.sources) // self.batch_size # the number of batches
        best = float('inf')
        # loop for N epoches
        for itr in range(self.epoch):
            indexes = np.random.permutation(len(self.sources)) # shuffle training inputs
            train_loss = 0.
            # train with mini-batches
            for idx in range(size):
                batch_idx = indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
                batches = list(zip(*self.sources[batch_idx]))
                feed = {k:v for k, v in zip(self.srcs, batches)}
                feed.update({k:self._corrupt(v) for k, v in zip(self.ipts, batches)})
                _, batch_loss = self.sess.run([opti, loss], feed)
                train_loss += batch_loss
            epoch_loss = train_loss / size
            # save the checkpoint with least loss
            if epoch_loss <= best:
                self.ckpt.save(self.sess, self.ckpt_path)
                best = epoch_loss
            self.logger.log('[Epoch{0}] loss: {1}'.format(itr, epoch_loss))

    def generate_meta_embed(self):
        """ Generate meta-embedding and save as local file.
            Variables used to predict are these with least losses during training.
        """
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
        """ Corrupt a batch using masking noises.
            :param batch: the batch to be corrupted
            :return: a new batch after corrupting
        """
        noised = np.copy(batch)
        batch_size, feature_size = np.shape(batch)
        for i in range(batch_size):
            mask = np.random.randint(0, feature_size, int(feature_size * self.noise))
            for m in mask:
                noised[i][m] = 0.
        return noised

class AbsModel(object):
    """ Base class of all proposed methods.
        :param dims: a list of dimensionalities of each input
        :param activ: the string name of activation function
        :param factors: a list of coefficients of each loss part
    """

    __metaclass__ = ABCMeta

    def __init__(self, dims, activ, factors):
        self.dims = dims
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
        """ Mean Squared Error with slicing.
            This method will slice vector with higher dimension to the lower one,
            if the two vector have different dimensions.
            :param x: first vector
            :param y: second vector
            :param f: coefficient
            :return: a tensor after calculating f * (1 / d) * ||x - y||^2
        """
        x_d = x.get_shape().as_list()[1]
        y_d = y.get_shape().as_list()[1]
        if x_d != y_d:
            smaller = min(x_d, y_d)
            x = tf.slice(x, [0, 0], [tf.shape(x)[0], smaller])
            y = tf.slice(y, [0, 0], [tf.shape(y)[0], smaller])
        return tf.scalar_mul(f, tf.reduce_mean(tf.squared_difference(x, y)))

    def extract(self):
        """ Extract the meta-embeddding model.
            :return: the meta-embedding model
        """
        return self.meta

    @abstractmethod
    def build(self, srcs, ipts):
        """ Abstract method.
            Build the model.
            :param srcs: source embeddings
            :param ipts: input embeddings
        """
        self.srcs = srcs
        self.ipts = ipts

    @abstractmethod
    def loss(self):
        """ Abstract method.
            Obtain the loss function of model.
            :return: a tensor calculating the loss function
        """
        pass

class DAEME(AbsModel):
    """ Decoupled Autoencoded Meta-Embedding.
        This method calculate meta-embedding as the concatenation of encoded source embeddings.
        The loss function is defined as the sum of mse of each autoencoder and the mse between meta parts.
    """

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
    """ Concatenated Autoencoded Meta-Embedding.
        This method calculate meta-embedding as the concatenation of encoded source embeddings.
        The loss function is defined as the sum of mse of each autoencoder.
    """

    def build(self, srcs, ipts):
        AbsModel.build(self, srcs, ipts)
        self.encoders = [tf.layers.dense(ipt, dim, self.activ) for ipt, dim in zip(self.ipts, self.dims)]
        self.meta = tf.nn.l2_normalize(tf.concat(self.encoders, 1), 1)
        self.outs = [tf.layers.dense(self.meta, dim) for dim in self.dims]

    def loss(self):
        return tf.add_n([self.mse(x, y, f) for x, y, f in zip(self.srcs, self.outs, self.factors)])

class AAEME(AbsModel):
    """ Averaged Autoencoded Meta-Embedding.
        This method calculate meta-embedding as the averaging of encoded source embeddings.
        The loss function is defined as the sum of mse of each autoencoder.
        :param emb_dim: the dimensionality of meta-embedding
    """

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
