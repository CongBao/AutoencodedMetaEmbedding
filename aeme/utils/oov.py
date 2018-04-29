# pre-processing to predict out-of-vocabulary words
# File: oov.py
# Author: Cong Bao

import argparse
import os
from itertools import combinations as comb

import numpy as np
from keras.layers import Dense, Input
from keras.losses import mse
from keras.models import Model
from keras.optimizers import Adam
from sklearn.preprocessing import normalize

from utils import Utils

__author__ = 'Cong Bao'

class Regressor(object):
    """ A simple regressive model.
        :param in_size: the size of input
        :param out_size: the size of output
        :param activ_func: activation function, default tanh
        :param batch_size: size of each mini-batch, default 128
        :param learning_rate: the learning_rate during training, default 0.001
        :param epoch: the number of epoches to train, default 20
    """
    
    def __init__(self,
                 in_size,
                 out_size,
                 activ_func='tanh',
                 batch_size=128,
                 learning_rate=0.001,
                 epoch=20):
        self.in_size = in_size
        self.out_size = out_size
        self.activ_func = activ_func
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epoch = epoch

    def build(self):
        """ Build the model. """
        src = Input(shape=(self.in_size,))
        out = Dense(self.out_size, activation=self.activ_func)(src)
        self.model = Model(src, out)
        self.model.compile(Adam(lr=self.learning_rate), loss=mse)
    
    def train(self, x, y):
        """ Train the model.
            :param x: the input data
            :param y: the labelled output
        """
        self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epoch)

    def predict(self, x):
        """ Predict output with given data.
            :param x: the data given to predict
            :return: predicted output
        """
        return self.model.predict(x, batch_size=self.batch_size)

def preprocess(src_dict):
    """ Preprocess data before training.
        :param src_dict: the source dict {word:embedding}
        :return: a new dict after preprocessing
    """
    word_list = list(src_dict.keys())
    arr = []
    for word in word_list:
        arr.append(src_dict[word])
    arr = normalize(arr)
    new_dict = {}
    for i, word in enumerate(word_list):
        new_dict[word] = arr[i]
    return new_dict

def process(inputs, outputs, dims, mode='max'):
    """ Process the oovs.
        :param inputs: a list of paths of inputs
        :param outputs: a list of paths of outputs
        :param dims: a list of dimensionalities of inputs
        :param mode: the mode of processing, if max, only iterate once, if all, iterate the length of inputs times
    """
    util = Utils()
    src_dicts = [preprocess(util.load_emb(path)) for path in inputs]
    new_dicts = [{}] * len(inputs)
    indexes = list(range(len(inputs)))
    for k in range(len(inputs), 1, -1):
        if mode == 'max' and k < len(inputs):
            break
        for group in comb(indexes, k):
            print('Group: %s' % (group,))
            for teachers in comb(group, k - 1):
                print('Teachers: %s' % (teachers,))
                student = (set(group) - set(teachers)).pop()
                print('Student: %s' % student)
                tech_CK = set.intersection(*[set(src_dicts[idx].keys()) for idx in teachers]) # teachers' common knowledge
                stu_CK = set(src_dicts[student].keys()) & tech_CK # student's common knowledge with teachers
                stu_NK = tech_CK - stu_CK # what student don't know, but teachers know
                if not stu_NK:
                    continue
                stu_CK = list(stu_CK)
                stu_NK = list(stu_NK)
                taught = []
                for tc in teachers:
                    print('Teacher %s teaching...' % tc)
                    reg = Regressor(dims[tc], dims[student])
                    reg.build()
                    x = []
                    y = []
                    for word in stu_CK:
                        x.append(src_dicts[tc][word])
                        y.append(src_dicts[student][word])
                    reg.train(np.asarray(x), np.asarray(y))
                    del x, y
                    t = []
                    for word in stu_NK:
                        t.append(src_dicts[tc][word])
                    taught.append(reg.predict(np.asarray(t)))
                    del t
                learned = []
                for new_know in zip(*taught):
                    learned.append(np.sum(new_know, axis=0))
                del taught
                learned = normalize(learned)
                for word, know in zip(stu_NK, learned):
                    new_dicts[student][word] = know
                del learned
    for idx in indexes:
        src_dicts[idx].update(new_dicts[idx])
    for i, path in enumerate(outputs):
        util.save_emb(src_dicts[i], path)

def main():
    """ Launch the processing. """
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-i', dest='input', type=str, required=True, nargs='+', help='inputs directories')
    add_arg('-o', dest='output', type=str, required=True, nargs='+', help='output directories')
    add_arg('-d', dest='dim', type=int, required=True, nargs='+', help='dimensionality of each input')
    add_arg('-m', dest='mode', type=str, default='max', help='mode of oov prediction, max or all')
    add_arg('--cpu-only', dest='cpu', action='store_true', help='whether use cpu only or not')
    args = parser.parse_args()
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    assert len(args.input) == len(args.output)
    assert len(args.input) == len(args.dim)
    print('Inputs: %s' % args.input)
    print('Outputs: %s' % args.output)
    print('Dimensions: %s' % args.dim)
    print('Mode: %s' % args.mode)
    print('Running on %s' % ('CPU' if args.cpu else 'GPU'))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    try:
        process(args.input, args.output, args.dim, args.mode)
    except (KeyboardInterrupt, SystemExit):
        print('Abort!')

if __name__ == '__main__':
    main()
