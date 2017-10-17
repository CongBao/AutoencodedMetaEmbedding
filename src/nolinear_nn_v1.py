#!/usr/bin/env python
"""
Autoencoding Meta-Embedding with non-linear nn model
"""

from __future__ import division

import argparse
import os
import random

import numpy as np
import tensorflow as tf

import utils
from logger import Logger

__author__ = 'Cong Bao'

LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 500
LAYERS = 3

logger = Logger(str(os.path.basename(__file__)).replace('.py', ''))

def next_batch(data, batch_size):
    if batch_size == 1:
        for cbow_item, glove_item in data:
            yield (np.transpose([cbow_item]), np.transpose([glove_item]))
    elif batch_size == len(data):
        cbow_batch = []
        glove_batch = []
        for cbow_item, glove_item in data:
            cbow_batch.append(cbow_item)
            glove_batch.append(glove_item)
        yield (np.transpose(cbow_batch), np.transpose(glove_batch))
    else:
        cbow_batch = []
        glove_batch = []
        for cbow_item, glove_item in data:
            cbow_batch.append(cbow_item)
            glove_batch.append(glove_item)
            if len(cbow_batch) == batch_size:
                yield (np.transpose(cbow_batch), np.transpose(glove_batch))
                cbow_batch = []
                glove_batch = []
        for cbow_item, glove_item in random.sample(data, batch_size - len(cbow_batch)):
            cbow_batch.append(cbow_item)
            glove_batch.append(glove_item)
        yield (np.transpose(cbow_batch), np.transpose(glove_batch))

def add_layer(inputs, in_size, out_size, activation_func=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]))
    connection = tf.matmul(inputs, weights) + biases
    if activation_func is None:
        return connection
    else:
        return activation_func(connection)

def build_layers(layers):
    pass

def train_embedding(source_list, output_path, learning_rate, batch_size, epoch, layers):
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input', nargs='+', type=str, required=True, help='the input file(s) containing source vectors')
    parser.add_argument('-o', dest='output', type=str, required=True, help='the output file')
    parser.add_argument('-r', dest='rate', type=float, default=LEARNING_RATE, help='the learning rate of gradient descent')
    parser.add_argument('-b', dest='batch', type=int, default=BATCH_SIZE, help='the size of batches')
    parser.add_argument('-e', dest='epoch', type=int, default=EPOCHS, help='the number of epoches to train')
    parser.add_argument('-l', dest='layer', type=int, default=LAYERS, help='Number of layers')
    args = parser.parse_args()
    logger.log('Input file(s): %s' % args.input)
    logger.log('Output file: %s' % args.output)
    logger.log('Learning rate: %s' % args.rate)
    logger.log('Batch size: %s' % args.batch)
    logger.log('Epoches to train: %s' % args.epoch)
    logger.log('Layers: %s' % args.layer)
    train_embedding(args.input, args.output, args.rate, args.batch, args.epoch, args.layer)

if __name__ == '__main__':
    main()