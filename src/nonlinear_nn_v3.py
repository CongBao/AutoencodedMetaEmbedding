#!/usr/bin/env python
"""
Autoencoding Meta-Embedding with non-linear nn model and dimension restraints
"""

from __future__ import division

import argparse
import os
import random

import numpy as np
import tensorflow as tf
from scipy.special import expit

import utils
from logger import Logger

__author__ = 'Cong Bao'

LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 1000

logger = Logger(str(os.path.basename(__file__)).replace('.py', ''))

def next_batch(data, batch_size):
    if batch_size == 1:
        for cbow_item, glove_item in data:
            yield (np.asarray([cbow_item]), np.asarray([glove_item]))
    elif batch_size == len(data):
        cbow_batch = []
        glove_batch = []
        for cbow_item, glove_item in data:
            cbow_batch.append(cbow_item)
            glove_batch.append(glove_item)
        yield (np.asarray(cbow_batch), np.asarray(glove_batch))
    else:
        cbow_batch = []
        glove_batch = []
        for cbow_item, glove_item in data:
            cbow_batch.append(cbow_item)
            glove_batch.append(glove_item)
            if len(cbow_batch) == batch_size:
                yield (np.asarray(cbow_batch), np.asarray(glove_batch))
                cbow_batch = []
                glove_batch = []
        for cbow_item, glove_item in random.sample(data, batch_size - len(cbow_batch)):
            cbow_batch.append(cbow_item)
            glove_batch.append(glove_item)
        yield (np.asarray(cbow_batch), np.asarray(glove_batch))

def train_embedding(source_list, output_path, learning_rate, batch_size, epoch):
    # load embedding data
    logger.log('Loading file: %s' % source_list[0])
    cbow_dict = utils.load_embeddings(source_list[0])
    logger.log('Loading file: %s' % source_list[1])
    glove_dict = utils.load_embeddings(source_list[1])

    # find intersection of two sources
    inter_words = set(cbow_dict.keys()) & set(glove_dict.keys())
    logger.log('Number of intersection words: %s' % len(inter_words))
    data = [[cbow_dict[i], glove_dict[i]] for i in inter_words]

    # define sources s1, s2
    with tf.name_scope('inputs'):
        s1 = tf.placeholder(tf.float32, [batch_size, 300], name='s1')
        s2 = tf.placeholder(tf.float32, [batch_size, 300], name='s2')

    # define E1, E2, D1, D2
    with tf.name_scope('Encoder1'):
        w_E1 = tf.Variable(tf.random_normal(shape=[300, 150], stddev=0.01), name='w_E1')
        b_E1 = tf.Variable(tf.zeros([1, 150]), name='b_E1')
        E1 = tf.nn.sigmoid(tf.matmul(s1, w_E1) + b_E1) - 0.5
        tf.summary.histogram('w_E1', w_E1)
        tf.summary.histogram('b_E1', b_E1)
    with tf.name_scope('Encoder2'):
        w_E2 = tf.Variable(tf.random_normal(shape=[300, 150], stddev=0.01), name='w_E2')
        b_E2 = tf.Variable(tf.zeros([1, 150]), name='b_E2')
        E2 = tf.nn.sigmoid(tf.matmul(s2, w_E2) + b_E2) - 0.5
        tf.summary.histogram('w_E2', w_E2)
        tf.summary.histogram('b_E2', b_E2)
    with tf.name_scope('Decoder1'):
        w_D1 = tf.Variable(tf.random_normal(shape=[150, 300], stddev=0.01), name='w_D1')
        b_D1 = tf.Variable(tf.zeros([1, 300]), name='b_D1')
        D1 = tf.matmul(E1, w_D1) + b_D1
        tf.summary.histogram('w_D1', w_D1)
        tf.summary.histogram('b_D1', b_D1)
    with tf.name_scope('Decoder2'):
        w_D2 = tf.Variable(tf.random_normal(shape=[150, 300], stddev=0.01), name='w_D2')
        b_D2 = tf.Variable(tf.zeros([1, 300]), name='b_D2')
        D2 = tf.matmul(E2, w_D2) + b_D2
        tf.summary.histogram('w_D2', w_D2)
        tf.summary.histogram('b_D2', b_D2)

    # loss = sum((E1*s1-E2*s2)^2+(D1*E1*s1-s1)^2+(D2*E2*s2-s2)^2)
    with tf.name_scope('loss'):
        part1 = tf.squared_difference(E1, E2)
        part2 = tf.squared_difference(D1, s1)
        part3 = tf.squared_difference(D2, s2)
        loss = tf.reduce_sum(part1) + tf.reduce_sum(part2) + tf.reduce_sum(part3)
        tf.summary.scalar('loss', loss)

    # minimize loss
    with tf.name_scope('train'):
        step = tf.Variable(0, trainable=False)
        rate = tf.train.exponential_decay(learning_rate, step, 50, 0.999)
        optimizer = tf.train.AdamOptimizer(rate).minimize(loss, global_step=step)
        #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./graphs/nonlinear_nn_v3', sess.graph)

        sess.run(tf.global_variables_initializer())

        for i in range(epoch):
            total_loss = 0
            for s1_batch, s2_batch in next_batch(data, batch_size):
                _, batch_loss = sess.run([optimizer, loss], feed_dict={s1: s1_batch, s2: s2_batch})
                total_loss += batch_loss
            logger.log('Epoch {0}: {1}'.format(i, total_loss / batch_size))

            if i % 100 == 0:
                cbow_test = []
                glove_test = []
                for cbow_item, glove_item in random.sample(data, batch_size):
                    cbow_test.append(cbow_item)
                    glove_test.append(glove_item)
                result = sess.run(merged, feed_dict={s1: np.asarray(cbow_test), s2: np.asarray(glove_test)})
                writer.add_summary(result, i)

        writer.close()

        w_E1, w_E2, w_D1, w_D2 = sess.run([w_E1, w_E2, w_D1, w_D2])
        b_E1, b_E2, b_D1, b_D2 = sess.run([b_E1, b_E2, b_D1, b_D2])

    # calculate the meta embedding
    meta_embedding = {}
    for word in inter_words:
        embed_cbow = (expit(np.dot(cbow_dict[word].reshape((1, 300)), w_E1) + b_E1) - 0.5).reshape((150))
        embed_glove = (expit(np.dot(glove_dict[word].reshape((1, 300)), w_E2) + b_E2) - 0.5).reshape((150))
        meta_embedding[word] = np.concatenate([embed_cbow, embed_glove])
    logger.log('Saving data into output file: %s' % output_path)
    utils.save_embeddings(meta_embedding, output_path)
    logger.log('Complete.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input', nargs='+', type=str, required=True, help='the input file(s) containing source vectors')
    parser.add_argument('-o', dest='output', type=str, required=True, help='the output file')
    parser.add_argument('-r', dest='rate', type=float, default=LEARNING_RATE, help='the learning rate of gradient descent')
    parser.add_argument('-b', dest='batch', type=int, default=BATCH_SIZE, help='the size of batches')
    parser.add_argument('-e', dest='epoch', type=int, default=EPOCHS, help='the number of epoches to train')
    parser.add_argument('--cpu-only', dest='cpu', action='store_true', help='if use cpu only')
    args = parser.parse_args()
    logger.log('Input file(s): %s' % args.input)
    logger.log('Output file: %s' % args.output)
    logger.log('Learning rate: %s' % args.rate)
    logger.log('Batch size: %s' % args.batch)
    logger.log('Epoches to train: %s' % args.epoch)
    logger.log('Running on %s' % ('CPU' if args.cpu else 'GPU'))
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    train_embedding(args.input, args.output, args.rate, args.batch, args.epoch)

if __name__ == '__main__':
    main()
