#!/usr/bin/env python
"""
Autoencoding Meta-Embedding with linear model and coupling restraints
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

LEARNING_RATE = 0.01
EPOCHS = 20

logger = Logger(str(os.path.basename(__file__)).replace('.py', ''))

def next_element(data):
    for cbow_item, glove_item in data:
        yield (np.transpose([cbow_item]), np.transpose([glove_item]))

def train_embedding(source_list, output_path, learning_rate=LEARNING_RATE, epoch=EPOCHS):
    # load embedding data
    logger.log('Loading file: %s' % source_list[0])
    cbow_dict = utils.load_embeddings(source_list[0])
    logger.log('Loading file: %s' % source_list[1])
    glove_dict = utils.load_embeddings(source_list[1])

    # find intersection of two sources
    inter_words = set(cbow_dict.keys()) & set(glove_dict.keys())
    logger.log('Number of intersection words: %s' % len(inter_words))
    data = np.asarray([[cbow_dict[i], glove_dict[i]] for i in inter_words])

    # define sources s1, s2
    with tf.name_scope('inputs'):
        s1 = tf.placeholder(tf.float32, [300, 1], name='s1')
        s2 = tf.placeholder(tf.float32, [300, 1], name='s2')

    # define matrix E1, E2, D1 = E1.T, D2 = E2.T
    with tf.name_scope('Encoder1'):
        E1 = tf.Variable(tf.random_normal(shape=[300, 300], stddev=0.01), name='E1')
        tf.summary.histogram('Encoder1', E1)
    with tf.name_scope('Encoder2'):
        E2 = tf.Variable(tf.random_normal(shape=[300, 300], stddev=0.01), name='E2')
        tf.summary.histogram('Encoder2', E2)
    with tf.name_scope('Decoder1'):
        D1 = tf.transpose(E1)
        tf.summary.histogram('Decoder1', D1)
    with tf.name_scope('Decoder2'):
        D2 = tf.transpose(E2)
        tf.summary.histogram('Decoder2', D2)

    # loss = sum((E1*s1-E2*s2)^2+(D1*E1*s1-s1)^2+(D2*E2*s2-s2)^2)
    with tf.name_scope('loss'):
        part1 = tf.squared_difference(tf.matmul(E1, s1), tf.matmul(E2, s2))
        part2 = tf.squared_difference(tf.matmul(D1, tf.matmul(E1, s1)), s1)
        part3 = tf.squared_difference(tf.matmul(D2, tf.matmul(E2, s2)), s2)
        loss = tf.add_n([tf.reduce_sum(part1), tf.reduce_sum(part2), tf.reduce_sum(part3)])
        tf.summary.scalar('loss', loss)

    # minimize loss
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    # compute the encoders and decoders
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./graphs/linear_model_v2', sess.graph)

        sess.run(tf.global_variables_initializer())

        for i in range(epoch):
            total_loss = 0
            for x1, x2 in next_element(data):
                _, acc = sess.run([optimizer, loss], feed_dict={s1: x1, s2: x2})
                total_loss += acc
            logger.log('Epoch {0}: {1}'.format(i, total_loss / len(data)))

            t1, t2 = data[random.randint(0, len(data) - 1)]
            result = sess.run(merged, feed_dict={s1: np.transpose([t1]), s2: np.transpose([t2])})
            writer.add_summary(result, i)

        writer.close()

        E1, E2, D1, D2 = sess.run([E1, E2, D1, D2])

    # calculate the meta embedding
    meta_embedding = {}
    for word in inter_words:
        meta_embedding[word] = np.concatenate([np.matmul(E1, cbow_dict[word].T).T, np.matmul(E2, glove_dict[word].T).T])
    logger.log('Saving data into output file: %s' % output_path)
    utils.save_embeddings(meta_embedding, output_path)
    logger.log('Complete.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input', nargs='+', type=str, required=True, help='the input file(s) containing source vectors')
    parser.add_argument('-o', dest='output', type=str, required=True, help='the output file')
    parser.add_argument('-e', dest='epoch', type=int, help='the number of epoches to train')
    args = parser.parse_args()
    logger.log('Input file(s): %s' % args.input)
    logger.log('Output file: %s' % args.output)
    logger.log('Epoches to train: %s' % (args.epoch if args.epoch else EPOCHS))
    if args.epoch:
        train_embedding(args.input, args.output, epoch=args.epoch)
    else:
        train_embedding(args.input, args.output)

if __name__ == '__main__':
    main()
