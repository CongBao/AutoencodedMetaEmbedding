#!/usr/bin/env python
"""
Denoising Autoencoding Meta-Embedding with non-linear deep nn model
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

def corrupt_input(input_batch, ratio=0.5):
    noisy_batch = np.copy(input_batch)
    batch_size = input_batch.shape[0]
    feature_size = input_batch.shape[1]
    for i in range(batch_size):
        mask = np.random.randint(0, feature_size, int(feature_size * ratio))
        for m in mask:
            noisy_batch[i][m] = 0.
    return noisy_batch

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

def add_layer(layer, shape, name=None):
    with tf.name_scope(name):
        weight = tf.Variable(tf.random_normal(shape=shape, stddev=0.01), name='w_' + name)
        bias = tf.Variable(tf.zeros(shape=(1, shape[1])), name='b_' + name)
        tf.summary.histogram('w_' + name, weight)
        tf.summary.histogram('b_' + name, bias)
    return tf.nn.sigmoid(tf.matmul(layer, weight) + bias) - 0.5

def train_embedding(source_list, output_path, learning_rate, batch_size, epoch, noise_rate):
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
        s1_noisy = tf.placeholder(tf.float32, (None, 300), name='s1_noisy')
        s2_noisy = tf.placeholder(tf.float32, (None, 300), name='s2_noisy')
        s1_true = tf.placeholder(tf.float32, (None, 300), name='s1_true')
        s2_true = tf.placeholder(tf.float32, (None, 300), name='s2_true')

    # define E1, E2, D1, D2
    E1_1 = add_layer(s1_noisy, (300, 225), 'Encoder1_layer1')
    E1_2 = add_layer(E1_1, (225, 150), 'Encoder1_layer2')
    D1_3 = add_layer(E1_2, (150, 225), 'Decoder1_layer3')
    D1_4 = add_layer(D1_3, (225, 300), 'Decoder1_layer4')
    E2_1 = add_layer(s2_noisy, (300, 225), 'Encoder2_layer1')
    E2_2 = add_layer(E2_1, (225, 150), 'Encoder2_layer2')
    D2_3 = add_layer(E2_2, (150, 225), 'Decoder2_layer3')
    D2_4 = add_layer(D2_3, (225, 300), 'Decoder2_layer4')

    # loss = sum((E1*s1-E2*s2)^2+(D1*E1*s1-s1)^2+(D2*E2*s2-s2)^2)
    with tf.name_scope('loss'):
        part1 = tf.squared_difference(E1_2, E2_2)
        part2 = tf.squared_difference(D1_4, s1_true)
        part3 = tf.squared_difference(D2_4, s2_true)
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
        writer = tf.summary.FileWriter('./graphs/nonlinear_nn_v4', sess.graph)

        sess.run(tf.global_variables_initializer())

        for i in range(epoch):
            total_loss = 0
            for s1_batch, s2_batch in next_batch(data, batch_size):
                s1_noised = corrupt_input(s1_batch, noise_rate)
                s2_noised = corrupt_input(s2_batch, noise_rate)
                _, batch_loss = sess.run([optimizer, loss], feed_dict={s1_noisy: s1_noised,
                                                                       s2_noisy: s2_noised,
                                                                       s1_true: s1_batch,
                                                                       s2_true: s2_batch})
                total_loss += batch_loss
            logger.log('Epoch {0}: {1}'.format(i, total_loss / batch_size))

            if i % 100 == 0:
                cbow_test = []
                glove_test = []
                for cbow_item, glove_item in random.sample(data, batch_size):
                    cbow_test.append(cbow_item)
                    glove_test.append(glove_item)
                result = sess.run(merged, feed_dict={s1_noisy: np.asarray(cbow_test),
                                                     s2_noisy: np.asarray(glove_test),
                                                     s1_true: np.asarray(cbow_test),
                                                     s2_true: np.asarray(glove_test)})
                writer.add_summary(result, i)

        writer.close()

        # calculate the meta embedding
        meta_embedding = {}
        for word in inter_words:
            s1_cbow = cbow_dict[word].reshape((1, 300))
            s2_glove = glove_dict[word].reshape((1, 300))
            s1_embed, s2_embed = sess.run([E1_2, E2_2], feed_dict={s1_noisy: s1_cbow,
                                                                   s2_noisy: s2_glove})
            meta_embedding[word] = np.concatenate([s1_embed.reshape((150)), s2_embed.reshape((150))])

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
    parser.add_argument('--noise_ratio', dest='noise', type=float, default=0.5, help='the ratio of noise to the input')
    args = parser.parse_args()
    logger.log('Input file(s): %s' % args.input)
    logger.log('Output file: %s' % args.output)
    logger.log('Learning rate: %s' % args.rate)
    logger.log('Batch size: %s' % args.batch)
    logger.log('Epoches to train: %s' % args.epoch)
    logger.log('Ratio of noise: %s' % args.noise)
    train_embedding(args.input, args.output, args.rate, args.batch, args.epoch, args.noise)

if __name__ == '__main__':
    main()
