"""
Autoencoding Meta-Embedding with linear model and coupling restraints
"""

import random
import numpy as np
import tensorflow as tf

import utils

CBOW_PATH = r'E:\Dropbox\Data\small_cbow.txt'
GLOVE_PATH = r'E:\Dropbox\Data\small_glove.txt'
RESULT_PATH = r'.\linear_restraint_model_result.txt'

LEARNING_RATE = 0.01
EPOCHS = 20

def next_element(data):
    for cbow_item, glove_item in data:
        yield [np.transpose([cbow_item]), np.transpose([glove_item])]

def main():
    # load embedding data
    cbow_dict = utils.load_embeddings(CBOW_PATH)
    glove_dict = utils.load_embeddings(GLOVE_PATH)

    # find intersection of two sources
    inter_words = set(cbow_dict.keys()) & set(glove_dict.keys())
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
        loss = tf.reduce_sum(tf.square(tf.matmul(E1, s1) - tf.matmul(E2, s2)) + tf.square(tf.matmul(D1, tf.matmul(E1, s1)) - s1) + tf.square(tf.matmul(D2, tf.matmul(E2, s2)) - s2), name='loss')
        tf.summary.scalar('loss', loss)

    # minimize loss
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    # compute the encoders and decoders
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./graphs/linear_model', sess.graph)

        sess.run(tf.global_variables_initializer())

        for i in range(EPOCHS):
            total_loss = 0
            for x1, x2 in next_element(data):
                _, acc = sess.run([optimizer, loss], feed_dict={s1: x1, s2: x2})
                total_loss += acc
            print('Epoch {0}: {1}'.format(i, total_loss / len(data)))

            t1, t2 = data[random.randint(0, len(data) - 1)]
            result = sess.run(merged, feed_dict={s1: np.transpose([t1]), s2: np.transpose([t2])})
            writer.add_summary(result, i)

        writer.close()

        E1, E2, D1, D2 = sess.run([E1, E2, D1, D2])

    # calculate the meta embedding
    meta_embedding = {}
    for word in inter_words:
        meta_embedding[word] = np.concatenate([np.matmul(E1, cbow_dict[word].T).T, np.matmul(E2, glove_dict[word].T).T])
    utils.save_embeddings(meta_embedding, RESULT_PATH)

if __name__ == '__main__':
    main()