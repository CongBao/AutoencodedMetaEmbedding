# K-means Cluster
# File: cluster.py
#Author: Cong Bao

from __future__ import print_function

import argparse

import numpy as np
from sklearn.cluster import KMeans

from utils import Utils

__author__ = 'Cong Bao'

def cluster(input_path, pred_word, n):
    """ Classify data into clusters.
        :param input_path: path of input data
        :param pred_word: word to predict
        :param n: number of clusters
    """
    util = Utils()
    emb_dict = util.load_emb(input_path)
    test = emb_dict.pop(pred_word)
    words = []
    embds = []
    for word, emb in emb_dict.items():
        words.append(word)
        embds.append(emb)
    del emb_dict
    kmeans = KMeans(n_clusters=n).fit(embds)
    print("[Data Labels]")
    for w, l in zip(words, kmeans.labels_):
        print("%s: %d" % (w, l))
    res = kmeans.predict([test])
    print("[Predict Result] %s: %d" % (pred_word, res[0]))

def main():
    """ Lunch the processing """
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-i', dest='input', type=str, required=True, help='embedding path')
    add_arg('-w', dest='word', type=str, required=True, help='word to predict')
    add_arg('-n', dest='num', type=int, default=2, help='number of clusters')
    args = parser.parse_args()
    try:
        cluster(args.input, args.word, args.num)
    except (KeyboardInterrupt, SystemExit):
        print('Abort!')

if __name__ == '__main__':
    main()
