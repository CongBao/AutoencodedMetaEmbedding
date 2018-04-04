# Visualize embeddings

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from utils import Utils

__author__ = 'Cong Bao'

def visualize(file_path):
    util = Utils()
    emb_dict = util.load_emb(file_path)
    labels = []
    tokens = []
    for word, embed in emb_dict.items():
        labels.append(word)
        tokens.append(embed)
    del emb_dict
    tsne = TSNE(perplexity=40, n_iter=2500, init='pca', verbose=2)
    zipped = tsne.fit_transform(tokens)
    del tokens
    x = []
    y = []
    for embed in zipped:
        x.append(embed[0])
        y.append(embed[1])
    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i], xy=(x[i], y[i]), xytest=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-i', dest='input', type=str, required=True, help='embedding path')
    args = parser.parse_args()
    try:
        visualize(args.input)
    except (KeyboardInterrupt, SystemExit):
        print('Abort!')

if __name__ == '__main__':
    main()
