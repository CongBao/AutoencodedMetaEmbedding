# Visualize embeddings
# File: visualize.py
# Author: Cong Bao

import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from utils import Utils

__author__ = 'Cong Bao'

def visualize(input_path, output_path, fig_size=(64, 64)):
    """ Visualize the word embeddings.
        :param input_path: path of word embedding file
        :param output_path: path of output image
        :param fig_size: size of image
    """
    util = Utils()
    emb_dict = util.load_emb(input_path)
    labels = []
    tokens = []
    for word, embed in emb_dict.items():
        labels.append(word)
        tokens.append(embed)
    del emb_dict
    tsne = TSNE(perplexity=40, n_iter=3000, init='pca', verbose=2)
    zipped = tsne.fit_transform(tokens)
    del tokens
    x = []
    y = []
    for embed in zipped:
        x.append(embed[0])
        y.append(embed[1])
    plt.figure(figsize=tuple(fig_size))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    print('Saving figure...')
    plt.savefig(output_path)
    plt.close('all')

def main():
    """ Launch the processing """
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-i', dest='input', type=str, required=True, help='embedding path')
    add_arg('-o', dest='output', type=str, required=True, help='figure saving path')
    add_arg('-s', dest='size', type=int, nargs='+', default=[64, 64], help='the size of figure in inches')
    args = parser.parse_args()
    try:
        visualize(args.input, args.output, args.size)
    except (KeyboardInterrupt, SystemExit):
        print('Abort!')

if __name__ == '__main__':
    main()
