# Pick out intersection of source embeddings
# File: pickout.py
# Author: Cong Bao

from __future__ import print_function

import argparse

import numpy as np
import sklearn.preprocessing as skpre

from utils import Utils

__author__ = 'Cong Bao'

def pickout_intersection(inputs, outputs, norm=False):
    """ Pick out intersection of source embeddings.
        :param inputs: path of source embeddings
        :param outputs: path of output files
        :param norm: whether to perform normalization on inputs or not
    """
    utils = Utils()
    src_dict_list = [utils.load_emb(path) for path in inputs]
    inter_words = list(set.intersection(*[set(src_dict.keys()) for src_dict in src_dict_list]))
    print('Intersection Words: %s' % len(inter_words))
    if norm:
        normalize = lambda x: skpre.normalize(np.reshape(x, (1, -1)))[0]
    else:
        normalize = lambda x: x
    for i, path in enumerate(outputs):
        selected = {}
        for word in inter_words:
            selected[word] = normalize(src_dict_list[i][word])
        utils.save_emb(selected, path)

def pickout_words(inputs, outputs, word_path, norm=False):
    """ Pick out embeddings with given words.
        :param inputs: path of source embeddings
        :param outputs: path of output files
        :param word_path: path of word list file
        :param norm: whether to perform nomalization on inputs or not
    """
    utils = Utils()
    src_dict_list = [utils.load_emb(path) for path in inputs]
    word_list = utils.load_words(word_path)
    if norm:
        normalize = lambda x: skpre.normalize(np.reshape(x, (1, -1)))[0]
    else:
        normalize = lambda x: x
    for i, path in enumerate(outputs):
        selected = {}
        for word in word_list:
            embed = src_dict_list[i].get(word)
            if embed is not None:
                selected[word] = normalize(embed)
        utils.save_emb(selected, path)

def main():
    """ Launch the processing """
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-i', dest='input',  nargs='+', type=str, required=True, help='input files')
    add_arg('-o', dest='output', nargs='+', type=str, required=True, help='output files')
    add_arg('-w', dest='words',  type=str,  default=None,            help='word list file')
    add_arg('-N', dest='norm',   action='store_true',                help='perform normalization')
    args = parser.parse_args()
    assert len(args.input) == len(args.output)
    print('Input files: %s'  % args.input)
    print('Output files: %s' % args.output)
    if args.words:
        print('Word list file: %s' % args.words)
        pickout_words(args.input, args.output, args.words, args.norm)
    else:
        pickout_intersection(args.input, args.output, args.norm)
    print('Complete!')

if __name__ == '__main__':
    main()
