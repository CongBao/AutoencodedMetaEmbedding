# pick out intersection of source embedding

from __future__ import print_function

import argparse

import numpy as np
import sklearn.preprocessing as skpre

from utils import Utils

__author__ = 'Cong Bao'

def pickout(inputs, outputs):
    utils = Utils()
    src_dict_list = [utils.load_emb(path) for path in inputs]
    inter_words = list(set.intersection(*[set(src_dict.keys()) for src_dict in src_dict_list]))
    print('Intersection Words: %s' % len(inter_words))
    normalize = lambda x: skpre.normalize(np.reshape(x, (1, -1)))[0]
    for i, path in enumerate(outputs):
        selected = {}
        for word in inter_words:
            selected[word] = normalize(src_dict_list[i][word])
        utils.save_emb(selected, path)

def main():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-i', dest='input',  nargs='+', type=str, required=True, help='input files')
    add_arg('-o', dest='output', nargs='+', type=str, required=True, help='output files')
    args = parser.parse_args()
    assert len(args.input) == len(args.output)
    print('Input files: %s' %  args.input)
    print('Output files: %s' % args.output)
    pickout(args.input, args.output)
    print('Complete')

if __name__ == '__main__':
    main()
