# find k-nearest neighbours of a word
# File: knn.py
# Author: Cong Bao

from __future__ import print_function

import argparse
from operator import itemgetter

from scipy.spatial.distance import cosine

from utils import Utils

__author__ = 'Cong Bao'

def _knn(emb_dict, word, k=None, output_path=None, return_dis=False):
    dis_list = []
    if word not in emb_dict.keys():
        print('Word not in vocabulary!')
        return
    word_val = emb_dict[word]
    for key, val in emb_dict.items():
        if key != word:
            dis_list.append((key, 1 - cosine(word_val, val)))
    dis_list.sort(key=itemgetter(1), reverse=True)
    if return_dis:
        return dis_list
    if output_path:
        res_dict = {}
        for item in dis_list[:k]:
            w, v = item
            res_dict[w] = v
        util.save_emb(res_dict, output_path)
    else:
        print('%s nearest words of %s: ' % (k, word))
        for item in dis_list[:k]:
            print(item[0])

def knn(input_path, word, k, output_path=None):
    emb_dict = util.load_emb(input_path)
    if isinstance(word, str):
        _knn(emb_dict, word, k, output_path)
    elif isinstance(word, (list, tuple)):
        for w in word:
            _knn(emb_dict, w, k, output_path)

def main():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-i', dest='input', type=str, required=True, help='embedding path')
    add_arg('-o', dest='output', type=str, default=None, help='result output path')
    add_arg('-w', dest='word', type=str, nargs='+', required=True, help='a test word')
    add_arg('-k', dest='num', type=int, required=True, help='number of nearest neighbours')
    args = parser.parse_args()
    try:
        knn(args.input, args.word, args.num, args.output)
    except (KeyboardInterrupt, SystemExit):
        print('Abort!')

if __name__ == '__main__':
    util = Utils()
    main()
