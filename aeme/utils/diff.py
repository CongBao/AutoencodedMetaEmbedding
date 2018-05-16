# find most different words in source embedding sets
# File: diff.py
# Author: Cong Bao

from __future__ import print_function

import argparse
from heapq import heappop, heappush
from operator import itemgetter

from scipy.spatial.distance import cosine
from tqdm import tqdm

from utils import Utils

__author__ = 'Cong Bao'

def _knn(emb_dict, word):
    dis_list = []
    if word not in emb_dict.keys():
        print('Word not in vocabulary!')
        return
    word_val = emb_dict[word]
    for key, val in emb_dict.items():
        if key != word:
            heappush(dis_list, (cosine(word_val, val), key))
    return dis_list

def diff(inputs, output, n, word_path=None):
    utils = Utils()
    src_dict_list = [utils.load_emb(path) for path in inputs]
    word_list = list(set.intersection(*[set(src_dict.keys()) for src_dict in src_dict_list]))
    if word_path is not None:
        word_list = set(utils.load_words(word_path)) & set(word_list)
    inter_num = []
    for word in tqdm(word_list, total=len(word_list), bar_format='Progress: {percentage:3.0f}% {r_bar}'):
        w_sets = []
        for emb_dict in src_dict_list:
            w_set = set()
            res = _knn(emb_dict, word)
            for _ in range(n):
                _, w = heappop(res)
                w_set.add(w)
            w_sets.append(w_set)
        inter_num.append((word, len(set.intersection(*w_sets))))
    inter_num.sort(key=itemgetter(1))
    if output:
        utils.save_words(inter_num, output)
    print(inter_num[:50])

def main():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-i', dest='input', type=str, required=True, nargs='+', help='embedding path')
    add_arg('-o', dest='output', type=str, default=None, help='result output path')
    add_arg('-n', dest='num', type=int, required=True, help='the number of nearest neighbours')
    add_arg('-w', dest='word', type=str, default=None, help='the path of word list')
    args = parser.parse_args()
    try:
        diff(args.input, args.output, args.num, args.word)
    except (KeyboardInterrupt, SystemExit):
        print('Abort!')

if __name__ == '__main__':
    main()
