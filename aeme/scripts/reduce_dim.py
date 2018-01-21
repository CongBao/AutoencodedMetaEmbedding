#!/usr/bin/env python
"""
reduce dimensionality
"""

from __future__ import print_function

import argparse
import sys
sys.path.append('/home/cong/fyp')

import numpy as np

from aeme.utils import data_process, embed_io

__author__ = 'Cong Bao'

METHOD = 'svd'

METHODS = ['src1', 'src2', 'avg', 'svd']

def reduce_dim(mtd, raw):
    if mtd == 'svd':
        return data_process.tsvd(raw)
    cbow = {}
    glove = {}
    for key, val in raw.items():
        cbow[key] = val[:300]
        glove[key] = val[300:]
    if mtd == 'src1':
        return cbow
    elif mtd == 'src2':
        return glove
    elif mtd == 'avg':
        new = {}
        for word in raw.keys():
            new[word] = np.add(cbow[word], glove[word])
        return data_process.normalize_emb(new)

def main():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-i', dest='input', type=str, required=True, help='input file')
    add_arg('-o', dest='output', type=str, required=True, help='output file')
    add_arg('-m', dest='method', type=str, nargs='+', default=METHOD, help='method used to reduce dimensionality')
    args = parser.parse_args()
    assert set(args.method).issubset(set(METHODS + ['all']))
    raw = embed_io.load_embeddings(args.input)
    out = str(args.output)
    for mth in (METHODS if args.method[0] == 'all' else args.method):
        print('Runing %s...' % mth)
        embed_io.save_embeddings(reduce_dim(mth, raw), out.replace('.txt', '.' + mth + '.txt'))

if __name__ == '__main__':
    main()
