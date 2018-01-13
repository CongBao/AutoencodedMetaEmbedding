#!/usr/bin/env python
"""
reduce dimensionality
"""

from __future__ import print_function

import argparse
import sys
sys.path.append('/home/cong/fyp')

import numpy as np

from aeme.utils import data_process, io

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
            new[word] = data_process.normalize(np.add(cbow[word], glove[word]), 1.0)
        return new

def main():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-i', dest='input', type=str, required=True, help='input file')
    add_arg('-o', dest='output', type=str, required=True, help='output file')
    add_arg('-m', dest='method', type=str, default=METHOD, help='method used to reduce dimensionality')
    args = parser.parse_args()
    assert args.method in METHODS + ['all']
    raw = io.load_embeddings(args.input)
    if args.method == 'all':
        out = str(args.output)
        for mth in METHODS:
            print('Runing %s...' % mth)
            io.save_embeddings(reduce_dim(mth, raw), out.replace('.txt', '.' + mth + '.txt'))
    else:
        io.save_embeddings(reduce_dim(args.method, raw), args.output)

if __name__ == '__main__':
    main()
