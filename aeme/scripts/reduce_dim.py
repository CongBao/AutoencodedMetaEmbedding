#!/usr/bin/env python
"""
reduce dimensionality
"""

import argparse
import sys

import numpy as np

from aeme.utils import data_process, io

__author__ = 'Cong Bao'

MODULE_PATH = '/home/cong/fyp'
METHOD = 'svd'

METHODS = ['src1', 'src2', 'avg', 'svd']

def main():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--module-path', dest='module', type=str, default=MODULE_PATH, help='the path of aeme module')
    add_arg('-i', dest='input', type=str, required=True, help='input file')
    add_arg('-o', dest='output', type=str, required=True, help='output file')
    add_arg('-m', dest='method', type=str, default=METHOD, help='method used to reduce dimensionality')
    args = parser.parse_args()
    sys.path.append(args.module)
    assert args.method in METHODS
    raw = io.load_embeddings(args.input)
    new = {}
    if args.method == 'svd':
        new = data_process.tsvd(raw)
    else:
        cbow = {}
        glove = {}
        for key, val in raw.items():
            cbow[key] = val[:300]
            glove[key] = val[300:]
        if args.method == 'src1':
            new = cbow
        elif args.method == 'src2':
            new = glove
        elif args.method == 'avg':
            for word in raw.keys():
                new[word] = data_process.normalize(np.add(cbow[word], glove[word]), 1.0)
    io.save_embeddings(new, args.output)

if __name__ == '__main__':
    main()
