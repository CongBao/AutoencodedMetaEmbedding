#!/usr/bin/env python
"""
Baseline use simple concatenation
"""

from __future__ import division

import argparse
import os

import numpy as np

from aeme.utils import io
from aeme.utils import preprocess
from aeme.utils.logger import Logger

__author__ = 'Cong Bao'

logger = Logger(str(os.path.basename(__file__)).replace('.py', ''))

def train_embedding(source_list, output_path):
    # load and normalize source embeddings
    logger.log('Loading file: %s' % source_list[0])
    cbow_dict = io.load_embeddings(source_list[0])
    logger.log('normalizing source embeddings')
    cbow_dict = preprocess.normalize_embeddings(cbow_dict, 1.0)

    logger.log('Loading file: %s' % source_list[1])
    glove_dict = io.load_embeddings(source_list[1])
    logger.log('normalizing source embeddings')
    glove_dict = preprocess.normalize_embeddings(glove_dict, 1.0)

    # find intersection of two sources
    inter_words = set(cbow_dict.keys()) & set(glove_dict.keys())
    logger.log('Number of intersection words: %s' % len(inter_words))
    
    # calculate the meta embedding
    meta_embedding = {}
    for word in inter_words:
        meta_embedding[word] = np.concatenate([cbow_dict[word], glove_dict[word]])
    logger.log('Saving data into output file: %s' % output_path)
    io.save_embeddings(meta_embedding, output_path)
    logger.log('Complete.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input', nargs='+', type=str, required=True, help='the input file(s) containing source vectors')
    parser.add_argument('-o', dest='output', type=str, required=True, help='the output file')
    args = parser.parse_args()
    logger.log('Input file(s): %s' % args.input)
    logger.log('Output file: %s' % args.output)
    train_embedding(args.input, args.output)

if __name__ == '__main__':
    main()