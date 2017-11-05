#!/usr/bin/env python
"""
Pick out the inter words.
"""

from __future__ import division

import argparse
import os

import numpy as np

import utils
from logger import Logger

__author__ = 'Cong Bao'

logger = Logger(str(os.path.basename(__file__)).replace('.py', ''))

def pickout_embedding(source_list, output_list):
    # load embedding data
    # load and normalize source embeddings
    logger.log('Loading file: %s' % source_list[0])
    cbow_dict = utils.load_embeddings(source_list[0])
    logger.log('normalizing source embeddings')
    cbow_dict = utils.normalize_embeddings(cbow_dict, 1.0)

    logger.log('Loading file: %s' % source_list[1])
    glove_dict = utils.load_embeddings(source_list[1])
    logger.log('normalizing source embeddings')
    glove_dict = utils.normalize_embeddings(glove_dict, 1.0)

    # find intersection of two sources
    inter_words = set(cbow_dict.keys()) & set(glove_dict.keys())
    logger.log('Number of intersection words: %s' % len(inter_words))
    
    # calculate the meta embedding
    selected_cbow = {}
    selected_glove = {}
    for word in inter_words:
        selected_cbow[word] = cbow_dict[word]
        selected_glove[word] = glove_dict[word]
    logger.log('Saving data into output file: %s' % output_list[0])
    utils.save_embeddings(selected_cbow, output_list[0])
    logger.log('Saving data into output file: %s' % output_list[1])
    utils.save_embeddings(selected_glove, output_list[1])
    logger.log('Complete.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input', nargs='+', type=str, required=True, help='the input file(s) containing source vectors')
    parser.add_argument('-o', dest='output', nargs='+', type=str, required=True, help='the output file(s)')
    args = parser.parse_args()
    logger.log('Input file(s): %s' % args.input)
    logger.log('Output file(s): %s' % args.output)
    pickout_embedding(args.input, args.output)

if __name__ == '__main__':
    main()