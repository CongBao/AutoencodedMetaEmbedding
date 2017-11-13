#!/usr/bin/env python

import argparse
import os

import utils
from logger import Logger

__author__ = 'Cong Bao'

logger = Logger(str(os.path.basename(__file__)).replace('.py', ''))

def measure(source_list):
    # load and normalize source embeddings
    logger.log('Loading file: %s' % source_list[0])
    cbow_dict = utils.load_embeddings(source_list[0])
    
    logger.log('Loading file: %s' % source_list[1])
    glove_dict = utils.load_embeddings(source_list[1])

    # find intersection and union of two sources
    inter_words = set(cbow_dict.keys()) & set(glove_dict.keys())
    union_words = set(cbow_dict.keys()) | set(glove_dict.keys())

    logger.log('Size of %s: %s' % (source_list[0], len(cbow_dict)))
    logger.log('Size of %s: %s' % (source_list[1], len(glove_dict)))
    logger.log('Number of intersection words: %s' % len(inter_words))
    logger.log('Number of union words: %s' % len(union_words))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input', nargs='+', type=str, required=True, help='the input file(s) containing source vectors')
    args = parser.parse_args()
    logger.log('Input file(s): %s' % args.input)
    measure(args.input)

if __name__ == '__main__':
    main()
