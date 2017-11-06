#!/usr/bin/env python
"""
execute the aeme
"""

import argparse
import json
import os
import sys

__author__ = 'Cong Bao'

MODULE_PATH = r'F:/GitHub/AutoencodingMetaEmbedding'

MODEL_TYPES = ['ae', 'conc', 'dae', 'linear', 'sdae']
MODEL_NAMES = ['LinearModel']
NOISE_TYPES = ['GS', 'MN', 'SP', 'None']

LOG_PATH = './log/'
GRAPH_PATH = './graphs/'

LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 1000

NOISE_TYPE = 'None'
NOISE_RATIO = 0.5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module-path', dest='module', type=str, default=MODULE_PATH, help='the path of aeme module')
    parser.add_argument('-m', dest='model', nargs='+', type=str, required=True, help='the model to train')
    parser.add_argument('-i', dest='input', nargs='+', type=str, required=True, help='the input file(s) containing source vectors')
    parser.add_argument('-o', dest='output', type=str, required=True, help='the output file')
    parser.add_argument('--log-path', dest='log', type=str, default=LOG_PATH, help='the directory of log')
    parser.add_argument('--graph-path', dest='graph', type=str, default=GRAPH_PATH, help='the directory of graph')
    parser.add_argument('-r', dest='rate', type=float, default=LEARNING_RATE, help='the learning rate of gradient descent')
    parser.add_argument('-b', dest='batch', type=int, default=BATCH_SIZE, help='the size of batches')
    parser.add_argument('-e', dest='epoch', type=int, default=EPOCHS, help='the number of epoches to train')
    parser.add_argument('--noise-type', dest='type', type=str, default=NOISE_TYPE, help='the type of noise')
    parser.add_argument('--noise-ratio', dest='ratio', type=float, default=NOISE_RATIO, help='the ratio of noise')
    parser.add_argument('--cpu-only', dest='cpu', action='store_true', help='if use cpu only')
    args = parser.parse_args()
    sys.path.append(args.module)
    model_type = args.model[0]
    model_name = args.model[1]
    assert model_type in MODEL_TYPES
    assert model_name in MODEL_NAMES
    exec('from aeme.models.' + model_type + '.' + model_type + '_model import ' + model_name)
    input_path = json.dumps({'cbow': args.input[0], 'glove': args.input[1]})
    assert args.type in NOISE_TYPES
    model = eval(model_name + '(\'' + 
                 input_path + '\', \'' +
                 args.output + '\', \'' +
                 args.log + '\', \'' +
                 args.graph + '\', ' +
                 str(args.rate) + ', ' + 
                 str(args.batch) + ', ' +
                 str(args.epoch) + ', \'' +
                 args.type + '\', ' +
                 str(args.ratio) + ')')
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    model.logger.log('AEME module path: %s' % args.module)
    model.logger.log('Model type: %s, model name: %s' % (model_type, model_name))
    model.logger.log('Input files: %s' % args.input)
    model.logger.log('Output file: %s' % args.output)
    model.logger.log('Log path: %s' % args.log)
    model.logger.log('Graph path: %s' % args.graph)
    model.logger.log('Learning rate: %s' % args.rate)
    model.logger.log('Batch size: %s' % args.batch)
    model.logger.log('Epoches to train: %s' % args.epoch)
    model.logger.log('Noise type: %s' % args.type)
    model.logger.log('Noise ratio: %s' % args.ratio)
    model.logger.log('Running on %s' % ('CPU' if args.cpu else 'GPU'))
    model.run()
    
if __name__ == '__main__':
    main()
