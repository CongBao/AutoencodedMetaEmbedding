#!/usr/bin/env python
"""
execute the aeme
"""

import argparse
import os
import sys

__author__ = 'Cong Bao'

MODULE_PATH = '/home/cong/fyp'
INPUT_PATH = ['/home/cong/data/CBOW-full.txt', '/home/cong/data/GloVe-full.txt']
MODEL = ['ae', 'AEModel']

MODEL_TYPES = ['ae', 'conc', 'linear', 'sae']
MODEL_NAMES = ['AEModel', 'TiedAEModel', 'ZipAEModel', 'DActivAEModel', 'DeepAEModel', 'AvgAEModel', 'ConcAEModel', 'DActivAvgAEModel', 'ConcModel', 'AvgModel', 'LinearModel', 'TiedLinearModel', 'SAEModel']
ACTIVATION_TYPES = ['sigmoid_m', 'sigmoid', 'tanh', 'relu', 'lrelu', 'prelu', 'elu', 'selu', 'None']
NOISE_TYPES = ['GS', 'MN', 'SP', 'None']
META_TYPES = ['src1', 'src2', 'conc', 'avg', 'svd']

LOG_PATH = './log/'
GRAPH_PATH = './graphs/'
CHECKPOINT_PATH = './checkpoints/'

LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 1000
VALIDATION_RATIO = 0.05
REGULARIZATION_RATIO = None
ACTIVATION = 'sigmoid_m'
META_TYPE = 'conc'
CHECKPOINT_RATIO = 0

NOISE_TYPE = 'MN'
NOISE_RATIO = 0.05

FACTORS = [1.0, 1.0, 1.0]
STACK_TRAIN = [2, 1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module-path', dest='module', type=str, default=MODULE_PATH, help='the path of aeme module')
    parser.add_argument('-m', dest='model', nargs=2, type=str, default=MODEL, help='the model to train')
    parser.add_argument('-i', dest='input', nargs=2, type=str, default=INPUT_PATH, help='the input file(s) containing source vectors')
    parser.add_argument('-o', dest='output', type=str, required=True, help='the output file')
    parser.add_argument('--log-path', dest='log', type=str, default=LOG_PATH, help='the directory of log')
    parser.add_argument('--graph-path', dest='graph', type=str, default=GRAPH_PATH, help='the directory of graph')
    parser.add_argument('--checkpoint-path', dest='checkpoint', type=str, default=CHECKPOINT_PATH, help='the directory of checkpoint')
    parser.add_argument('-r', dest='rate', type=float, default=LEARNING_RATE, help='the learning rate of gradient descent')
    parser.add_argument('-b', dest='batch', type=int, default=BATCH_SIZE, help='the size of batches')
    parser.add_argument('-e', dest='epoch', type=int, default=EPOCHS, help='the number of epoches to train')
    parser.add_argument('-a', dest='activ', type=str, default=ACTIVATION, help='the activation function')
    parser.add_argument('-f', dest='factor', type=float, nargs='+', default=FACTORS, help='factors add to loss function')
    parser.add_argument('--valid-ratio', dest='valid', type=float, default=VALIDATION_RATIO, help='the ratio of validation set')
    parser.add_argument('--reg-ratio', dest='reg', type=float, default=REGULARIZATION_RATIO, help='the ratio of regularization')
    parser.add_argument('--noise-type', dest='type', type=str, default=NOISE_TYPE, help='the type of noise')
    parser.add_argument('--noise-ratio', dest='ratio', type=float, default=NOISE_RATIO, help='the ratio of noise')
    parser.add_argument('--stacked-train', dest='stack', nargs='+', type=int, default=STACK_TRAIN, help='the times of stacked training')
    parser.add_argument('--meta-type', dest='meta', type=str, default=META_TYPE, help='the type to generate meta embedding')
    parser.add_argument('--checkpoint-ratio', dest='ckptratio', type=int, default=CHECKPOINT_RATIO, help='the number of epoches that checkpoint is saved')
    parser.add_argument('--restore-model', dest='restore', action='store_true', help='if use existing model')
    parser.add_argument('--cpu-only', dest='cpu', action='store_true', help='if use cpu only')
    args = parser.parse_args()

    sys.path.append(args.module)
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    model_type, model_name = args.model
    assert model_type in MODEL_TYPES
    assert model_name in MODEL_NAMES
    assert args.activ in ACTIVATION_TYPES
    assert args.type in NOISE_TYPES
    assert args.meta in META_TYPES
    exec('from aeme.models.' + model_type + '.' + model_type + '_model import ' + model_name)
    params = {
        'input_path': {
            'cbow': args.input[0],
            'glove': args.input[1]
        },
        'output_path': args.output,
        'graph_path': args.graph,
        'checkpoint_path': args.checkpoint,
        'learning_rate': args.rate,
        'batch_size': args.batch,
        'epoch': args.epoch,
        'valid_ratio': args.valid,
        'reg_ratio': args.reg,
        'activ_func': args.activ,
        'factors': tuple(args.factor),
        'noise_type': args.type,
        'noise_ratio': args.ratio,
        'meta_type': args.meta,
        'checkpoint_ratio': args.ckptratio,
        'stacked_train': {
            'separate': args.stack[0],
            'combine': args.stack[1]
        },
        'restore_model': args.restore
    }
    model = eval(model_name + '(\'' + args.log + '\')')
    model.logger.log('AEME module path: %s' % args.module)
    model.logger.log('Model type: %s, model name: %s' % (model_type, model_name))
    model.logger.log('Input files: %s' % args.input)
    model.logger.log('Output file: %s' % args.output)
    model.logger.log('Log path: %s' % args.log)
    if model_type != 'conc':
        model.logger.log('Graph path: %s' % args.graph)
        model.logger.log('Checkpoint path: %s' % args.checkpoint)
        model.logger.log('Learning rate: %s' % args.rate)
        model.logger.log('Batch size: %s' % args.batch)
        model.logger.log('Epoches to train: %s' % args.epoch)
        model.logger.log('Validation set ratio: %s' % args.valid)
        model.logger.log('Regularization ratio: %s' % args.reg)
        if model_type == 'ae' or model_type == 'sae':
            model.logger.log('Activation function: %s' % args.activ)
        model.logger.log('Loss factors: %s' % args.factor)
        model.logger.log('Noise type: %s' % args.type)
        if args.type is not None:
            model.logger.log('Noise ratio: %s' % args.ratio)
        if model_type == 'sae':
            model.logger.log('Stacked training times, separate: %s, combine: %s' % (args.stack[0], args.stack[1]))
        model.logger.log('Meta type: %s' % args.meta)
        model.logger.log('Checkpoint saving ratio: %s' % args.ckptratio)
        if args.restore:
            model.logger.log('Using variables in prestored model')
        model.logger.log('Running on %s' % ('CPU' if args.cpu else 'GPU'))
    model.run(params)

if __name__ == '__main__':
    main()
