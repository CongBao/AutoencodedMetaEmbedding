# train the model and generate meta embeddings

from __future__ import print_function

import argparse
import os

__author__ = 'Cong Bao'

GRAPH = './graphs/'
CHECKPOINT = './checkpoints/'

MODELS = ['DAEME', 'CAEME', 'AAEME']
ACTIVS = ['relu', 'sigmoid']

LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCH = 500
ACTIV = 'relu'
NOISE = 0.05
FACTOR = 1.0

def main():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-i', dest='input',  type=str, required=True, nargs='+', help='directory of source embeddings')
    add_arg('-o', dest='output', type=str, required=True, help='directory of output meta embedding')
    add_arg('-m', dest='model',  type=str, required=True, help='the model to train, within %s' % MODELS)
    add_arg('-d', dest='dims', type=int, required=True, nargs='+', help='the dimensionality of each source embedding')
    add_arg('-r', dest='rate',   type=float, default=LEARNING_RATE,    help='learning rate, default %s' % LEARNING_RATE)
    add_arg('-b', dest='batch',  type=int,   default=BATCH_SIZE,       help='batch size, default %s' % BATCH_SIZE)
    add_arg('-e', dest='epoch',  type=int,   default=EPOCH,            help='number of epoches, default %s' % EPOCH)
    add_arg('-a', dest='activ',  type=str,   default=ACTIV,       help='activation function within %s, default %s' % (ACTIVS, ACTIV))
    add_arg('-n', dest='noise',  type=float, default=NOISE, help='ratio of noise, default %s' % NOISE)
    add_arg('-f', dest='factor', type=float, default=FACTOR, nargs='+', help='factors of loss function')
    add_arg('--graph-path',      dest='graph',      type=str, default=GRAPH,      help='path to save tensor graphs, default %s' % GRAPH)
    add_arg('--checkpoint-path', dest='checkpoint', type=str, default=CHECKPOINT, help='path to save checkpoint files, default %s' % CHECKPOINT)
    add_arg('--cpu-only',        dest='cpu',        action='store_true',               help='whether use cpu only or not, default False')
    args = parser.parse_args()
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    assert args.activ in ACTIVS
    assert args.model in MODELS
    assert len(args.input) == len(args.dims)
    params = {
        'input': tuple(args.input),
        'output': args.output,
        'graph': args.graph,
        'checkpoint': args.checkpoint,
        'model': args.model,
        'dims': tuple(args.dims),
        'learning_rate': args.rate,
        'batch': args.batch,
        'epoch': args.epoch,
        'activ': args.activ,
        'factors': tuple(args.factor) if not isinstance(args.factor, float) else tuple([FACTOR] * len(args.input)),
        'noise': args.noise
    }
    print('Source directories: %s' % (params['input'],))
    print('Output directory: %s' % params['output'])
    print('Graph path: %s' % params['graph'])
    print('Checkpoint path: %s' % params['checkpoint'])
    print('Model type: %s' % params['model'])
    print('Dimensionalities: %s' % (params['dims'],))
    print('Learning rate: %s' % params['learning_rate'])
    print('Batch size: %s' % params['batch'])
    print('Epoch: %s' % params['epoch'])
    print('Activation function: %s' % params['activ'])
    print('Factors: %s' % (params['factors'],))
    print('Noise rate: %s' % params['noise'])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from model import AEME
    aeme = AEME(**params)
    aeme.load_data()
    aeme.build_model()
    aeme.train_model()
    aeme.generate_meta_embed()

if __name__ == '__main__':
    main()
