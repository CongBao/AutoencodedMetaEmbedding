# Launch the training of model and generating meta-embeddings
# File: run.py
# Author: Cong Bao

import argparse
import os

from utils import Corrector

__author__ = 'Cong Bao'

LOG = './log/'
GRAPH = './graphs/'
CHECKPOINT = './checkpoints/'

MODELS = ['DAEME', 'CAEME', 'AAEME']

LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCH = 500
ACTIV = 'relu'
NOISE = 0.05
FACTOR = 1.0

def main():
    """ The main method to receive user inputs """
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-i', dest='input',  type=str,    required=True, nargs='+',     help='directory of source embeddings')
    add_arg('-o', dest='output', type=str,    required=True,                help='directory of yielded meta-embedding')
    add_arg('-m', dest='model',  type=str,    required=True,                help='the model to train, within %s' % MODELS)
    add_arg('-d', dest='dims',   type=int,    required=True, nargs='+',     help='the dimensionality of each source embedding')
    add_arg('-r', dest='rate',   type=float,  default=LEARNING_RATE,        help='learning rate, default %s' % LEARNING_RATE)
    add_arg('-b', dest='batch',  type=int,    default=BATCH_SIZE,           help='batch size, default %s' % BATCH_SIZE)
    add_arg('-e', dest='epoch',  type=int,    default=EPOCH,                help='number of epoches, default %s' % EPOCH)
    add_arg('-a', dest='activ',  type=str,    default=ACTIV,                help='activation function, default %s' % ACTIV)
    add_arg('-n', dest='noise',  type=float,  default=NOISE,                help='ratio of noise, default %s' % NOISE)
    add_arg('-f', dest='factor', type=float,  default=FACTOR, nargs='+',    help='factors of loss function')
    add_arg('--embed-dim',       dest='emb',  type=int, default=300,        help='the dimension of embeddings when applying AAEME')
    add_arg('--log-path',        dest='log',  type=str, default=LOG,        help='the directory of log, default %s' % LOG)
    add_arg('--ckpt-path',       dest='ckpt', type=str, default=CHECKPOINT, help='the directory to store checkpoint file, default %s' % CHECKPOINT)
    add_arg('--oov',             dest='oov',  action='store_true',          help='whether to deal with OOV, default False')
    add_arg('--restore',         dest='load', action='store_true',          help='whether restore saved checkpoint, default False')
    add_arg('--cpu-only',        dest='cpu',  action='store_true',          help='whether use cpu only or not, default False')
    args = parser.parse_args()
    assert args.model in MODELS
    assert len(args.input) == len(args.dims)
    corr = Corrector().correct
    params = {
        'input': tuple(args.input),
        'output': args.output,
        'log': corr(args.log),
        'ckpt': corr(args.ckpt),
        'model': args.model,
        'dims': tuple(args.dims),
        'learning_rate': args.rate,
        'batch': args.batch,
        'epoch': args.epoch,
        'activ': args.activ,
        'factors': args.factor,
        'noise': args.noise,
        'emb': args.emb,
        'oov': args.oov,
        'restore': args.load
    }
    if not isinstance(args.factor, float):
        params['factors'] = tuple(args.factor)
    elif args.model == 'DAEME':
        params['factors'] = tuple([FACTOR] * (len(args.input) + 1))
    else:
        params['factors'] = tuple([FACTOR] * len(args.input))
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if not os.path.exists(params['ckpt']):
        os.makedirs(params['ckpt'])
    from model import AEME
    aeme = AEME(**params)
    log = aeme.logger.log
    log('Source paths: %s' % (params['input'],))
    log('Output path: %s' % params['output'])
    log('Log directory: %s' % params['log'])
    log('Checkpoint directory: %s' % params['ckpt'])
    log('Model type: %s' % params['model'])
    log('Dimensionalities: %s' % (params['dims'],))
    log('Learning rate: %s' % params['learning_rate'])
    log('Batch size: %s' % params['batch'])
    log('Epoch: %s' % params['epoch'])
    log('Activation function: %s' % params['activ'])
    log('Factors: %s' % (params['factors'],))
    log('Noise rate: %s' % params['noise'])
    if params['model'] == 'AAEME':
        log('Embedding dimensionality: %s' % params['emb'])
    log('Output embedding will%s include OOV words' % ('' if args.oov else ' NOT'))
    if params['restore']:
        log('Variables will be restored from saved checkpoint file')
    log('Running on %s' % ('CPU' if args.cpu else 'GPU'))
    aeme.load_data()
    aeme.build_model()
    try:
        aeme.train_model()
        aeme.generate_meta_embed()
        log('Complete')
    except (KeyboardInterrupt, SystemExit):
        log('Abort!', level=aeme.logger.WARN)

if __name__ == '__main__':
    main()
