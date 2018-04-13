# train the model and generate meta embeddings

import argparse
import os

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
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-i', dest='input',  type=str, required=True, nargs='+',    help='directory of source embeddings')
    add_arg('-o', dest='output', type=str, required=True,               help='directory of output meta embedding')
    add_arg('-m', dest='model',  type=str, required=True,               help='the model to train, within %s' % MODELS)
    add_arg('-d', dest='dims',   type=int, required=True, nargs='+',    help='the dimensionality of each source embedding')
    add_arg('-r', dest='rate',   type=float, default=LEARNING_RATE,     help='learning rate, default %s' % LEARNING_RATE)
    add_arg('-b', dest='batch',  type=int,   default=BATCH_SIZE,        help='batch size, default %s' % BATCH_SIZE)
    add_arg('-e', dest='epoch',  type=int,   default=EPOCH,             help='number of epoches, default %s' % EPOCH)
    add_arg('-a', dest='activ',  type=str,   default=ACTIV,             help='activation function, default %s' % ACTIV)
    add_arg('-n', dest='noise',  type=float, default=NOISE,             help='ratio of noise, default %s' % NOISE)
    add_arg('-f', dest='factor', type=float, default=FACTOR, nargs='+', help='factors of loss function')
    add_arg('--embed-dim',       dest='emb', type=int, default=300,     help='the dimension of embeddings when applying AAEME')
    add_arg('--log-path',        dest='log', type=str, default=LOG,     help='the directory of log, default %s' % LOG)
    add_arg('--cpu-only',        dest='cpu', action='store_true',       help='whether use cpu only or not, default False')
    args = parser.parse_args()
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    assert args.model in MODELS
    assert len(args.input) == len(args.dims)
    params = {
        'input': tuple(args.input),
        'output': args.output,
        'log': args.log,
        'model': args.model,
        'dims': tuple(args.dims),
        'learning_rate': args.rate,
        'batch': args.batch,
        'epoch': args.epoch,
        'activ': args.activ,
        'factors': args.factor,
        'noise': args.noise,
        'emb': args.emb
    }
    if not isinstance(args.factor, float):
        params['factors'] = tuple(args.factor)
    elif args.model == 'DAEME':
        params['factors'] = tuple([FACTOR] * (len(args.input) + 1))
    else:
        params['factors'] = tuple([FACTOR] * len(args.input))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from model import AEME
    aeme = AEME(**params)
    aeme.logger.log('Source directories: %s' % (params['input'],))
    aeme.logger.log('Output directory: %s' % params['output'])
    aeme.logger.log('Model type: %s' % params['model'])
    aeme.logger.log('Dimensionalities: %s' % (params['dims'],))
    aeme.logger.log('Learning rate: %s' % params['learning_rate'])
    aeme.logger.log('Batch size: %s' % params['batch'])
    aeme.logger.log('Epoch: %s' % params['epoch'])
    aeme.logger.log('Activation function: %s' % params['activ'])
    aeme.logger.log('Factors: %s' % (params['factors'],))
    aeme.logger.log('Noise rate: %s' % params['noise'])
    if params['model'] == 'AAEME':
        aeme.logger.log('Embedding dimensionality: %s' % params['emb'])
    aeme.logger.log('Running on %s' % ('CPU' if args.cpu else 'GPU'))
    aeme.load_data()
    aeme.build_model()
    try:
        aeme.train_model()
        aeme.generate_meta_embed()
    except (KeyboardInterrupt, SystemExit):
        aeme.logger.log('Abort!', level=aeme.logger.WARN)

if __name__ == '__main__':
    main()
