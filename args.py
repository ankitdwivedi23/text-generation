"""Command-line arguments for setup.py, train.py, test.py.
"""

import argparse

def get_setup_args():
    """Get arguments needed in setup.py"""
    parser = argparse.ArgumentParser('Pre-process data')

    add_common_args(parser)
    
    parser.add_argument('--glove_file',
                        type=str,
                        default='./data/glove/glove.6B/glove.6B.50d.txt',
                        help='Path to GloVe word vectors')
    parser.add_argument('--glove_dim',
                        type=int,
                        default=50,
                        help='Size of GloVe vectors to use')
    parser.add_argument('--glove_num_vecs',
                        type=int,
                        default=400000,
                        help='Number of GloVe vectors')
       
    args = parser.parse_args()

    return args

def get_train_args():
    """Get arguments needed in train.py"""

    parser = argparse.ArgumentParser('Train a language model')
    add_common_args(parser)
    
    args = parser.parse_args()

    return args


def add_common_args(parser):

    parser.add_argument('--train_file',
                        type=str,
                        default='./data/train.txt')
    parser.add_argument('--dev_file',
                        type=str,
                        default='./data/dev.txt')
    parser.add_argument('--test_file',
                        type=str,
                        default='./data/test.txt')
    parser.add_argument('--train_features_file',
                        type=str,
                        default='./data/train.npz')
    parser.add_argument('--dev_features_file',
                        type=str,
                        default='./data/dev.npz')
    parser.add_argument('--test_features_file',
                        type=str,
                        default='./data/test.npz')
    parser.add_argument('--word_emb_file',
                        type=str,
                        default='./data/word_emb.json')
    parser.add_argument('--word2idx_file',
                        type=str,
                        default='./data/word2idx.json')