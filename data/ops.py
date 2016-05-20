"""Actually dealing with the stuff"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


import data.scrape as scr


def _build_vocab(data):
    """Builds a mapping from symbols -> integer ids."""
    vocab = {}
    id_num = 0
    for sequence in data:
        for symbol in sequence:
            data[symbol] = id_num
            id_num += 1
    return vocab


def _translate_and_pad(data, vocab):
    """Translates according to the given sequence and adds padding"""
    data_ids = [[vocab[symb] for symb in seq] for seq in data]
    maxlen = max(data_ids, key=len)
    lengths = [len(seq) for seq in data]
    if 'PAD' not in vocab:  # we need to pad with something
        vocab['PAD'] = len(vocab)
    padded_data = []
    for seq in data_ids:
        padded_data.append(
            seq + [vocab['PAD']] * (maxlen - len(seq)))
    return padded_data, lengths


def get_data():
    """Gets the raw data.

    Returns:
        (data, lengths, vocab): the data (list of padded int vectors),
            length of sequences (list of ints) and the vocab.
    """
    data = scr.get_names()
    vocab = _build_vocab(data)
    data, lengths = _translate_and_pad(data, vocab)
    return data, lengths, vocab


def batch_iter(data, batch_size):
    """yields batches and time shifted baches for learning sequences :)"""
    pass
