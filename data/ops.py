"""Actually dealing with the stuff.

Note: go symbol is '>'
      pad symbol is '~'
      end symbol is '|'
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string

import numpy as np
import tensorflow as tf

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


def get_data():
    """Gets the raw data.

    Returns:
        (data, lengths, vocab): the data (list of padded int vectors),
            length of sequences (list of ints) and the vocab.
    """
    data = scr.get_names()
    vocab = get_default_symbols()
    vocab = {symb: i for i, symb in enumerate(sorted(vocab))}
    data, lengths = _translate_and_pad(data, vocab)
    return data, lengths, vocab


def _translate_and_pad(data, vocab):
    """Translates raw data by replacing symbols according to vocab.

    Returns a list of lists (or possibly arrays) of ints.
    """
    all_data = []
    for line in data:
        all_data.append([vocab['>']] +
                        [vocab[symb] if symb in vocab else vocab['?']
                         for symb in line] +
                        [vocab['|']])
    # and pad them all to the maximum length
    max_len = max([len(seq) for seq in all_data])
    lengths = []
    for i in range(len(all_data)):
        lengths.append(len(all_data[i]))
        if lengths[-1] < max_len:
            all_data[i].extend([vocab['~']] * (max_len - lengths[-1]))

    # inv_vocab = {b: a for a, b in vocab.items()}
    # print('\n'.join([''.join([inv_vocab[sym] for sym in seq])
    #                  for seq in all_data]))

    return all_data, lengths


def get_default_symbols():
    """returns the symbols used by the default preprocessing"""
    return list(string.ascii_letters) + ['>', '?', '.', '-', ',', '&', '~',
                                         '|', ' ']


def _clean(input_):
    """Tidies up -- removes unwanted punctuation. Expects a numpy array
    of some kind of unicode, we are going to just call str on it.
    Converts to array of int32 in the end"""
    # place newlines with the go symbol
    stripped = input_.replace('\n', '>')
    # replace anything not in the vocab with '?'
    # and also convert to ids
    vocab = {symb: i for i, symb in enumerate(get_default_symbols())}
    return np.array([vocab[item] if item in vocab else vocab['?']
                     for item in stripped])


def get_batch_tensor(batch_size, sequence_length, num_epochs,
                     filename='names.txt',
                     preprocessor=_clean):
    """Gets the data in good tensorflow ways. Adds a queue runner so be sure to
    start it."""
    with tf.name_scope('input'):
        # the data is tiny so just load it, clean it and throw it into a
        # constant
        with open(filename) as f:
            all_data = f.read()
        # process it
        all_data = preprocessor(all_data)
        # just chop off the end to make sure sequence_length * batch_size
        # divides the total number of records
        print(all_data)
        num_batches = all_data.shape[0] // (sequence_length * batch_size)
        all_data = all_data[:num_batches * sequence_length * batch_size]
        all_data = np.reshape(all_data, (-1, sequence_length))
        # and make the queue
        data = tf.train.slice_input_producer(
            [tf.constant(all_data)],
            num_epochs=num_epochs,
            shuffle=True,
            capacity=batch_size*sequence_length)

        # very much unconvinced this is all the right way round
        batch = tf.train.batch([data], batch_size=batch_size,
                               enqueue_many=True, num_threads=2)
        batch = tf.transpose(batch)
        return tf.unpack(batch)
