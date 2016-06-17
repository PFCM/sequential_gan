""""pull the abstracts out of the Papers.csv file on Kaggle."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sys
import os
import csv

import numpy as np
import tensorflow as tf


def char_split(input_str):
    """yields each character in turn"""
    for char in input_str:
        yield char


def load_and_write(csv_path, split_fn=char_split):
    """reads the data and spits it back out into a tensorflow format"""
    with open(csv_path, newline='') as in_f:
        reader = csv.DictReader(in_f, delimiter=',')
        # read them into a list so we can process them a bit
        all_abstracts = [row['Abstract'] for row in reader]
    # build a vocabulary mapping
    symbol_counter = collections.Counter(
        split_fn(''.join(all_abstracts)))

    # these are quite long (average ≈1000 characters) so we will add a special
    # start character to them, so that the net knows when they start and
    # stop. Hopefully this means we could in theory cat them all together
    # and train on big contiguous blocks, but that we might still be able to
    # generate individual samples.
    symbol_counter['GO'] += 1
    print('Got {} symbols'.format(len(symbol_counter)))
    all_abstracts = ['GO' + abst for abst in all_abstracts]
    # print('\n'.join(['{}: {}'.format(a, b)
    #                  for a, b in symbol_counter.most_common()]))
    # make a mapping of symbol -> id
    vocab = {b[0]: a for a, b in enumerate(symbol_counter.most_common())}
    # write it
    if not os.path.exists('nips_data'):
        os.mkdir('nips_data')
    with open('nips_data/vocab.txt', 'w') as vfile:
        vfile.write('\n'.join(['{},{}'.format(sym, vocab[sym])
                               for sym in sorted(vocab)]))

    # now go through each record, fill in the ids and write it out in
    # tfrecord format.
    with tf.python_io.TFRecordWriter('nips_data/abstracts.tfrecords') as out_f:
        # grab some stats while we're here
        longest = 0
        shortest = 1e6
        total = 0
        for abstract in all_abstracts:
            length = len(abstract)
            total += length
            if length > longest:
                longest = length
            if length < shortest:
                shortest = length
            # this nesting is absurd
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'text': tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[vocab[sym]
                                       for sym in abstract]))}))
            out_f.write(example.SerializeToString())
    num = len(all_abstracts)
    print('Done {} records.'.format(num))
    print('----longest: {}'.format(longest))
    print('-- shortest: {}'.format(shortest))
    print('-------mean: {}'.format(total/num))


if __name__ == '__main__':
    load_and_write(sys.argv[1])
