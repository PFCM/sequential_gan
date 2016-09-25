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


def load_and_write(csv_path, label, split_fn=char_split):
    """reads the data and spits it back out into a tensorflow format"""
    with open(csv_path, newline='') as in_f:
        reader = csv.DictReader(in_f, delimiter=',')
        # read them into a list so we can process them a bit
        all_abstracts = [row[label] for row in reader]
    # build a vocabulary mapping
    symbol_counter = collections.Counter(
        split_fn(''.join(all_abstracts)))
    all_abstracts = [list(abstr) for abstr in all_abstracts]
    # TODO: something sensible with the padding characters etc
    symbol_counter['>'] += 1  # sequence start char
    symbol_counter['|'] += 1  # sequence end char
    symbol_counter['~'] += 1  # pad char
    print('Got {} symbols'.format(len(symbol_counter)))
    all_abstracts = [['>'] + abst + ['|'] for abst in all_abstracts]
    # print('\n'.join(['{}: {}'.format(a, b)
    #                  for a, b in symbol_counter.most_common()]))
    # make a mapping of symbol -> id
    vocab = {b[0]: a for a, b in enumerate(symbol_counter.most_common())}
    # write it
    if not os.path.exists('nips_data'):
        os.mkdir('nips_data')
    vocab_file = 'nips_data/{}-vocab.txt'.format(label)
    with open(vocab_file, 'w') as vfile:
        vfile.write('\n'.join(['{},{}'.format(sym, vocab[sym])
                               for sym in sorted(vocab)]))

    # now go through each record, fill in the ids and write it out in
    # tfrecord format.
    fname = 'nips_data/{}s.tfrecords'.format(label)
    with tf.python_io.TFRecordWriter(fname) as out_f:
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
        # now we can pad them :)
        for abstract in all_abstracts:
            # this nesting is absurd
            actual_len = len(abstract)
            pad_amt = longest - actual_len
            padding = ['~'] * pad_amt
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'length': tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[actual_len])),
                        'text': tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[vocab[sym]
                                       for sym in (abstract+padding)]))}))
            out_f.write(example.SerializeToString())
    num = len(all_abstracts)
    print('Done {} records.'.format(num))
    print('----longest: {}'.format(longest))
    print('-- shortest: {}'.format(shortest))
    print('-------mean: {}'.format(total/num))


def get_vocab(path='./nips_data/Title-vocab.txt'):
    """Loads the vocab file, if present"""
    if not os.path.exists(path) or os.path.isdir(path):
        raise ValueError('No file at {}'.format(path))

    vocab = {}
    with open(path) as fp:
        for row in fp:
            items = row[:-1].split(',')
            if len(items) == 2:
                vocab[items[0]] = int(items[1])
            elif len(items) == 3:
                vocab[','] = int(items[-1])
            else:
                print('Problem with this row: {}'.format(row))
    return vocab


def get_nips_tensor(batch_size, sequence_length, label, num_epochs):
    fname = 'nips_data/{}s.tfrecords'.format(label)

    with tf.name_scope('nips_data'):
        fname_producer = tf.train.string_input_producer([fname], num_epochs)

        reader = tf.TFRecordReader()

        _, example = reader.read(fname_producer)

        record = tf.parse_single_example(
            example,
            features={
                'text': tf.FixedLenFeature([sequence_length], tf.int64),
                'length': tf.FixedLenFeature([], tf.int64)})

        # shouldn't have to do any decoding
        # just have to batch it up
        sequence, length = tf.train.shuffle_batch(
            [record['text'], record['length']], batch_size=batch_size,
            num_threads=2, capacity=batch_size*3 + 10, min_after_dequeue=100)
        sequence = tf.transpose(sequence)
        sequence = tf.cast(sequence, tf.int32)
        length = tf.cast(length, tf.int32)
        return sequence, length


if __name__ == '__main__':
    load_and_write(sys.argv[1], sys.argv[2])
    # data = get_nips_tensor(2, 126, 'Title', 100)
    # sess = tf.Session()
    # sess.run(tf.initialize_all_variables())
    # sess.run(tf.initialize_local_variables())
    #
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # vocab = get_vocab()
    # inv_vocab = {b: a for a, b in vocab.items()}
    # name, length = sess.run([data[0], data[1]])
    # print(''.join([inv_vocab[symb] for symb in name]))
    # print(length)
