"""generative adversarial?"""
import numpy as np
import tensorflow as tf


class new_collection(object):
    """decorator that runs a function and adds any variables it adds to
    TRAINABLE_VARIABLES to a new collection with specified name."""

    def __init__(self, name):
        self.name = name

    def __call__(self, func):
        def _wrap(*args, **kwargs):
            initial_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            # do it
            retval = func(*args, **kwargs)
            # see what it added
            new_tvars = [var for var
                         in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                         if var not in initial_tvars]
            for var in new_tvars:
                tf.add_to_collection(self.name, var)
            return retval  # in case, probably nothing
        return _wrap


@new_collection('discriminator')
def discriminative_model(inputs, num_layers, width, classifier_shape,
                         embedding_matrix, sequence_lengths):
    """Gets the discriminator. Puts the variabls into a collection called
    'discriminator'.
    """
    initial_state, outputs, final_state = _recurrent_model(
        inputs, num_layers, width, batch_size, sequence_lengths,
        embedding_matrix, feed_previous=False)
    # now some feed forward layers on the final output
    layer_input = outputs[-1]
    for i, layer_size in enumerate(classifier_shape):
        layer_input = _ff_layer(layer_input, layer_size,
                                name='_ff_layer_{}'.format(i+1))
    return layer_input


def _ff_layer(in_var, size, nonlin=tf.nn.relu, name='layer', collections=None):
    """Gets a feed forward layer."""
    weights = tf.get_variable(name+'_W', shape=[in_var.get_shape()[1].value,
                                                size],
                              dtype=tf.float32)
    biases = tf.get_variable(name+'_b', shape=[size], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
    return nonlin(tf.nn.bias_add(tf.matmul(inputs, size), biases))


@new_collection('generator')
def generative_model(inputs, num_layers, width, embedding_matrix):
    """Gets the generative part of the model. Creates a sequence of from some
    kind of initialisation (probably noise). Puts the variables into
    a collections called 'generator'.
    Returns the list of output tensors for now, may need more in the future.
    """
    batch_size = inputs[0].get_shape()[0].value
    initial_state, outputs, final_state = _recurrent_model(
        inputs, num_layers, width, batch_size, None,
        embedding_matrix=embedding_matrix, feed_previous=True)
    return outputs


def _recurrent_model(inputs, num_layers, width,
                     batch_size, sequence_lengths,
                     embedding_matrix=None, feed_previous=True):
    """gets the recurrent part of a model

    Args:
        inputs: the inputs. either ints or float vectors.
        num_layers: how many layers the model should have
        width: how many units in each of the layers.
        batch_size: how many to do at a time.
        sequence_lengths: a batch_size vector if ints which indicates
          how long each sequence is, ignored if feed_previous is true.
    """
    cell = tf.nn.rnn_cell.LSTMCell(width)
    if num_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

    # get real inputs
    inputs = [tf.nn.embedding_lookup([embedding_matrix], input_)
              for input_ in inputs]
    initial_state = cell.zero_state(batch_size, tf.float32)
    if feed_previous:
        outputs, final_state = tf.nn.seq2seq.rnn_decoder(
            inputs, initial_state, cell,
            loop_function=lambda prev, i: tf.nn.embedding_lookup(
                [embedding_matrix],
                tf.argmax(prev, 1)))  # could sample?
    else:
        outputs, final_state = tf.nn.rnn(
            cell, inputs, initial_state=initial_state,
            sequence_length=sequence_lengths)  # use all the inputs.
    return initial_state, outputs, final_state


if __name__ == '__main__':
    import string
    # quick test
    batch_size = 10
    seq_len = 15
    num_symbols = len(string.ascii_lowercase)

    # make both nets the same for now
    num_layers = 2
    layer_width = 10

    # need some random integers
    noise_var = [tf.random_uniform(
        [batch_size], maxval=num_symbols, dtype=tf.int32)] * seq_len

    embedding = tf.get_variable('embedding', shape=[num_symbols, layer_width])
    generator_outputs = generative_model(
        noise_var, num_layers, layer_width, embedding)
    argmax_outputs = [tf.argmax(output, 1) for output in generator_outputs]
    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.initialize_all_variables())
        outs = sess.run(argmax_outputs)

        symbols = [string.ascii_lowercase[i[0]] for i in outs]
        print(''.join(symbols))
