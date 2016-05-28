"""generative adversarial?"""
import numpy as np
import tensorflow as tf


def discriminative_model(inputs, num_layers, width, classifier_shape,
                         embedding_matrix, sequence_lengths):
    """Gets the discriminator"""
    initial_state, outputs, final_state = _recurrent_model(
        inputs, num_layers, width, batch_size, sequence_lengths,
        embedding_matrix, feed_previous=False)
    # now some feed forward layers on the final output
    layer_input = outputs[-1]
    for i, layer_size in enumerate(classifier_shape):
        layer_input = _ff_layer(layer_input, layer_size,
                                name='_ff_layer_{}'.format(i+1))
    return layer_input


def _ff_layer(in_var, size, nonlin=tf.nn.relu, name='layer'):
    """Gets a feed forward layer."""
    weights = tf.get_variable(name+'_W', shape=[in_var.get_shape()[1].value,
                                                size],
                              dtype=tf.float32)
    biases = tf.get_variable(name+'_b', shape=[size], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
    return nonlin(tf.nn.bias_add(tf.matmul(inputs, size), biases))


def generative_model(inputs, num_layers, width, embedding_matrix):
    """Gets the generative part of the model. Creates a sequence of from some
    kind of initialisation (probably noise).
    Returns the list of output tensors.
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
    cell = tf.nn.rnn_cell.BasicLSTMCell(width)
    if num_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

    # get real inputs

    inputs = [tf.nn.embedding_lookup(embedding_matrix, input_)
              for input_ in inputs]
    initial_state = cell.zero_state(batch_size)
    if feed_previous:
        outputs, final_state = tf.nn.seq2seq.rnn_decoder(
            inputs, initial_state, cell,
            loop_function=lambda prev, i: tf.argmax(prev))  # could sample?
    else:
        outputs, final_state = tf.nn.rnn(
            cell, inputs, initial_state=initial_state,
            sequence_length=sequence_lengths)  # use all the inputs.
    return initial_state, outputs, final_state


if __name__ == '__main__':
    real_seq_inputs = [tf.placeholder(tf.int64)]
