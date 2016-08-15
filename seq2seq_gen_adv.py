"""A better way? We will see"""
import numpy as np
import tensorflow as tf

import gen_adv


@gen_adv.new_collection('generator')
def encoder(inputs, num_layers, width, sequence_lengths,
            embedding_matrix=None):
    """Gets an encoder, which maps a sequence to a fixed length vector
    representation.

    Args:
        inputs: list of [batch_size] int tensors, we will either turn them
            to one hots or embed them.
        num_layers: how many layers of recurrent cells.
        width: how many recurrent cells in each layer.
        sequence_lengths: [batch_size] int tensor indicating the actual length
            of the sequences passed in.
        embedding_matrix (optional): float matrix to look up embeddings in,
            if none the input is converted to a one hot encoding.

    Returns:
        state: the final state of the network. If you're using more than one
            layer (or using LSTMs) then this might be a tuple, it's up to you
            how to deal with it (but probably either concatenate or just take
            the last element).
    """
    if embedding_matrix is None:
        inputs = [tf.nn.one_hot(input_) for input_ in inputs]

    # not sure why this is necessary
    batch_size = inputs[0].get_shape()[0].value

    initial_state, outputs, final_state =gen_adv._recurrent_model(
        inputs, num_layers, width, batch_size, sequence_lengths,
        embedding_matrix=embedding_matrix, feed_previous=False, cell='gru')

    return final_state


@gen_adv.new_collection('generator')
def decoder(inputs, initial_state, num_layers, width, embedding_matrix,
            vocab_size):
    """Get a decoder, which attends to a single input (probably the GO symbol)
    and is started off in a specific state. It will then produce a sequence.

    Args:
        inputs: list of inputs, only the first is used but the length of the
            list defines the maximum length of the sequences.
        initial_state: the starting state of the net.
        num_layers: number of recurrent layers.
        width: width of recurrent layers.
        embedding_matrix: matrix containing symbol embeddings (for input).
        vocab_size: the number of possible output symbols. We could probably
            just figure this out from the embedding matrix.

    Returns:
        outputs: the projected outputs of the net.
    """
    # we are going to have to project it to vocab_size
    with tf.variable_scope('softmax_projection'):
        proj_W = tf.get_variable('weights', shape=[width, vocab_size])
        proj_b = tf.get_variable('bias', shape=[vocab_size])
    batch_size = inputs[0].get_shape()[0].value
    _, outputs, _ = gen_adv._recurrent_model(
        inputs, num_layers, width, batch_size, None,
        embedding_matrix=embedding_matrix, feed_previous=True,
        argmax=True, starting_state=initial_state,
        output_projection=(proj_W, proj_b))
    return outputs[0]


@gen_adv.new_collection('discriminator')
def discrimator(input_var, shape):
    """Gets the discriminator, which is a feed forward MLP with len(shape)
    layers, in which the i-th layer has layers[i] units. Internal layers use
    ReLU nonlinearities, the raw logit is returned for the output.

    Args:
        input: the input tensor.
        shape: list of ints, describing the size of the hiden layers.

    Returns:
        tensor: the logit, representing prediction re. whether the input was
            generated from the data or not.
    """
    for i, layer in enumerate(layers):
        input_var = gen_adv._ff_layer(input_var, layer,
                                      'discriminator-{}'.format(i))
        input_var = tf.nn.relu(input_var)

    logit = _ff_layer(input_var, 1, 'discriminator-out')

    return logit


def reconstruction_loss(sequence, target, weights):
    """Gets a reconstruction loss, which is the average softmax cross-entropy
    per timestep between the tensors in sequence and the tensors in target.

    Args:
        sequence: list of [batch_size, vocab_size] tensors containing the
            outputs of the net.
        target: list of [batch_size] int tensors with the real symbols.
        weights: weights for the loss. We really just expect this to be
            0 after the target sequence is finished, otherwise 1.

    Returns:
        scalar tensor, the average loss.
    """
    return tf.nn.seq2seq.sequence_loss(sequence, target, weights)


def main(_):
    # for now, let's just ensure the unsupervised training for the generator
    # does something
    import string
    import progressbar
    import data

    batch_size = 32
    data, lengths, vocab = data.get_data()
    vocab_size = len(vocab)
    max_sequence_length = np.max(lengths)
    embedding_size = 64

    num_epochs = 5000

    num_layers = 1
    layer_width = 100

    disc_shape = [500, 25]

    embedding = tf.get_variable('embedding',
                                shape=[vocab_size, embedding_size])


    with tf.variable_scope('generative'):
        sequence_embedding = encoder()


if __name__ == '__main__':
    tf.app.run()
