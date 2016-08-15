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

    initial_state, outputs, final_state = gen_adv._recurrent_model(
        inputs, num_layers, width, batch_size, None, #sequence_lengths,
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
        output_projection=(proj_W, proj_b), cell='gru')
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


def fill_feed(input_vars, input_data, length_var, lengths, weight_vars):
    """fills up a feed dict. Note that input_data will be the wrong way around
    """
    input_data = input_data.T
    feed = {}
    counter = 0
    for in_var, step, weight_var in zip(input_vars, input_data[:, ...],
                                        weight_vars):
        feed[in_var] = step
        feed[weight_var] = np.array([1.0 if counter < lengths[i] else 0.0
                                     for i in range(lengths.shape[0])],
                                    dtype=np.float32)
        counter += 1
    feed[length_var] = lengths

    return feed


def main(_):
    # for now, let's just ensure the unsupervised training for the generator
    # does something
    import string
    import progressbar
    import data

    batch_size = 32
    np_data, lengths, vocab = data.get_data()
    inv_vocab = {b: a for a, b in vocab.items()}
    vocab_size = len(vocab)
    max_sequence_length = np.max(lengths)
    embedding_size = 64

    num_epochs = 5000

    num_layers = 1
    layer_width = 100

    disc_shape = [500, 25]

    print('{:~^60}'.format('getting data stuff'), end='', flush=True)
    embedding = tf.get_variable('embedding',
                                shape=[vocab_size, embedding_size])

    # these are also the targets
    input_pls = [tf.placeholder(
        tf.int32, [batch_size], name='input_{}'.format(i))
                 for i in range(max_sequence_length)]
    # actual input to the models will be this reversed
    model_in = tf.unpack(tf.reverse(tf.pack(input_pls), [True, False]))
    length_pl = tf.placeholder(tf.int32, [batch_size])
    weights_pls = [tf.placeholder(
        tf.float32, [batch_size], name='weight_{}'.format(i))
                   for i in range(max_sequence_length)]
    print('\r{:\\^60}'.format('got data stuff'))

    print('{:~^60}'.format('getting model'), end='', flush=True)
    with tf.variable_scope('generative'):
        sequence_embedding = encoder(model_in, num_layers, layer_width,
                                     input_pls, embedding_matrix=embedding)
        generated_logits = decoder(input_pls, sequence_embedding, num_layers,
                                   layer_width, embedding, vocab_size)
        generated_sequence = [tf.argmax(step, 1) for step in generated_logits]
        reconstruction_error = reconstruction_loss(generated_logits, input_pls,
                                                   weights_pls)

        unsup_opt = tf.train.AdamOptimizer(0.001)
        unsup_train_op = unsup_opt.minimize(reconstruction_error)
    print('\r{:/^60}'.format('got model'))

    sess = tf.Session()

    print('{:~^60}'.format('initialising'), end='', flush=True)
    sess.run(tf.initialize_all_variables())
    print('\r{:\\^60}'.format('initialised'))

    widgets = ['(ﾉ◕ヮ◕)ﾉ* ',
               progressbar.AnimatedMarker(markers='←↖↑↗→↘↓↙'),
               progressbar.Bar(marker='-',
                               left='-',
                               fill='/'),
               ' (', progressbar.AdaptiveETA(), ') ']

    bar = progressbar.ProgressBar(widgets=widgets, redirect_stdout=True,
                                  max_value=num_epochs)
    for epoch in bar(range(num_epochs)):
        epoch_loss = 0
        epoch_steps = 0
        for batch_data, batch_lengths in data.iterate_batches(np_data,
                                                              lengths,
                                                              batch_size):
            feed = fill_feed(input_pls, batch_data, length_pl, batch_lengths,
                             weights_pls)
            batch_error, _ = sess.run([reconstruction_error, unsup_train_op],
                                      feed_dict=feed)
            epoch_loss += batch_error
            epoch_steps += 1
        # have a look maybe?
        print('Epoch {}, unsupervised reconstruction error: {}'.format(
            epoch+1, epoch_loss/epoch_steps))
        last_batch = sess.run(generated_sequence, feed_dict=feed)
        test_index = np.random.randint(batch_size)
        print(''.join([inv_vocab[step[test_index]]
                       for step in batch_data.T[:, ...]]))
        print(' -->')
        print(''.join([inv_vocab[step[test_index]]
                       for step in last_batch]))



if __name__ == '__main__':
    tf.app.run()
