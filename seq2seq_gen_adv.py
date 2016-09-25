"""A better way? We will see"""
import os
import numpy as np
import tensorflow as tf

import gen_adv


@gen_adv.new_collection('generator')
def generator(inputs, shape, start_number=0):
    """Gets a generator which is just an mlp.

    Args:
        inputs: probably random noise.
        shape: list of ints, describing the shape. The last layer is left
            linear and returned, so should be the same size as the embeddings
            we are copying.

    Returns:
        tensor: the final layer, with no non linearity.
    """
    for i, layer in enumerate(shape):
        inputs = gen_adv._ff_layer(inputs, layer,
                                      'generator-{}'.format(i + start_number))
        if i != len(shape) - 1:
            inputs = tf.nn.relu(inputs)

    return inputs


@gen_adv.new_collection('embedding')
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
        inputs = [tf.one_hot(input_) for input_ in inputs]

    # not sure why this is necessary
    batch_size = inputs[0].get_shape()[0].value

    initial_state, outputs, final_state = gen_adv._recurrent_model(
        inputs, num_layers, width, batch_size, None, #sequence_lengths,
        embedding_matrix=embedding_matrix, feed_previous=False, cell='gru')

    return final_state


@gen_adv.new_collection('embedding')
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
def discriminator(input_var, shape):
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
    for i, layer in enumerate(shape):
        input_var = gen_adv._ff_layer(input_var, layer,
                                      'discriminator-{}'.format(i))
        input_var = tf.nn.relu(input_var)

    logit = gen_adv._ff_layer(input_var, 1, 'discriminator-out')

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
    # return tf.nn.seq2seq.sequence_loss(sequence, target, weights)
    return tf.reduce_mean(
        tf.nn.seq2seq.sequence_loss_by_example(sequence, target, weights))


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
    import process_nips

    embedding_model_path = 'models/seq2seq/'
    train_embedding = True

    batch_size = 64
    embedding_size = 8  # very small

    num_epochs = 5000

    data_batch, length_batch = process_nips.get_nips_tensor(
        batch_size, 126, 'Title', num_epochs)
    vocab = process_nips.get_vocab()
    inv_vocab = {b: a for a, b in vocab.items()}
    vocab_size = len(vocab)
    max_sequence_length = 126

    num_layers = 1
    layer_width = 512

    disc_shape = [100, 25]

    gen_shape = [100, layer_width]
    noise_var = tf.random_normal([batch_size, 64])

    print('{:~^60}'.format('getting data stuff'), end='', flush=True)
    embedding = tf.get_variable('embedding',
                                shape=[vocab_size, embedding_size])

    # these are also the targets
    # input_pls = [tf.placeholder(
    #     tf.int32, [batch_size], name='input_{}'.format(i))
    #              for i in range(max_sequence_length)]
    # actual input to the models will be this reversed
    model_in = tf.unpack(tf.reverse(data_batch, [True, False]))
    # the weights for the loss are slightly awkward
    ranges = tf.pack([tf.range(max_sequence_length)] * batch_size)
    ranges = tf.transpose(ranges, [1, 0])
    weights = tf.select(ranges > length_batch,
                        tf.zeros_like(data_batch, tf.float32),
                        tf.ones_like(data_batch, tf.float32))
    weights = tf.unpack(weights)
    print('\r{:\\^60}'.format('got data stuff'))

    print('{:~^60}'.format('getting model'), end='', flush=True)
    with tf.variable_scope('generative') as scope:
        sequence_embedding = encoder(model_in, num_layers, layer_width,
                                     length_batch, embedding_matrix=embedding)
        generated_logits = decoder(
            tf.unpack(data_batch), sequence_embedding, num_layers,
            layer_width, embedding, vocab_size)
        generated_sequence = [tf.argmax(step, 1) for step in generated_logits]
        reconstruction_error = reconstruction_loss(
            generated_logits, tf.unpack(data_batch), weights)

        unsup_opt = tf.train.AdamOptimizer(0.001)
        unsup_train_op = unsup_opt.minimize(
            reconstruction_error, var_list=tf.get_collection('embedding'))

        # now get a feedforward generator which takes noise to a fake embedding
        fake_embeddings = generator(noise_var, gen_shape)

        scope.reuse_variables()
        fake_sequence = decoder(
            tf.unpack(data_batch), fake_embeddings, num_layers, layer_width,
            embedding, vocab_size)
        fake_sequence = [tf.argmax(step, 1) for step in fake_sequence]

    with tf.variable_scope('discriminative') as scope:
        disc_real = discriminator(sequence_embedding, disc_shape)

        scope.reuse_variables()
        disc_fake = discriminator(fake_embeddings, disc_shape)

        disc_loss = gen_adv.discriminator_loss(disc_fake, disc_real)

        disc_opt = tf.train.AdamOptimizer(0.1)
        disc_train_op = disc_opt.minimize(
            disc_loss, var_list=tf.get_collection('discriminator'))

    with tf.variable_scope('generative'):
        # the loss for the generator is how correct the discriminator was on
        # its batch.
        # (this could be the wrong way round, depends on class labels)
        gen_loss = -tf.reduce_mean(tf.log(1.0 - tf.nn.sigmoid(disc_fake)))
        gen_opt = tf.train.AdamOptimizer(0.01)
        gen_train_op = gen_opt.minimize(
            gen_loss, var_list=tf.get_collection('generator'))

    print('\r{:/^60}'.format('got model'))

    sess = tf.Session()

    embedding_saver = tf.train.Saver(var_list=tf.get_collection('embedding'),
                                     max_to_keep=1)

    print('{:~^60}'.format('initialising'), end='', flush=True)
    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())

    # check if we have an embedding model to start with
    if os.path.exists(embedding_model_path):
        model_path = tf.train.latest_checkpoint(embedding_model_path)
        embedding_saver.restore(sess, model_path)
    else:
        os.makedirs(embedding_model_path)

    print('\r{:\\^60}'.format('initialised'))

    widgets = ['(ﾉ◕ヮ◕)ﾉ* ',
               progressbar.AnimatedMarker(markers='←↖↑↗→↘↓↙'),
               progressbar.Bar(marker='/',
                               left='-',
                               fill='~'),
               ' (', progressbar.DynamicMessage('loss'), ')'
               ' (', progressbar.AdaptiveETA(), ')']
    max_steps = (403 // batch_size) * num_epochs
    bar = progressbar.ProgressBar(widgets=widgets, redirect_stdout=True,
                                  max_value=max_steps)
    print(max_steps)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        step = 0
        while not coord.should_stop():
            if train_embedding:
                batch_error, _, last_batch, last_target = sess.run(
                    [reconstruction_error, unsup_train_op, generated_sequence,
                     data_batch])
                # have a look maybe?
                if (step % 1000) == 0:
                    embedding_saver.save(
                        sess, embedding_model_path+'gru_encdec',
                        global_step=step, write_meta_graph=False)
                    print('Step {}, unsupervised reconstruction error: {}'.format(
                        step, batch_error))
                    test_index = np.random.randint(batch_size)
                    print(''.join([inv_vocab[step[test_index]]
                                   for step in last_target[:, ...]]))
                    print(' -->')
                    print(''.join([inv_vocab[step[test_index]]
                                   for step in last_batch]))
                    # if (epoch_loss / epoch_steps) <= 0.1:
                    #     bar.finish()
                    #     print('Happy with the embeddings because threshold.')
                    #     break
                bar.update(step, loss=batch_error)
                step += 1
            else:  # train the generative adversarial part
                print('gen-adv training')
                bar = progressbar.ProgressBar(
                    widgets=widgets, redirect_stdout=True)
                disc_error, gen_error, _, _ = sess.run(
                    [disc_loss, gen_loss, disc_train_op, gen_train_op])

                if (step % 250) == 1:
                    print('Step {}'.format(step))
                    print('~~Discriminator loss: {}'.format(disc_error))
                    print('~~Generator     loss: {}'.format(gen_error))
                    new_seqs = sess.run(fake_sequence)
                    print(''.join([inv_vocab[step[0]] for step in new_seqs]))
    except tf.errors.OutOfRangeError:
        print('Out of data now')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    tf.app.run()
