"""generative adversarial?

On sequences. The challenge is:
    we want to sample a sequence according to the generator (probably)
    this is going to be hard to backpropagate
        solutions a) treat is as RL, give the generator an appropriate reward
                     back it up like a policy gradient
                  b) come up with some continuous letter representation which
                     we can pass through directly.
                  c) make the discriminator predict at each time step, then
                     just roll straight through
    for now we are going with a).
    This is kind of not very good though --
    c would be a lot better (the biggest issue is that policy gradient is
    very slow, but we could train faster because we probably could resolve
    the temporal credit assignment).
"""
import numpy as np
import tensorflow as tf

import data

import mrnn

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
            return retval
        return _wrap


@new_collection('discriminator')
def discriminative_model(inputs, num_layers, width, classifier_shape,
                         embedding_matrix, sequence_lengths, project_to=-1,
                         seq_predictions=False):
    """Gets the discriminator. Puts the variables into a collection called
    'discriminator'. If embedding matrix is none, we assume the inputs are
    floats.
    """
    with tf.name_scope('discriminator'):
        if project_to > 0:
            projection_matrix = tf.get_variable('projection_weights',
                                                shape=[width, project_to])
            projection_bias = tf.get_variable(
                'projection_bias', shape=[project_to],
                initializer=tf.constant_initializer(0.0))
            projection = (projection_matrix, projection_bias)
        else:
            projection = None
        initial_state, outputs, final_state = _recurrent_model(
            inputs, num_layers, width, batch_size, sequence_lengths,
            embedding_matrix, feed_previous=False,
            output_projection=projection)
        # now some feed forward layers
        if seq_predictions:
            inputs = outputs
            pred_outs = []
        else:
            inputs = outputs[-1:]
            pred_outs = None
        for step_input in inputs:
            layer_input = step_input
            for i, layer_size in enumerate(classifier_shape):
                with tf.variable_scope('ff',
                                       reuse=step_input != inputs[0]):
                    layer_input = _ff_layer(layer_input, layer_size,
                                            name='_ff_layer_{}'.format(i+1))
                    if i < len(classifier_shape)-1:
                        layer_input = tf.nn.relu(layer_input)
            if pred_outs is not None:
                pred_outs.append(layer_input)
    return pred_outs if pred_outs else outputs, layer_input


def _ff_layer(in_var, size, name='layer', collections=None):
    """Gets a feed forward layer."""
    weights = tf.get_variable(name+'_W', shape=[in_var.get_shape()[1].value,
                                                size],
                              dtype=tf.float32)
    biases = tf.get_variable(name+'_b', shape=[size], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
    return tf.nn.bias_add(tf.matmul(in_var, weights), biases)


@new_collection('generator')
def generative_model(inputs, num_layers, width, embedding_matrix, num_outs,
                     feed_previous=True, argmax=False, epsilon=0.0):
    """Gets the generative part of the model. Creates a sequence of from some
    kind of initialisation (probably noise). Puts the variables into
    a collections called 'generator'.
    Returns:
        (outputs, samples): the projected outputs (unnormalised log
            probabilities) and the samples used to feed back into the network.
    """
    with tf.name_scope('generator'):
        batch_size = inputs[0].get_shape()[0].value

        # we are going to need to project the outputs into the appropriate
        # space
        proj_w = tf.get_variable(
            'output_weights', [width, num_outs], trainable=True)
        proj_b = tf.get_variable(
            'output_biases', [num_outs], trainable=True,
            initializer=tf.constant_initializer(0.0))
        initial_state, outputs, final_state = _recurrent_model(
            inputs, num_layers, width, batch_size, None,
            embedding_matrix=embedding_matrix, feed_previous=feed_previous,
            output_projection=(proj_w, proj_b), argmax=argmax,
            epsilon=epsilon)
    return outputs


def _recurrent_model(inputs, num_layers, width,
                     batch_size, sequence_lengths,
                     embedding_matrix=None, feed_previous=True,
                     output_projection=None, argmax=False,
                     epsilon=0.0, cell='lstm', starting_state=None):
    """gets the recurrent part of a model

    Args:
        inputs: the inputs. either ints or float vectors.
        num_layers: how many layers the model should have
        width: how many units in each of the layers.
        batch_size: how many to do at a time.
        sequence_lengths: a batch_size vector if ints which indicates
          how long each sequence is, ignored if feed_previous is true.
        cell: the RNN cell to actually use. Default is `lstm`, `gru` or
          `vanilla` are also supported.
        starting_state: the first state of the network, if you need to care
          about it (eg. when decoding).

    Returns:
        initial_state, outputs, final_state. If feed_previous is true,
            output is in fact a tuple of (outputs, samples) where `samples`
            are the samples drawn from `outputs` and fed back in. The other
            outputs will have been projected if a projection is present.
    """
    if cell == 'lstm':
        cell = tf.nn.rnn_cell.LSTMCell(width, state_is_tuple=True)
    elif cell == 'gru':
        cell = tf.nn.rnn_cell.GRUCell(width)
    elif cell == 'vanilla':
        cell = tf.nn.rnn_cell.BasicRNNCell(width)
    else:
        raise ValueError('Unknown cell: {}'.format(cell))

    if num_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers,
                                           state_is_tuple=True)

    # get real inputs
    if embedding_matrix is not None:
        inputs = [tf.nn.embedding_lookup([embedding_matrix], input_)
                  for input_ in inputs]
    if starting_state is None:
        initial_state = cell.zero_state(batch_size, tf.float32)
    else:
        initial_state = starting_state
    if feed_previous:
        sampled_outputs = []
        if output_projection:
            projected_outputs = []
            num_outs = output_projection[1].get_shape()[0].value
        else:
            num_outs = width
        rando = tf.random_uniform([])

        def loop_fn(prev, i):
            if output_projection:
                prev = tf.nn.bias_add(tf.matmul(prev, output_projection[0]),
                                      output_projection[1])
                projected_outputs.append(prev)
            if argmax:
                sample = tf.cast(tf.argmax(prev, 1), tf.int32)
            else:
                sample = tf.cast(tf.squeeze(tf.multinomial(prev, 1)), tf.int32)
                if epsilon > 0.0:
                    sample = tf.cond(rando > tf.constant(epsilon),
                                     lambda: sample,
                                     lambda: tf.random_uniform(
                            [batch_size], maxval=num_outs,
                                        dtype=tf.int32))
            sampled_outputs.append(sample)
            return tf.nn.embedding_lookup(
                [embedding_matrix],
                sample)

        outputs, final_state = tf.nn.seq2seq.rnn_decoder(
            inputs, initial_state, cell,
            loop_function=loop_fn)
        if output_projection:  # still need to do the last one
            projected_outputs.append(
                tf.nn.bias_add(
                    tf.matmul(outputs[-1], output_projection[0]),
                    output_projection[1]))
            outputs = projected_outputs
        outputs = (outputs, sampled_outputs)
    else:
        outputs, final_state = tf.nn.rnn(
            cell, inputs, initial_state=initial_state,
            sequence_length=sequence_lengths)  # use all the inputs.
        if output_projection:
            outputs = [tf.nn.bias_add(
                tf.matmul(step, output_projection[0]),
                output_projection[1])
                       for step in outputs]
    return initial_state, outputs, final_state


def sample_outputs(all_logits):
    """Samples from a list batches of logits (eg the output of an RNN).

    Args:
        all_logits: list of `[batch_size x num_classes]` float tensors,
            assumed to represent unnormalised log probabilities.

    Returns:
        list of `[batch_size]` int tensors with the appropriate samples.
    """
    return [tf.cast(tf.squeeze(tf.multinomial(step, 1)), tf.int32)
            for step in all_logits]


def advantage(logits, choices, rewards):
    """Essentially an advantage function for the policy gradient learning.
    This boils down to the sum of rewards (no discounting here) weighted by log
    probabilities. This gives us a quantity we want to maximise, so we return
    the negative (ie the reward weighted by the nll).

    We are assuming a higher positive reward is good. This means if you pass in
    the logit from the discriminator as reward, make sure it is the right way
    round.

    Args:
        logits: list of `[batch_size x num_classes]` raw generator outputs.
        choices: list of `[batch_size]` integers indicating what happened.
        rewards: `[batch_size x 1]` rewards. Higher is better.

    Returns:
        scalar - the average advantage for this batch. Because there is no
            discounting, this boils down to elementwise multiplying the rewards
            with the gathered log likelihoods, taking the mean and returned it
            negated.
    """
    # possibly be more efficient to concatenate?
    log_probs = [tf.nn.log_softmax(step) for step in logits]
    # now we have a list of the log probs
    # we want to somehow gather the ones in choices at each step
    # this is bit gross
    batch_size, num_classes = log_probs[0].get_shape().as_list()
    lls = []
    flat_idx = tf.range(0, batch_size) * num_classes
    for action, likelihood in zip(choices, log_probs):
        batch_probs = tf.gather(tf.reshape(likelihood, [-1]),
                                flat_idx + action)
        lls.append(batch_probs)
    return -tf.reduce_sum(tf.pack([ll * rewards for ll in lls]))


def step_advantage(logits, choices, rewards):
    """Gets advantage when rewards is in fact a sequence of rewards. Uses the
    same method otherwise (no discounting)."""
    log_probs = [tf.nn.log_softmax(step) for step in logits]
    batch_size, num_classes = log_probs[0].get_shape().as_list()
    lls = []
    flat_idx = tf.range(0, batch_size) * num_classes
    for action, likelihood in zip(choices, log_probs):
        batch_probs = tf.gather(tf.reshape(likelihood, [-1]),
                                flat_idx + action)
        lls.append(batch_probs)

    return -tf.reduce_mean(tf.pack([ll * reward
                                   for ll, reward in zip(lls, rewards)]))


def softmax_weighted_sum(logits, embedding):
    """Produces a blurred embedding vector. Or rather, a list of batches of
    them"""
    return [tf.matmul(tf.nn.softmax(logit), embedding) for logit in logits]


def feature_matching_loss(discriminator_fake, discriminator_real):
    """Gets the average squared l2 between the discriminator's activations
    on real data and fake data. The idea is then if we use this to train
    the generator, maybe life will be good again? Note that in order for
    the gradients to propagate back, we will have to be feeding the
    discriminator with something that isn't sampled.
    """
    diffs = [tf.squared_difference(real, fake)
             for real,fake in zip(discriminator_real,
                                  discriminator_fake)]
    return diffs # tf.reduce_mean(diffs)


def discriminator_loss(generative_batch, discriminative_batch):
    """Gets the average cross entropy, assuming that all of those in
    `generative batch` should have label 0 and `discriminative_batch` label 1.
    This should work with the advantage the way it is defined above -- the
    generator is trying to make the discriminator assign its samples high
    scores."""
    # labels
    batch_size = generative_batch.get_shape()[0].value
    discriminator_labels = tf.constant(
            np.vstack((np.ones((batch_size, 1), dtype=np.float32),
                       np.zeros((batch_size, 1), dtype=np.float32))))
    big_batch = tf.concat(0, [generative_batch, discriminative_batch])
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        big_batch, discriminator_labels))


def seq_discriminator_loss(generative_batch, discriminative_batch):
    """A per discriminative loss, but for when the batches are time-major lists
    """
    labels = [tf.constant(
            np.vstack((np.ones((batch_size, 1), dtype=np.float32),
                       np.zeros((batch_size, 1), dtype=np.float32))))
              for _ in generative_batch]
    big_batches = [tf.concat(0, [gen_batch, disc_batch])
                   for gen_batch, disc_batch in
                   zip(generative_batch, discriminative_batch)]
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        tf.pack(big_batches), tf.pack(labels)))


def get_train_step(g_loss, d_loss, global_step=None, generator_freq=1):
    """Gets (and groups) training ops for the two models.
    Can set it up so the generator trains a bunch of times before the
    discriminator is updated.
    """
    if global_step is None:
        global_step = tf.Variable(0, name='global_step', dtype=tf.int32,
                                  trainable=False)

    g_opt = tf.train.MomentumOptimizer(0.01, 0.9)
    # g_loss += tf.add_n([0.0001 * tf.nn.l2_loss(var)
    #                     for var in tf.get_collection('generator')])
    d_opt = tf.train.MomentumOptimizer(0.01, 0.9)
    if generator_freq > 1:  # g_step is actually a lot of them
        return tf.cond(
            tf.equal((global_step % generator_freq), 0),
            lambda: d_opt.minimize(
                d_loss, var_list=tf.get_collection('discriminator'),
                gate_gradients=0, global_step=global_step),
            lambda: g_opt.minimize(
                g_loss, var_list=tf.get_collection('generator'),
                gate_gradients=0, global_step=global_step))
    else:
        g_step = g_opt.minimize(
            g_loss, var_list=tf.get_collection('generator'),
            gate_gradients=0, global_step=global_step)

        d_step = d_opt.minimize(
            d_loss, var_list=tf.get_collection('discriminator'),
            gate_gradients=0, global_step=global_step)

        return g_step, d_step


if __name__ == '__main__':
    import string
    import random
    import progressbar
    # quick test
    batch_size = 128
    seq_len = 25
    vocab = data.get_default_symbols()
    num_symbols = len(vocab)
    num_epochs = 500000

    real_data = data.get_batch_tensor(batch_size, seq_len, num_epochs)

    # make both nets the same for now
    num_layers = 1
    layer_width = 100

    # need some random integers
    noise_var = [tf.random_uniform(
        [batch_size], maxval=num_symbols, dtype=tf.int32)] * seq_len
    embedding = tf.get_variable('embedding', shape=[num_symbols, layer_width],
                                collections=['discriminator',
                                             tf.GraphKeys.VARIABLES])

    with tf.variable_scope('Generative'):
        generator_outputs, sampled_outs = generative_model(
            noise_var, num_layers, layer_width, embedding,
            num_symbols, argmax=False, epsilon=0.0)
        # generator_outputs = generative_model(
        #     noise_var, num_layers, layer_width, embedding,
        #     num_symbols, feed_previous=False)
        # get these to have a look at
        # sampled_outs = sample_outputs(generator_outputs)
        # and get something blurry to feed into the discriminator
        # combined_outs = softmax_weighted_sum(generator_outputs, embedding)

    with tf.variable_scope('Discriminative') as scope:
        # first get the output of the discriminator run on the generator's out
        g_acts, discriminator_g = discriminative_model(
            sampled_outs, num_layers, layer_width, [1], embedding, None,
            project_to=0, seq_predictions=True)
        # g_acts, discriminator_g = discriminative_model(
        #     combined_outs, num_layers, layer_width, [32, 1], None, None,
        #     project_to=num_symbols)
        scope.reuse_variables()
        # get the same model, but with the actual data as inputs
        d_acts, discriminator_d = discriminative_model(
            real_data, num_layers, layer_width, [1], embedding, None,
            project_to=0, seq_predictions=True)
        # discriminator_g = tf.Print(discriminator_g, [discriminator_g[0, 0],
        #                                              discriminator_d[0, 0]])

    with tf.variable_scope('training') as scope:
        # generator_loss = advantage(generator_outputs, sampled_outs,
        #                            (2.0 * tf.nn.sigmoid(discriminator_g)) - 1.0)
        # generator_loss = feature_matching_loss(d_acts, g_acts)
        generator_loss = step_advantage(generator_outputs, sampled_outs,
                                        [(2.0 * tf.nn.sigmoid(acts)) - 1.0
                                         for acts in g_acts])
        # discriminator_loss = discriminator_loss(discriminator_g,
        #                                         discriminator_d)
        discriminator_loss = seq_discriminator_loss(g_acts, d_acts)
        g_train, d_train = get_train_step(
            generator_loss, discriminator_loss,
            generator_freq=1)

    # finally we can do stuff
    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.initialize_all_variables())
        widgets = ['(ﾉ◕ヮ◕)ﾉ* ',
                   progressbar.AnimatedMarker(markers='←↖↑↗→↘↓↙'),
                   progressbar.Bar(marker='-',
                                   left='-',
                                   fill='/'),
                   ' (', progressbar.AdaptiveETA(), ') ']

        bar = progressbar.ProgressBar(widgets=widgets, redirect_stdout=True,
                                      max_value=num_epochs)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        d_loss = 0.69
        try:
            step = 0
            bar.start()
            g_trains, d_trains = 0, 0
            while (not coord.should_stop()) and (step < num_epochs):
                if step % 2 == 0 or d_loss < 0.1:
                    sess.run(g_train)
                    g_trains += 1
                else:
                    d_loss, _ = sess.run([discriminator_loss, d_train])
                    d_trains += 1

                # sess.run(train_step)
                if (step+1) % 200 == 0:
                    outs = sess.run(sampled_outs +
                                    [generator_loss, discriminator_loss])
                    print('{:~<30}'.format(step+1))
                    symbols = [[vocab[step[i]]
                                for step in outs[:-3]]
                               for i in range(10)]
                    print('\n'.join([''.join(row) for row in symbols]))
                    print('Generator loss     : {}'.format(outs[-2]))
                    print('Discriminator loss : {}'.format(outs[-1]))
                    print('Training ratio     : {}/{} (g/d)'.format(
                            g_trains, d_trains))
                    d_trains, g_trains = 0, 0
                    d_loss = outs[-1]
                bar.update(step)
                step += 1
        except tf.errors.OutOfRangeError:
            bar.finish()
            print('Done. ({} steps)'.format(step))
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()
