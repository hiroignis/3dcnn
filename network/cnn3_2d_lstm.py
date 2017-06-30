import numpy as np
import tensorflow as tf

import parameter


FC_LAYER_SIZE = 256
DATA_TYPE = tf.float32


def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """

    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

    return


def dropout_layer(layer, keep_prob):
    tf.summary.scalar('dropout_keep_probability', keep_prob)

    return tf.nn.dropout(layer, keep_prob)


def inference(box_placeholder, keep_prob, lstm_memory_size):

    def _weight_variable(shape):

        if parameter.RESTORE_MODE:
            init = tf.truncated_normal(shape=shape, dtype=DATA_TYPE, stddev=0.1)
            w_var = tf.Variable(init)
        else:
            w_var = tf.get_variable(name='weight', shape=shape, dtype=DATA_TYPE,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        return w_var

    def _bias_variable(shape):
        if parameter.RESTORE_MODE:
            init = tf.constant(0.1, shape=shape, dtype=DATA_TYPE)
            b_var = tf.Variable(init)
        else:
            b_var = tf.get_variable(name='bias', shape=shape, dtype=DATA_TYPE,
                                    initializer=tf.constant_initializer(0.1, dtype=DATA_TYPE))
        return b_var

    input_layer = box_placeholder
    in_channels = parameter.CHANNEL_SCALE

    with tf.variable_scope('conv1') as scope:
        out_channels = parameter.DATA_SEQUENCE_LENGTH * 3

        with tf.name_scope('weights'):
            weight_filter = _weight_variable([1, 4, 4, in_channels, out_channels])
            variable_summaries(weight_filter)

        conv = tf.nn.conv3d(input=input_layer,
                            filter=weight_filter,
                            strides=[1, 1, 2, 2, 1],
                            padding='VALID')

        with tf.name_scope('biases'):
            biases = _bias_variable([out_channels])
            variable_summaries(biases)

        with tf.name_scope('conv_plus_b'):
            units = tf.nn.bias_add(value=conv, bias=biases)
            tf.summary.histogram('conv_filter', units)

        conv1 = tf.nn.relu(features=units)
        tf.summary.histogram('activated_units', conv1)

    input_layer = conv1
    in_channels = out_channels

    with tf.variable_scope('pool1') as scope:
        pool1 = tf.nn.max_pool3d(input=input_layer,
                                 ksize=[1, 1, 3, 3, 1],
                                 strides=[1, 1, 2, 2, 1],
                                 padding='VALID')
        norm1 = pool1

    input_layer = norm1

    with tf.variable_scope('conv2') as scope:
        out_channels = parameter.DATA_SEQUENCE_LENGTH * 12

        with tf.name_scope('weights'):
            weight_filter = _weight_variable([1, 3, 3, in_channels, out_channels])
            variable_summaries(weight_filter)

        conv = tf.nn.conv3d(input=input_layer,
                            filter=weight_filter,
                            strides=[1, 1, 2, 2, 1],
                            padding='VALID')

        with tf.name_scope('biases'):
            biases = _bias_variable([out_channels])
            variable_summaries(biases)

        with tf.name_scope('conv_plus_b'):
            units = tf.nn.bias_add(value=conv, bias=biases)
            tf.summary.histogram('conv_filter', units)

        conv2 = tf.nn.relu(features=units)
        tf.summary.histogram('activated_units', conv2)

    input_layer = conv2
    in_channels = out_channels

    with tf.variable_scope('pool2') as scope:
        pool2 = tf.nn.max_pool3d(input=input_layer,
                                 ksize=[1, 1, 3, 3, 1],
                                 strides=[1, 1, 2, 2, 1],
                                 padding='VALID')
        norm2 = pool2

    input_layer = norm2

    with tf.variable_scope('conv3') as scope:
        out_channels = parameter.DATA_SEQUENCE_LENGTH * 24

        with tf.name_scope('weights'):
            weight_filter = _weight_variable([1, 3, 3, in_channels, out_channels])
            variable_summaries(weight_filter)

        conv = tf.nn.conv3d(input=input_layer,
                            filter=weight_filter,
                            strides=[1, 1, 2, 2, 1],
                            padding='VALID')

        with tf.name_scope('biases'):
            biases = _bias_variable([out_channels])
            variable_summaries(biases)

        with tf.name_scope('conv_plus_b'):
            units = tf.nn.bias_add(conv, biases)
            tf.summary.histogram('conv_filter', units)

        conv3 = tf.nn.relu(features=units)
        tf.summary.histogram('activated_units', conv3)

    input_layer = conv3
    in_channels = out_channels

    with tf.variable_scope('conv4') as scope:
        out_channels = parameter.DATA_SEQUENCE_LENGTH * 64

        with tf.name_scope('weights'):
            weight_filter = _weight_variable([1, 3, 3, in_channels, out_channels])
            variable_summaries(weight_filter)

        conv = tf.nn.conv3d(input=input_layer,
                            filter=weight_filter,
                            strides=[1, 1, 2, 2, 1],
                            padding='VALID')

        with tf.name_scope('biases'):
            biases = _bias_variable([out_channels])
            variable_summaries(biases)

        with tf.name_scope('conv_plus_b'):
            units = tf.nn.bias_add(conv, biases)
            tf.summary.histogram('conv_filter', units)

        conv4 = tf.nn.relu(features=units)
        tf.summary.histogram('activated_units', conv4)

    input_layer = conv4

    with tf.variable_scope('lstm5') as scope:
        dimension = np.prod(input_layer.get_shape().as_list()[1:])
        input_layer_flat = tf.reshape(input_layer, [-1, dimension])
        input_layer_split = tf.split(input_layer_flat, parameter.DATA_SEQUENCE_LENGTH, 1)
        # input_layer_split = tf.split(0, parameter.DATA_SEQUENCE_LENGTH, input_layer_flat) # version <= 0.12.0

        cell = tf.contrib.rnn.BasicLSTMCell(FC_LAYER_SIZE, forget_bias=0.8, state_is_tuple=False)
        lstm5, _ = tf.contrib.rnn.static_rnn(cell,
                                             input_layer_split,
                                             initial_state=cell.zero_state(lstm_memory_size, tf.float32))

    input_layer = lstm5[-1]

    with tf.variable_scope('fc6') as scope:
        dimension = np.prod(input_layer.get_shape().as_list()[1:])
        input_layer_flat = tf.reshape(input_layer, [-1, dimension])

        with tf.name_scope('weights'):
            weights = _weight_variable([dimension, FC_LAYER_SIZE])
            variable_summaries(weights)

        with tf.name_scope('biases'):
            biases = _bias_variable([FC_LAYER_SIZE])
            variable_summaries(biases)

        with tf.name_scope('wx_plus_b'):
            units = tf.matmul(input_layer_flat, weights) + biases
            tf.summary.histogram('pre_activated_units', units)

        fc6 = tf.nn.relu(features=units)
        # set dropout
        fc6_dropout = tf.nn.dropout(fc6, keep_prob)

    input_layer = fc6_dropout

    with tf.variable_scope('fc7') as scope:
        dimension = np.prod(input_layer.get_shape().as_list()[1:])
        input_layer_flat = tf.reshape(input_layer, [-1, dimension])

        with tf.name_scope('weights'):
            weights = _weight_variable([dimension, parameter.NUM_OUTPUTS])
            variable_summaries(weights)

        with tf.name_scope('biases'):
            biases = _bias_variable([parameter.NUM_OUTPUTS])
            variable_summaries(biases)

        with tf.name_scope('wx_plus_b'):
            units = tf.matmul(input_layer_flat, weights) + biases
            tf.summary.histogram('pre_activated_units', units)

    with tf.name_scope('identity') as scope:
        # y_conv = tf.nn.softmax(units)
        y_conv = units

    return y_conv
