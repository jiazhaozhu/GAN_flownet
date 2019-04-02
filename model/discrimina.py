"""functions used to construct different architectures
"""

import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS


def int_shape(x):
    return list(map(int, x.get_shape()))


def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = len(x.get_shape()) - 1
    return tf.nn.elu(tf.concat(values=[x, -x], axis=axis))


def set_nonlinearity(name):
    if name == 'concat_elu':
        return concat_elu
    elif name == 'elu':
        return tf.nn.elu
    elif name == 'concat_relu':
        return tf.nn.crelu
    elif name == 'relu':
        return tf.nn.relu
    else:
        raise ('nonlinearity ' + name + ' is not supported')


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable(name, shape, initializer):
    """Helper to create a Variable.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    # getting rid of stddev for xavier ## testing this for faster convergence
    var = tf.get_variable(name, shape, initializer=initializer)
    return var


def conv_layer(inputs, kernel_size, stride, num_features, idx, nonlinearity=None):
    with tf.variable_scope('Dis_{0}_conv'.format(idx),reuse=tf.AUTO_REUSE) as scope:
        input_channels = int(inputs.get_shape()[3])

        weights = _variable('weights', shape=[kernel_size, kernel_size, input_channels, num_features],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = _variable('biases', [num_features], initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME')
        conv_biased = tf.nn.bias_add(conv, biases)
        if nonlinearity is not None:
            conv_biased = nonlinearity(conv_biased)
        return conv_biased

def res_block(x, a=None, filter_size=16, nonlinearity=concat_elu, keep_p=1.0, stride=1, gated=False, name="resnet"):
    orig_x = x
    orig_x_int_shape = int_shape(x)
    if orig_x_int_shape[3] == 1:
        x_1 = conv_layer(x, 3, stride, filter_size, name + '_conv_1')
    else:
        x_1 = conv_layer(nonlinearity(x), 3, stride, filter_size, name + '_conv_1')
    
    x_1 = nonlinearity(x_1)
    if keep_p < 1.0:
        x_1 = tf.nn.dropout(x_1, keep_prob=keep_p)
    if not gated:
        x_2 = conv_layer(x_1, 3, 1, filter_size, name + '_conv_2')
    else:
        x_2 = conv_layer(x_1, 3, 1, filter_size * 2, name + '_conv_2')
        x_2_1, x_2_2 = tf.split(axis=3, num_or_size_splits=2, value=x_2)
        x_2 = x_2_1 * tf.nn.sigmoid(x_2_2)

    if int(orig_x.get_shape()[2]) > int(x_2.get_shape()[2]):
        assert (int(orig_x.get_shape()[2]) == 2 * int(x_2.get_shape()[2]), "res net block only supports stirde 2")
        orig_x = tf.nn.avg_pool(orig_x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    # pad it
    out_filter = filter_size
    in_filter = int(orig_x.get_shape()[3])
    if out_filter != in_filter:
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter - in_filter), 0]])

    return orig_x + x_2


def dis(inputs,target, nr_res_blocks=1, keep_prob=1.0, nonlinearity_name='concat_elu', gated=True):
    """Builds conv part of net.
    Args:
      inputs: input images
      keep_prob: dropout layer
    """
    nonlinearity = set_nonlinearity(nonlinearity_name)
    filter_size = 8
    # store for as
    a = []
    # res_1
    x = tf.concat([inputs,target],axis=3)

    # res_2
    a.append(x)
    filter_size = 2 * filter_size
    x = res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, stride=2, gated=gated,
                  name="resnet_2_downsample")

    # res_3
    a.append(x)
    filter_size = 2 * filter_size
    x = res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, stride=2, gated=gated,
                  name="resnet_3_downsample")

    # res_4
    a.append(x)
    filter_size = 2 * filter_size
    x = res_block(x, filter_size=filter_size, nonlinearity=nonlinearity, keep_p=keep_prob, stride=2, gated=gated,
                  name="resnet_4_downsample")


    x = conv_layer(x, 5, 1, 1, "last_conv")
    x = tf.nn.sigmoid(x)

    # tf.summary.image('sflow_p_x', x[:, :, :, 1:2])
    # tf.summary.image('sflow_p_v', x[:, :, :, 0:1])

    return x


