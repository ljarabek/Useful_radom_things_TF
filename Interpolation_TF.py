import numpy as np
import tensorflow as tf

a = [[[0.125, 0.25, 0.125], [0.25, 0.5, 0.25], [0.125, 0.25, 0.125]],
     [[0.25, 0.5, 0.25], [0.50, 1.00, 0.50], [0.25, 0.5, 0.25]],
     [[0.125, 0.25, 0.125], [0.25, 0.5, 0.25], [0.125, 0.25, 0.125]]]
kernel = np.expand_dims(np.expand_dims(a, axis=-1), axis=-1)


def interpolation_linear(input, channels, name="interp_2x"):
    """
        linear 2x interpolation for 3D data as a special case of transpose convolution
    :param input: data
    :param name: name
    :return: interpolated matrix (double the size)
    """
    with tf.variable_scope(name):
        b_, x_, y_, z_, ch_ = list(input.get_shape())
        x_ *= 2
        y_ *= 2
        z_ *= 2
        interp_kernel = np.tile(kernel, (ch_, ch_))
        A = tf.constant(value=interp_kernel, name="interp_kernel_" + name, dtype=tf.float32)

        output_shape = tf.constant([b_, x_, y_, z_, ch_], dtype=tf.int32)

    return tf.math.divide(
        tf.nn.conv3d_transpose(input, filter=A, output_shape=output_shape, strides=[1, 2, 2, 2, 1], padding="SAME",
                               name="interp_" + name), channels)


b = np.ones(shape=kernel.shape)

def interpolation_NN(input, channels, name="interp_2x"):
    """
        nearest-neighbor 2x interpolation for 3D data as a special case of transpose convolution
    :param input: data
    :param name: name
    :return: interpolated matrix (double the size)
    """
    with tf.variable_scope(name):
        b_, x_, y_, z_, ch_ = list(input.get_shape())
        x_ *= 2
        y_ *= 2
        z_ *= 2
        interp_kernel = np.tile(b, (ch_, ch_))
        A = tf.constant(value=interp_kernel, name="interp_kernel_" + name, dtype=tf.float32)

        output_shape = tf.constant([b_, x_, y_, z_, ch_], dtype=tf.int32)

    return tf.math.divide(
        tf.nn.conv3d_transpose(input, filter=A, output_shape=output_shape, strides=[1, 2, 2, 2, 1], padding="SAME",
                               name="interp_" + name), channels)