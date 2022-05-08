import tensorflow as tf


def calc_euclidian_dists(x, y):
    """
    Calculate euclidian distance between two 3D tensors.

    Args:
        x (tf.Tensor):
        y (tf.Tensor):

    Returns (tf.Tensor): 2-dim tensor with distances.

    """
    n = x.shape[0]
    m = y.shape[0]
    x = tf.tile(tf.expand_dims(x, 1), [1, m, 1])
    y = tf.tile(tf.expand_dims(y, 0), [n, 1, 1])
    return tf.reduce_mean(tf.math.pow(x - y, 2), 2)
