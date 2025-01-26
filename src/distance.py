import tensorflow as tf
import numpy as np

@tf.function(jit_compile=True)
def squared_distances(mb, fv):
    """calculates the squared euclidean distances of every vector in fv to the set of vectors mb.

    Args:
        mb (tensor): tensor representing set of m n-d vectors as a m*n Matrix
        fv (tensor): tensor representing set of k n-d vectors as a k*n Matrix

    Returns:
        tensor of shape k equal to dist(fv[i],mb)**2
    """
    DM = tf.math.reduce_sum(tf.math.pow(fv,2),axis=-1)[:,tf.newaxis] \
        - 2*tf.linalg.matmul(fv,mb,transpose_b=True) \
        + tf.math.reduce_sum(tf.math.pow(mb,2),axis=-1)[tf.newaxis,:]
    return tf.math.reduce_min(DM,axis=-1)

@tf.function(jit_compile=True)
def k_squared_distances(mb, fv, k):
    """calculates the squared euclidean distances of every vector in fv to the set of vectors mb.
        returns the k closest distances to elements in mb for every vector in fv

    Args:
        mb (tensor): tensor representing set of m n-d vectors as a m*n Matrix
        fv (tensor): tensor representing set of j n-d vectors as a j*n Matrix

    Returns:
        tensor of shape (j,k) where result[j,0] is the distance of fv[j] to mb
    """
    DM = tf.math.reduce_sum(tf.math.pow(fv,2),axis=-1)[:,tf.newaxis] \
        - 2*tf.linalg.matmul(fv,mb,transpose_b=True) \
        + tf.math.reduce_sum(tf.math.pow(mb,2),axis=-1)[tf.newaxis,:]
    return -tf.math.top_k(-DM,k)[0]