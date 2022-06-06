import tensorflow as tf
from tensorflow.keras.backend import flatten


def tf_round(x, decimals = 0):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier


def MIoU(y_true, y_pred):
    """Compute Mean Intersection Over Union For A Single Channel Binary Prediction Mask"""
    y_true, y_pred = flatten(y_true), flatten(tf.where(y_pred >= 0.5, 1.0, 0.0))
    # size = tf.cast(tf.size(y_pred), tf.float32)
    total_positives = tf.keras.backend.sum(y_pred)
    true_positives = tf.math.maximum(tf.keras.backend.sum(y_true * y_pred), tf.keras.backend.constant(0.000001))
    false_positives = tf.math.maximum(total_positives - true_positives, tf.keras.backend.constant(0.0))
    false_negatives = tf.keras.backend.sum(y_true * tf.where(y_pred == 0, 1.0, 0.0))
    # return true_positives
    # return false_positives
    # return false_negatives
    # return (true_positives + false_positives + false_negatives)
    return tf_round(true_positives / (true_positives + false_positives + false_negatives), decimals=3)
