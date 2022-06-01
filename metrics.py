import tensorflow as tf
from tensorflow.keras.backend import flatten


def MIoU(y_true, y_pred):
    """Compute Mean Intersection Over Union For A Single Channel Binary Prediction Mask"""
    y_true, y_pred = flatten(y_true), flatten(tf.where(y_pred >= 0.5, 1.0, 0.0))
    # size = tf.cast(tf.size(y_pred), tf.float32)
    total_positives = tf.keras.backend.sum(y_pred)
    true_positives = tf.keras.backend.sum(y_true * y_pred)
    false_positives = total_positives - true_positives
    false_negatives = tf.keras.backend.sum(y_true * tf.where(y_pred == 0, 1.0, 0.0))
    # return true_positives
    # return false_positives
    # return false_negatives
    # return (true_positives + false_positives + false_negatives)
    return true_positives / (true_positives + false_positives + false_negatives + 0.00001)
