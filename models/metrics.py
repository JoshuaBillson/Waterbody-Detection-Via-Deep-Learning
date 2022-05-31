import tensorflow as tf
from tensorflow.keras.backend import flatten, sum


def MIoU(y_true, y_pred):
    """Compute Mean Intersection Over Union For A Single Channel Binary Prediction Mask"""
    y_true = flatten(y_true)
    y_pred = flatten(tf.where(y_pred >= 0.5, 1.0, 0.0))
    true_positives = sum(y_true * y_pred)
    false_positives = sum(y_pred) - true_positives
    false_negatives = sum(y_pred * tf.where(y_true == 0, 1.0, 0.0))
    return (true_positives / (true_positives + false_positives + false_negatives))
