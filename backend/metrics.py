import tensorflow as tf
from tensorflow.keras.backend import flatten


def tf_round(x, decimals = 0):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier


def MIOU():
    m = tf.keras.metrics.MeanIoU(num_classes=2)
    def MIoU(y_true, y_pred):
        m.reset_states()
        y_true, y_pred = y_true, tf.where(y_pred >= 0.5, 1.1, 0.0)
        _ = m.update_state(y_true, y_pred)
        return m.result()
    return MIoU


def CombinedMIoU(y_true, y_pred):
    """Compute Mean Intersection Over Union For A Single Channel Binary Prediction Mask"""
    y_true, y_pred = flatten(y_true), flatten(tf.where(y_pred >= 0.5, 1.0, 0.0))
    water_MIoU = MIoU2(y_true, y_pred)
    background_MIoU = MIoU2(tf.where(y_true == 0, 1.0, 0.0), tf.ones_like(y_pred) - y_pred)
    return tf_round((water_MIoU + background_MIoU) / tf.constant(2.0), decimals=3)


#def MIoU(y_true, y_pred):
   #  """Compute Mean Intersection Over Union For A Single Channel Binary Prediction Mask"""
 #   y_true, y_pred = flatten(y_true), flatten(tf.where(y_pred >= 0.5, 1.0, 0.0))
  #  smoothing = tf.keras.backend.constant(0.000001) # Prevents Division By Zero
   # total_positives = tf.keras.backend.sum(y_pred)
    #true_positives = tf.keras.backend.sum(y_true * y_pred)
    #false_positives = total_positives - true_positives
    #false_negatives = tf.keras.backend.sum(y_true * tf.where(y_pred == 0, 1.0, 0.0))
    #return tf_round(true_positives / (true_positives + false_positives + false_negatives + smoothing), decimals=3)


def MIoU2(y_true, y_pred):
    """Compute Mean Intersection Over Union For A Single Channel Binary Prediction Mask"""
    y_true, y_pred = flatten(y_true), flatten(tf.where(y_pred >= 0.5, 1.0, 0.0))
    total_positives = tf.keras.backend.sum(y_pred)
    true_positives = tf.math.maximum(tf.keras.backend.sum(y_true * y_pred), tf.keras.backend.constant(0.000001))
    false_positives = tf.math.maximum(total_positives - true_positives, tf.keras.backend.constant(0.0))
    false_negatives = tf.keras.backend.sum(y_true * tf.where(y_pred == 0, 1.0, 0.0))
    return tf_round(true_positives / (true_positives + false_positives + false_negatives), decimals=3)
