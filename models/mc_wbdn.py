import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def identity_block(x, filter):
    # Layer 1
    conv_1 = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    norm_1 = tf.keras.layers.BatchNormalization(axis=3)(conv_1)
    activation_1 = tf.keras.layers.Activation('relu')(norm_1)

    # Layer 2
    conv_2 = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(activation_1)
    norm_2 = tf.keras.layers.BatchNormalization(axis=3)(conv_2)

    # Add Residue
    add_1 = tf.keras.layers.Add()([norm_2, x])     
    activation_2 = tf.keras.layers.Activation('relu')(add_1)

    # Return Output Layer
    return activation_2 