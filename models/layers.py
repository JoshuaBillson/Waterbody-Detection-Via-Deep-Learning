import tensorflow as tf
from random import randint
from tensorflow import keras
from typing import Tuple, Sequence
from keras.activations import swish
from keras.models import Model
import numpy as np
from tensorflow import Tensor
from keras.layers import Conv2D, Layer, Input, concatenate, Lambda, Reshape


@tf.function
def channel_wise_norm(tensor):
    num_channels = tf.shape(tensor)[-1]
    channels = tf.TensorArray(tf.float32, size=num_channels)
    for channel_idx in tf.range(num_channels):
        channel = (tensor[..., channel_idx] - tf.reduce_mean(tensor[..., channel_idx])) / tf.math.reduce_std(tensor[..., channel_idx])
        channels = channels.write(channel_idx, channel)
    new_tensor = tf.transpose(channels.stack(), perm=[1, 2, 3, 0])
    return tf.ensure_shape(new_tensor, tensor.shape)


def preprocessing_layer(patch_size, is_rgb=True):
    inputs = Input(shape=(patch_size, patch_size, 3 if is_rgb else 1))
    threshold_layer = Lambda(lambda x: tf.clip_by_value(x, 0, 3000))(inputs)
    normalization_layer = Lambda(lambda x: channel_wise_norm(x))(inputs if is_rgb else threshold_layer)
    return inputs, normalization_layer


def fusion_head(patch_size: int = 512, channels: Sequence[int] = None, rgb_output=False):
    channels = (64, 32, 32) if channels is None else channels
    rgb_input, rgb_out = preprocessing_layer(patch_size)
    rgb_conv = Conv2D(channels[0], (7, 7), strides=(2, 2), activation=swish, kernel_initializer='he_uniform', padding="same")(rgb_out)

    nir_input, nir_out = preprocessing_layer(patch_size, is_rgb=False)
    nir_conv = Conv2D(channels[1], (7, 7), strides=(2, 2), activation=swish, kernel_initializer='he_uniform', padding="same")(nir_out)

    swir_input, swir_out = preprocessing_layer(patch_size // 2, is_rgb=False)
    swir_conv = Conv2D(channels[2], (1, 1), strides=(1, 1), activation=swish, kernel_initializer='he_uniform', padding="same")(swir_out)

    concat = concatenate([rgb_conv, nir_conv, swir_conv], axis=3)

    output = Conv2D(3, (1, 1), strides=(1, 1), activation=swish, kernel_initializer='he_uniform')(concat) if rgb_output else concat

    return [rgb_input, nir_input, swir_input], output
