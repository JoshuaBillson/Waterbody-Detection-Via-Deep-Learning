import tensorflow as tf
from typing import Sequence, Tuple, Dict, Any
from tensorflow.keras.activations import swish
from tensorflow import Tensor
from tensorflow.keras.layers import Conv2D, Layer, Input, concatenate, MaxPooling2D


def rgb_input_layer(inputs: Layer):
    """
    Construct An Input Layer For RGB Bands
    :param config: A dictionary storing the script configuration
    :return: The input and output layers as the tuple (inputs, outputs)
    """
    rgb_conv = Conv2D(32, (3, 3), strides=(1, 1), activation=swish, kernel_initializer='he_uniform', padding="same")(inputs)
    return rgb_conv


def nir_input_layer(inputs: Layer):
    """
    Construct An Input Layer For The NIR Bands
    :param config: A dictionary storing the script configuration
    :return: The input and output layers as the tuple (inputs, outputs)
    """
    nir_conv = Conv2D(32, (3, 3), strides=(1, 1), activation=swish, kernel_initializer='he_uniform', padding="same")(inputs)
    return nir_conv


def swir_input_layer(config: Dict[str, Any]):
    """
    Construct An Input Layer For The SWIR Bands
    :param config: A dictionary storing the script configuration
    :return: The input and output layers as the tuple (inputs, outputs)
    """
    return preprocessing_layer(config["patch_size"] // 2, is_rgb=False)


def rgb_nir_input_layer(config: Dict[str, Any]):
    """
    Construct An Input Layer For RGB + NIR Bands
    :param config: A dictionary storing the script configuration
    :return: The input and output layers as the tuple (inputs, outputs)
    """
    rgb_input, rgb_output = rgb_input_layer(config)
    nir_input, nir_output = nir_input_layer(config)
    concat = concatenate([rgb_output, nir_output], axis=3)
    return [rgb_input, nir_input], concat


@tf.function
def channel_wise_norm(tensor: Tensor) -> Tensor:
    """
    Normalize the intensity of each channel of the input tensor
    :param tensor: The input tensor to be normalized
    :return: The normalized tensor
    """
    num_channels = tensor.shape[-1]
    channels = tf.TensorArray(tf.float32, size=num_channels)
    for channel_idx in range(num_channels):
        channel = (tensor[..., channel_idx] - tf.reduce_mean(tensor[..., channel_idx])) / tf.math.reduce_std(tensor[..., channel_idx])
        channels = channels.write(channel_idx, channel)
    new_tensor = tf.transpose(channels.stack(), perm=[1, 2, 3, 0])
    return tf.ensure_shape(new_tensor, tensor.shape)


def preprocessing_layer(patch_size: int, is_rgb: bool = True) -> Tuple[Layer, Layer]:
    """
    Normalize the intensity of each channel of the input tensor
    :param tensor: The input tensor to be normalized
    :return: The normalized tensor
    """
    inputs = Input(shape=(patch_size, patch_size, 3 if is_rgb else 1))
    # threshold_layer = Lambda(lambda x: tf.clip_by_value(x, 0, 3000))(inputs)
    # normalization_layer = Lambda(channel_wise_norm)(inputs if is_rgb else threshold_layer)
    # normalization_layer = Lambda(channel_wise_norm)(inputs)
    return inputs


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
