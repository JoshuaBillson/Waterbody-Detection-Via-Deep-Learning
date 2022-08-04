from typing import  Dict, Any, Sequence
import tensorflow as tf
from tensorflow.keras.activations import swish
from tensorflow.keras.layers import Conv2D, Conv3D, DepthwiseConv2D, Layer, concatenate, Reshape, Lambda, BatchNormalization
from backend.config import get_fusion_head


def rgb_nir_swir_input_layer(rgb_inputs: Layer, nir_inputs: Layer, swir_inputs: Layer, config: Dict[str, Any]) -> Layer:
    """
    Construct An Input Layer For RGB + NIR + SWIR Bands
    :param rgb_inputs: The input layer for RGB features
    :param nir_inputs: The input layer for NIR features
    :param swir_inputs: The input layer for SWIR features
    :param config: A dictionary storing the script configuration
    :returns: The fusion head specified by config as a Keras layer
    """
    fusion_heads = {
        "naive": fusion_head_naive,
        "prism": fusion_head_prism, 
        "depthwise": fusion_head_depthwise,
        "3D": fusion_head_3d,
        "paper": fusion_head_paper, 
        "grayscale": fusion_head_grayscale, 
    }
    constructor = fusion_heads[get_fusion_head(config)]
    return constructor(rgb_inputs, nir_inputs, swir_inputs) if get_fusion_head(config) != "prism" else constructor([rgb_inputs, nir_inputs, swir_inputs])


def fusion_head_naive(rgb_inputs: Layer, nir_inputs: Layer, swir_inputs: Layer) -> Layer:
    """
    A layer for combining RGB, NIR, and SWIR inputs by concatenating the features along the channel axis
    :param rgb_inputs: The input layer for RGB features
    :param nir_inputs: The input layer for NIR features
    :param swir_inputs: The input layer for SWIR features
    :returns: The fusion head as a Keras layer
    """
    return concatenate([rgb_inputs, nir_inputs, swir_inputs], axis=3)


def fusion_head_depthwise(rgb_inputs: Layer, nir_inputs: Layer, swir_inputs: Layer) -> Layer:
    """
    A layer for combining RGB, NIR, and SWIR inputs by applying a depthwise separable convolution followed by a 1x1 convolution
    :param rgb_inputs: The input layer for RGB features
    :param nir_inputs: The input layer for NIR features
    :param swir_inputs: The input layer for SWIR features
    :returns: The fusion head as a Keras layer
    """
    concat = concatenate([rgb_inputs, nir_inputs, swir_inputs], axis=3)
    depthwise_conv = DepthwiseConv2D((3, 3), strides=(1, 1), activation=None, kernel_initializer='he_uniform', padding="same")(concat)
    return Conv2D(128, (1, 1), activation=swish, kernel_initializer='he_uniform', padding="same")(depthwise_conv)


def fusion_head_3d(rgb_inputs: Layer, nir_inputs: Layer, swir_inputs: Layer) -> Layer:
    """
    A layer for combining RGB, NIR, and SWIR inputs by applying a 3D convolution
    :param rgb_inputs: The input layer for RGB features
    :param nir_inputs: The input layer for NIR features
    :param swir_inputs: The input layer for SWIR features
    :returns: The fusion head as a Keras layer
    """
    concat = concatenate([rgb_inputs, nir_inputs, swir_inputs], axis=3)
    reshaped = Reshape((512, 512, 5, 1))(concat)
    conv3d = Conv3D(25, (7, 7, 5), padding="same", activation=swish, kernel_initializer='he_uniform')(reshaped)
    return Reshape((512, 512, 125))(conv3d)


def fusion_head_paper(rgb_inputs: Layer, nir_inputs: Layer, swir_inputs: Layer) -> Layer:
    """
    A layer for combining RGB, NIR, and SWIR inputs as outlined in 'Deep-Learning-Based Multispectral Satellite ImageSegmentation for Water Body Detection'
    :param rgb_inputs: The input layer for RGB features
    :param nir_inputs: The input layer for NIR features
    :param swir_inputs: The input layer for SWIR features
    :returns: The fusion head as a Keras layer
    """
    rgb_conv = Conv2D(64, (7, 7), strides=(2, 2), activation=swish, kernel_initializer='he_uniform', padding="same")(rgb_inputs)

    nir_conv = Conv2D(32, (7, 7), strides=(2, 2), activation=swish, kernel_initializer='he_uniform', padding="same")(nir_inputs)

    swir_conv = Conv2D(32, (1, 1), strides=(1, 1), activation=swish, kernel_initializer='he_uniform', padding="same")(swir_inputs)

    concat = concatenate([rgb_conv, nir_conv, swir_conv], axis=3)

    conv = Conv2D(64, (1, 1), strides=(1, 1), activation=None, kernel_initializer='he_uniform', padding="same")(concat)
    bn = BatchNormalization()(conv)
    return tf.nn.swish(bn)


def fusion_head_grayscale(rgb_inputs: Layer, nir_inputs: Layer, swir_inputs: Layer) -> Layer:
    """
    A layer which combines RGB, NIR, and SWIR inputs by transforming the RGB bands to grayscale
    :param rgb_inputs: The input layer for RGB features
    :param nir_inputs: The input layer for NIR features
    :param swir_inputs: The input layer for SWIR features
    :returns: The fusion head as a Keras layer
    """
    # Turn RGB Band To Grayscale
    grayscale = Lambda(tf.image.rgb_to_grayscale)(rgb_inputs)

    # Concat Inputs
    concat = concatenate([grayscale, nir_inputs, swir_inputs], axis=3)

    # Return Final Layer
    return concat


def fusion_head_prism(inputs: Sequence[Layer]) -> Layer:
    """
    A layer which combines the given input bands and uses a convolution to produce a 3 band output
    :param inputs: The list of inputs we want to fuse together
    :returns: The fusion head as a Keras layer
    """
    x = concatenate(inputs, axis=3) if len(inputs) > 1 else inputs[0]
    return Conv2D(3, 1, 1, padding="same")(x)