from math import sqrt
from backend.config import get_patch_size, get_bands
from models.utils import get_model_name
from models.layers import rgb_nir_swir_input_layer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def convolution_block(block_input, num_filters=256, kernel_size=3, dilation_rate=1, use_bias=False,):
    x = layers.Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", use_bias=use_bias, kernel_initializer=keras.initializers.HeNormal())(block_input)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    return layers.Dropout(0.1)(x)


def encoder_block(block_input, num_filters=256):
    x = convolution_block(block_input, num_filters=num_filters)
    x = convolution_block(x, num_filters=num_filters)
    return x


def decoder_block(block_input, skip_input, num_filters=256):
    x = layers.Concatenate(axis=-1)([block_input, skip_input])
    x = convolution_block(x, num_filters=num_filters)
    x = convolution_block(x, num_filters=num_filters)
    return x


def pool_block(block_input):
    x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(block_input)
    return x


def upsample_block(block_input, num_filters=256):
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear",)(block_input)
    x = convolution_block(x, num_filters=num_filters, kernel_size=1)
    return x


def S2D(input_size, input_channels, output_size):
    def s2d(layer_input):
        output_channels = (input_size * input_size * input_channels) // (output_size * output_size)
        block_size = int(sqrt(output_channels // input_channels))
        return tf.nn.space_to_depth(layer_input, block_size=block_size)
    return s2d


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear")(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output


def UnetBase(model_input, num_filters, input_size, model_name):
    # Connect Input
    feature_map_sizes = [input_size // (2 ** x) for x in range(len(num_filters))]
    x = layers.Concatenate(axis=-1)(model_input) if len(model_input) > 1 else model_input[0]
    x = encoder_block(x, num_filters[0])
    skip_layers = [x]
    x = S2D(feature_map_sizes[0], num_filters[0], feature_map_sizes[1])(x)
    # x = pool_block(x)

    # Build Encoder
    for i, num_filter in enumerate(num_filters[1:-1]):
        x = encoder_block(x, num_filter)
        skip_layers.append(x)
        # x = pool_block(x) if i != (len(num_filters) - 2) else x
        x = S2D(feature_map_sizes[i+1], num_filters[i+2], feature_map_sizes[i+2])(x)
    
    # Bridge Encoder And Decoder
    x = convolution_block(x, num_filters=num_filters[-1])
    x = DilatedSpatialPyramidPooling(x)
    x = convolution_block(x, num_filters=num_filters[-1])
    skip_layers.append(x)
    
    # Build Decoder
    skip_layers.reverse()
    num_filters.reverse()
    for skip_layer, num_filter in zip(skip_layers[1:], num_filters[1:]):
        x = upsample_block(x, num_filters=num_filter)
        x = decoder_block(x, skip_layer, num_filters=num_filter)
    
    # Create Output Layer
    model_output = layers.Conv2D(1, kernel_size=(3, 3), name="out", activation='sigmoid', dtype="float32", padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output, name=model_name)


def MyUnet(config, num_filters=None):
    num_filters = [64, 128, 256, 512, 1024] if num_filters is None else num_filters
    patch_size = get_patch_size(config)
    rgb_inputs = layers.Input(shape=(patch_size, patch_size, 3))
    nir_inputs = layers.Input(shape=(patch_size, patch_size, 1))
    swir_inputs = layers.Input(shape=(patch_size, patch_size, 1))
    return UnetBase([rgb_inputs, nir_inputs, swir_inputs], num_filters=num_filters, input_size=512, model_name=get_model_name(config))



def deeplab_input(config):
    bands = get_bands(config)
    patch_size = get_patch_size(config)
    rgb_inputs = layers.Input(shape=(patch_size, patch_size, 3))
    nir_inputs = layers.Input(shape=(patch_size, patch_size, 1))
    swir_inputs = layers.Input(shape=(patch_size, patch_size, 1))
    if "RGB" in bands and "NIR" in bands and "SWIR" in bands:
        return [rgb_inputs, nir_inputs, swir_inputs], rgb_nir_swir_input_layer(rgb_inputs, nir_inputs, swir_inputs, config)
    elif "NIR" in bands:
        return [nir_inputs], nir_inputs
    return [rgb_inputs], rgb_inputs
