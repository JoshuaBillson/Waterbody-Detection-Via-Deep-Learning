from backend.config import get_patch_size, get_bands
from models.utils import get_model_name
from models.layers import rgb_nir_swir_input_layer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def convolution_block(block_input, num_filters=256, kernel_size=3, dilation_rate=1, use_bias=False,):
    x = layers.Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", use_bias=use_bias, kernel_initializer=keras.initializers.HeNormal())(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


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


def DeeplabV3Plus(config):
    image_size = get_patch_size(config)
    inputs, model_input = deeplab_input(config)
    resnet50 = keras.applications.ResNet50( weights=None, include_top=False, input_tensor=model_input)
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D( size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]), interpolation="bilinear",)(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D( size=(image_size // x.shape[1], image_size // x.shape[2]), interpolation="bilinear",)(x)
    model_output = layers.Conv2D(1, kernel_size=(1, 1), name="out", activation='sigmoid', dtype="float32")(x)

    return keras.Model(inputs=inputs, outputs=model_output, name=get_model_name(config))


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
