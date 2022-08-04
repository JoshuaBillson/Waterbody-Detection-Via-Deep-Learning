from math import sqrt
from backend.config import get_input_channels, get_patch_size, get_bands, get_backbone
from models.layers import rgb_nir_swir_input_layer
from models.utils import get_model_name, assemble_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def D2S(input_size, output_size):
    def d2s(layer_input):
        block_size = output_size // input_size
        return tf.nn.depth_to_space(layer_input, block_size=block_size)
    return d2s


def S2D(input_size, input_channels, output_size):
    def s2d(layer_input):
        output_channels = (input_size * input_size * input_channels) // (output_size * output_size)
        block_size = int(sqrt(output_channels // input_channels))
        return tf.nn.space_to_depth(layer_input, block_size=block_size)
    return s2d


def convolution_block(block_input, num_filters=256, kernel_size=3, dilation_rate=1, use_bias=False, activation=tf.nn.relu):
    x = layers.Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", use_bias=use_bias, kernel_initializer=keras.initializers.HeNormal())(block_input)
    x = layers.BatchNormalization()(x)
    return activation(x)


def DilatedSpatialPyramidPooling(dspp_input):
    x = convolution_block(dspp_input, kernel_size=1, use_bias=True)
    out_pool = layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1, num_filters=1024)
    return output


def mcwbdn_base_1(config, weights=None):
    # Construct ResNet Encoder
    image_size = get_patch_size(config)
    inputs = layers.Input(shape=(image_size, image_size, get_input_channels(config)))
    resnet50 = keras.applications.ResNet50(weights=weights, include_top=False, input_tensor=inputs)
    resnet50.summary()

    # Construct ASPP Module
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    
    # Decoder Block 1 
    input_a = D2S(32, 64)(x)
    input_b = resnet50.get_layer("conv3_block4_out").output
    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x, num_filters=512)
    x = convolution_block(x, num_filters=512)

    # Decoder Block 2
    input_a = D2S(64, 128)(x)
    input_b = resnet50.get_layer("conv2_block3_out").output
    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x, num_filters=256)
    x = convolution_block(x, num_filters=256)

    # Implement Decoder And Output
    x = layers.UpSampling2D( size=(image_size // x.shape[1], image_size // x.shape[2]), interpolation="bilinear",)(x)
    x = convolution_block(x, num_filters=128)
    x = convolution_block(x, num_filters=128)
    model_output = layers.Conv2D(1, kernel_size=(1, 1), name="out", activation='sigmoid', dtype="float32")(x)

    # Construct Base Model
    return keras.Model(inputs=inputs, outputs=model_output, name="unet_base")


def mcwbdn_base(config, weights=None):
    # Construct ResNet Encoder
    image_size = get_patch_size(config)
    inputs = layers.Input(shape=(image_size, image_size, get_input_channels(config)))
    resnet50 = keras.applications.ResNet50(weights=weights, include_top=False, input_tensor=inputs)
    resnet50.summary()

    # Construct ASPP Module
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    # Grab Skip Connections 
    input_a = D2S(32, 128)(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    # Implement Decoder And Output
    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = D2S(128, 512)(x)
    model_output = layers.Conv2D(1, kernel_size=(1, 1), name="out", activation='sigmoid', dtype="float32")(x)

    # Construct Base Model
    return keras.Model(inputs=inputs, outputs=model_output, name="unet_base")


def MC_WBDN_ResNet50(config):
    base_model = mcwbdn_base(config)
    base_model.summary()
    return assemble_model(base_model, config)


def MC_WBDN_ResNet50_ImageNet(config):
    if get_input_channels(config) != 3:
        config["hyperparameters"]["fusion_head"] = "prism"
    base_model = mcwbdn_base(config, weights="imagenet")
    base_model.summary()
    return assemble_model(base_model, config)
