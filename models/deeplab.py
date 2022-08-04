from backend.config import get_input_channels, get_patch_size, get_bands, get_backbone
from models.layers import rgb_nir_swir_input_layer
from models.utils import get_model_name, assemble_model
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


def deeplab_base(config, weights=None):
    # Construct ResNet Encoder
    image_size = get_patch_size(config)
    inputs = layers.Input(shape=(image_size, image_size, get_input_channels(config)))
    resnet50 = keras.applications.ResNet50(weights=weights, include_top=False, input_tensor=inputs)

    # Construct ASPP Module
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    # Grab Skip Connections 
    input_a = layers.UpSampling2D( size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]), interpolation="bilinear",)(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    # Implement Decoder And Output
    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D( size=(image_size // x.shape[1], image_size // x.shape[2]), interpolation="bilinear",)(x)
    model_output = layers.Conv2D(1, kernel_size=(1, 1), name="out", activation='sigmoid', dtype="float32")(x)

    # Construct Base Model
    return keras.Model(inputs=inputs, outputs=model_output, name="unet_base")


def DeeplabV3Plus(config):
    base_model = deeplab_base(config)
    base_model.summary()
    return assemble_model(base_model, config)


def DeeplabV3PlusImageNet(config):
    if get_input_channels(config) != 3:
        config["hyperparameters"]["fusion_head"] = "prism"
    base_model = deeplab_base(config, weights="imagenet")
    base_model.summary()
    return assemble_model(base_model, config)
