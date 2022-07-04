import random
from math import sqrt
from backend.config import get_patch_size, get_bands
from models.utils import get_model_name
from models.layers import rgb_nir_swir_input_layer
from tensorflow.keras.activations import swish
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

def identity_block(x, filter, name=None):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.Activation('relu')(x)
    # Layer 2
    x = layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = layers.BatchNormalization(axis=3)(x)
    # Add Residue
    x = layers.Add()([x, x_skip])     
    x = layers.Activation('relu', name=name)(x) if name is not None else layers.Activation('relu')(x)
    return x


def convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = layers.Conv2D(filter, (3,3), padding = 'same', strides = (2,2))(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.Activation('relu')(x)
    # Layer 2
    x = layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = layers.BatchNormalization(axis=3)(x)
    # Processing Residue with conv(1,1)
    x_skip = layers.Conv2D(filter, (1,1), strides = (2,2))(x_skip)
    # Add Residue
    x = layers.Add()([x, x_skip])     
    x = layers.Activation('relu')(x)
    return x

def ResNet34(inputs, x_input, classes = 10):
    block_layers = [3, 4, 6, 3]
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for block in range(4):
        if block == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[block]):
                x = identity_block(x_input if j==0 else x, filter_size, name=f"out_{block+1}" if j == (block_layers[block] - 1) else None)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[block] - 1):
                x = identity_block(x, filter_size, name=f"out_{block+1}" if j == (block_layers[block] - 2) else None)
    # Step 4 End Dense Network
    x = tf.keras.layers.AveragePooling2D((2,2), padding = 'same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    x = tf.keras.layers.Dense(classes, activation = 'softmax')(x)
    model = tf.keras.models.Model(inputs = inputs, outputs = x, name = "ResNet34")
    return model


def mc_wbdn(config):
    # Create Fusion Head
    inputs, model_input = mc_wbdn_input(config)
    model_input = layers.Conv2D(64, (1,1), padding='same')(model_input)

    # Create Encoder
    # resnet50 = keras.applications.ResNet50( weights="imagenet", include_top=False, input_tensor=model_input)
    resnet34 = ResNet34(inputs, model_input)
    resnet34.summary()
    x = resnet34.get_layer("out_4").output
    x = DilatedSpatialPyramidPooling(x)

    # Concat Feature Maps From Lower Levels With Output Of DSPP Module
    input_a = D2S(input_size=32, output_size=128)(x)
    input_b = resnet34.get_layer("out_2").output
    # input_b = convolution_block(input_b, num_filters=48, kernel_size=1)
    input_c = resnet34.get_layer("out_1").output
    input_c = S2D(input_size = 256, input_channels=64, output_size=126)(input_c)
    input_d = resnet34.get_layer("out_3").output
    input_d = D2S(input_size=64, output_size=128)(input_d)

    """
    # Create Decoder & Output Layer
    x = layers.Concatenate(axis=-1)([input_a, input_b, input_c, input_d])
    x = convolution_block(x)
    x = D2S(input_size=128, output_size=512)(x)
    x = convolution_block(x)
    model_output = layers.Conv2D(1, kernel_size=(1, 1), name="out", activation='sigmoid', dtype="float32")(x)
    return keras.Model(inputs=inputs, outputs=model_output, name=get_model_name(config))
    """

    # Create Decoder & Output Layer
    x = layers.Concatenate(axis=-1)([input_a, input_b, input_c, input_d])
    x = convolution_block(x)
    x = layers.UpSampling2D( size=(4, 4), interpolation="bilinear")(x)
    x = convolution_block(x, num_filters=128)
    model_output = layers.Conv2D(1, kernel_size=(1, 1), name="out", activation='sigmoid', dtype="float32")(x)
    return keras.Model(inputs=inputs, outputs=model_output, name=get_model_name(config))

    # Create Decoder & Output Layer
    x = layers.Concatenate(axis=-1)([input_a, input_b, input_c, input_d])
    x = convolution_block(x)
    x = layers.UpSampling2D( size=(get_patch_size(config) // x.shape[1], get_patch_size(config) // x.shape[2]), interpolation="bilinear",)(x)
    x = convolution_block(x)
    model_output = layers.Conv2D(1, kernel_size=(1, 1), name="out", activation='sigmoid', dtype="float32")(x)
    return keras.Model(inputs=inputs, outputs=model_output, name=get_model_name(config))

    # Create Decoder & Output Layer
    x = layers.Concatenate(axis=-1)([input_a, input_b, input_c, input_d])
    x = convolution_block(x, num_filters=128)
    x = D2S(input_size=128, output_size=512)(x)
    model_output = layers.Conv2D(1, kernel_size=(1, 1), name="out", activation='sigmoid', dtype="float32")(x)
    return keras.Model(inputs=inputs, outputs=model_output, name=get_model_name(config))


def mc_wbdn_input(config):
    patch_size = get_patch_size(config)
    rgb_inputs = layers.Input(shape=(patch_size, patch_size, 3))
    nir_inputs = layers.Input(shape=(patch_size, patch_size, 1))
    swir_inputs = layers.Input(shape=(patch_size // 2, patch_size // 2, 1))
    return [rgb_inputs, nir_inputs, swir_inputs], rgb_nir_swir_input_layer(rgb_inputs, nir_inputs, swir_inputs, config)
