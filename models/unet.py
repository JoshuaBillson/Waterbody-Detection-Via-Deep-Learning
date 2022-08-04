from backend.config import get_patch_size, get_model_config
from models.utils import get_model_name, assemble_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def convolution_block(block_input, num_filters=256, kernel_size=3, dilation_rate=1, use_bias=False, use_batch_norm=False):
    """A single convolution block"""
    x = layers.Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", use_bias=use_bias, kernel_initializer=keras.initializers.HeNormal())(block_input)
    x = layers.BatchNormalization()(x) if use_batch_norm else x
    return tf.nn.relu(x)


def encoder_block(block_input, num_filters=256, num_blocks=2, use_batch_norm=False):
    """A block of convolutions used by the encoder"""
    x = convolution_block(block_input, num_filters=num_filters, use_batch_norm=use_batch_norm)
    for _ in range(num_blocks - 1):
        x = convolution_block(x, num_filters=num_filters, use_batch_norm=use_batch_norm)
    return x


def decoder_block(block_input, skip_input, num_filters=256, num_blocks=2, use_batch_norm=False):
    """A block of convolutions used by the decoder"""
    x = layers.Concatenate(axis=-1)([block_input, skip_input])
    x = convolution_block(x, num_filters=num_filters, use_batch_norm=use_batch_norm)
    for _ in range(num_blocks - 1):
        x = convolution_block(x, num_filters=num_filters, use_batch_norm=use_batch_norm)
    return x


def downsample_block(block_input):
    """A downsampling layer"""
    return layers.MaxPooling2D(pool_size=(2, 2), padding="same")(block_input)


def upsample_block(block_input, num_filters=256):
    """An upsampling layer"""
    return layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(block_input)


def get_skip_layers(model: keras.models.Model):
    skip_connections = []
    for layer in ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]:
        skip_connections.append(model.get_layer(layer).output)
    return skip_connections


def UnetBase(config, num_filters, num_blocks, use_batch_norm, model_name):
    """Base U-Net model"""
    # Connect Input
    input_channels, _ = get_model_config(config)
    patch_size = get_patch_size(config)
    inputs = layers.Input((patch_size, patch_size, input_channels))
    x = encoder_block(inputs, num_filters[0], num_blocks=num_blocks, use_batch_norm=use_batch_norm)
    skip_layers = [x]
    x = downsample_block(x)

    # Build Encoder
    for i, num_filter in enumerate(num_filters[1:]):
        x = encoder_block(x, num_filter, num_blocks=num_blocks, use_batch_norm=use_batch_norm)
        skip_layers.append(x)
        x = downsample_block(x) if i != (len(num_filters) - 2) else x
    
    # Build Decoder
    skip_layers.reverse()
    num_filters.reverse()
    for skip_layer, num_filter in zip(skip_layers[1:], num_filters[1:]):
        x = upsample_block(x, num_filters=num_filter)
        x = decoder_block(x, skip_layer, num_filters=num_filter, num_blocks=2, use_batch_norm=True)
    
    # Create Output Layer
    model_output = layers.Conv2D(1, kernel_size=(3, 3), name="out", activation='sigmoid', dtype="float32", padding="same")(x)
    return keras.Model(inputs=inputs, outputs=model_output, name=model_name)


def UnetResNetBase(model_input, num_filters, num_blocks, use_batch_norm, model_name):
    """Base U-Net model"""
    # Build Encoder
    num_filters = [64, 256, 512, 1024]
    resnet50 = keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=model_input)
    skip_layers = get_skip_layers(resnet50)
    
    # Build Decoder
    skip_layers.reverse()
    num_filters.reverse()
    x = skip_layers[0]
    for skip_layer, num_filter in zip(skip_layers[1:], num_filters):
        x = upsample_block(x, num_filters=num_filter)
        x = decoder_block(x, skip_layer, num_filters=num_filter, num_blocks=num_blocks, use_batch_norm=use_batch_norm)
    
    # Create Output Layer
    model_output = layers.Conv2D(1, kernel_size=(3, 3), name="out", activation='sigmoid', dtype="float32", padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output, name=model_name)


def Unet(config):
    # Construct Base Model
    model = UnetBase(config, num_filters=[64, 128, 256, 512, 1024, 2048], num_blocks=2, use_batch_norm=True, model_name="unet_base")
    model.summary()

    # Replace Input And Output Layers
    return assemble_model(model, config)


def DeepUnet(config):
    # Construct Base Model
    model = MyUnet(config, num_filters=[64, 128, 256, 512, 1024], num_blocks=4, use_batch_norm=True)
    model.summary()

    # Replace Input And Output Layers
    return assemble_model(model, config)
