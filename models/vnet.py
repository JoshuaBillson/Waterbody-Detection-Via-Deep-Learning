from tensorflow import keras
from models.layers import fusion_head, preprocessing_layer
from keras.layers import Conv2D, Concatenate, Layer, Input, Dropout, MaxPooling2D, Conv2DTranspose, concatenate
from keras.models import Model
from keras_unet_collection.models import vnet_2d


def vnet_rgb(config):
    """
        Summary:
            Create VNET from keras unet collection library model object
        Arguments:
            Model configuration from config.yaml
        Return:
            Keras.model object
    """

    # Construct Base Model
    model = vnet_2d((config["patch_size"], config['patch_size'], 3), filter_num=[16, 32, 64, 128, 256],
                    n_labels=1, res_num_ini=1, res_num_max=3, activation='PReLU', output_activation='Sigmoid',
                    batch_norm=True, pool=False, unpool=False, name='vnet')

    # Replace Output Layer
    x = model.layers[-3].output  # fetch the last layer previous layer output
    output = Conv2D(1, kernel_size=(1, 1), name="out", activation='sigmoid', dtype="float32")(x)  # create new last layer
    model = Model(inputs=model.input, outputs=output)

    # Replace Model Input
    inputs, out = preprocessing_layer(config["patch_size"])
    outputs = model(out)
    return Model(inputs=inputs, outputs=outputs)


def vnet_multispectral(config):
    pass
