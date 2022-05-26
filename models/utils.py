from tensorflow import keras
from models.layers import fusion_head, preprocessing_layer
from keras.layers import Conv2D
from keras.models import Model


def rgb_model(base_model, config):
    """
    Create An RGB Model From A Given Base Model And Configuration
    :param base_model: The base model (U-Net, U2-Net, U-Net++, etc.) that we want to extent
    :param config: The model configuration
    :return: The final RGB model.
    """
    # Replace Output Layer
    x = base_model.layers[-3].output  # fetch the last layer previous layer output
    output = Conv2D(1, kernel_size=(1, 1), name="out", activation='sigmoid', dtype="float32")(x)  # create new last layer
    model = Model(inputs=base_model.input, outputs=output)

    # Replace Model Input
    inputs, out = preprocessing_layer(config["patch_size"])
    outputs = model(out)
    return Model(inputs=inputs, outputs=outputs, name=config["hyperparameters"]["model"])
