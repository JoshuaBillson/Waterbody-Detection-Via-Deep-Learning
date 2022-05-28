import time
from typing import Dict, Any
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from models.layers import rgb_input_layer, nir_input_layer, swir_input_layer, rgb_nir_input_layer
from config import get_model_type, get_bands, get_backbone


def assemble_model(base_model: Model, config: Dict[str, Any]) -> Model:
    """
    Takes a base model and modifies its input and output layers as appropriate for the specified input bands 
    :param base_model: The base model (U-Net, U2-Net, U-Net++, etc.) that we want to extend
    :param config: The model configuration
    :return: The final model.
    """
    bands = get_bands(config)
    if "RGB" in bands and "NIR" in bands:
        return rgb_nir_model(base_model, config)
    elif "RGB" in bands:
        return rgb_model(base_model, config)
    elif "NIR" in bands:
        return nir_model(base_model, config)
    elif "SWIR" in bands:
        return swir_model(base_model, config)
    raise ValueError("Invalid Bands Received!")


def rgb_model(base_model: Model, config: Dict[str, Any]) -> Model:
    """
    Create An RGB Model From A Given Base Model And Configuration
    :param base_model: The base model (U-Net, U2-Net, U-Net++, etc.) that we want to extend
    :param config: The model configuration
    :return: The final RGB model.
    """
    # Replace Output Layer
    model = replace_output(base_model, config)

    # Replace Model Input
    inputs, out = rgb_input_layer(config)
    outputs = model(out)
    return Model(inputs=inputs, outputs=outputs, name=get_model_name(config))


def nir_model(base_model: Model, config: Dict[str, Any]) -> Model:
    """
    Create A NIR Model From A Given Base Model And Configuration
    :param base_model: The base model (U-Net, U2-Net, U-Net++, etc.) that we want to extend
    :param config: The model configuration
    :return: The final NIR model.
    """
    # Replace Output Layer
    model = replace_output(base_model, config)

    # Replace Model Input
    inputs, out = nir_input_layer(config)
    outputs = model(out)
    return Model(inputs=inputs, outputs=outputs, name=get_model_name(config))


def swir_model(base_model: Model, config: Dict[str, Any]) -> Model:
    """
    Create A SWIR Model From A Given Base Model And Configuration
    :param base_model: The base model (U-Net, U2-Net, U-Net++, etc.) that we want to extend
    :param config: The model configuration
    :return: The final SWIR model.
    """
    # Replace Output Layer
    model = replace_output(base_model, config)

    # Replace Model Input
    inputs, out = swir_input_layer(config)
    outputs = model(out)
    return Model(inputs=inputs, outputs=outputs, name=get_model_name(config))


def rgb_nir_model(base_model: Model, config: Dict[str, Any]) -> Model:
    """
    Create An RGB + NIR Model From A Given Base Model And Configuration
    :param base_model: The base model (U-Net, U2-Net, U-Net++, etc.) that we want to extend
    :param config: The model configuration
    :return: The final RGB + NIR model.
    """
    # Replace Output Layer
    model = replace_output(base_model, config)

    # Replace Model Input
    inputs, out = rgb_nir_input_layer(config)
    outputs = model(out)
    return Model(inputs=inputs, outputs=outputs, name=get_model_name(config))


def replace_output(base_model: Model, config: Dict[str, Any]) -> Model:
    """
    Replace the output layer of the given base model. If the input bands include SWIR, we need to upsample the mask by a factor of 2
    :param base_model: The base model (U-Net, U2-Net, U-Net++, etc.) whose output layer we want to replace
    :param config: The model configuration
    :return: The final model.
    """
    if "SWIR" in get_bands(config):
        x = base_model.layers[-3].output
        up_sample = Conv2DTranspose(1, (2, 2), strides=(1, 1), padding='same')(x)
        outputs = Conv2D(1, kernel_size=(1, 1), name="out", activation='sigmoid', dtype="float32")(up_sample)  # create new last layer
        return Model(inputs=base_model.input, outputs=outputs, name=f"{get_model_type(config)}_base")
    x = base_model.layers[-3].output
    outputs = Conv2D(1, kernel_size=(1, 1), name="out", activation='sigmoid', dtype="float32")(x)
    return Model(inputs=base_model.input, outputs=outputs, name=f"{get_model_type(config)}_base")


def get_model_name(config: Dict[str, Any]) -> str:
    return f"{get_model_type(config)}.{'+'.join(get_bands(config))}.{get_backbone(config)}.{int(time.time())}".lower()
