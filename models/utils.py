import os
from typing import Dict, Any
from tensorflow.keras.layers import Conv2D, Input, concatenate
from tensorflow.keras.models import Model
from models.layers import rgb_input_layer, nir_input_layer, swir_input_layer, rgb_nir_input_layer, rgb_nir_swir_input_layer
from backend.config import get_model_type, get_bands, get_backbone, get_patch_size, get_experiment_tag
from backend.config import get_patch_size


def assemble_model(base_model: Model, config: Dict[str, Any]) -> Model:
    """
    Takes a base model and modifies its input and output layers as appropriate for the specified input bands 
    :param base_model: The base model (U-Net, U2-Net, U-Net++, etc.) that we want to extend
    :param config: The model configuration
    :return: The final model.
    """
    bands = get_bands(config)
    if "RGB" in bands and "NIR" in bands and "SWIR" in bands:
        return rgb_nir_swir_model(base_model, config)
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
    model.summary()

    # Replace Model Input
    patch_size = get_patch_size(config)
    inputs = Input(shape=(patch_size, patch_size, 3))
    outputs = model(inputs)
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
    patch_size = get_patch_size(config)
    inputs = Input(shape=(patch_size, patch_size, 1))
    x = nir_input_layer(inputs)
    outputs = model(x)
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
    patch_size = get_patch_size(config)
    inputs = Input(shape=(patch_size // 2, patch_size // 2, 1))
    x = swir_input_layer(inputs)
    outputs = model(x)
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
    patch_size = get_patch_size(config)
    rgb_inputs = Input(shape=(patch_size, patch_size, 3))
    nir_inputs = Input(shape=(patch_size, patch_size, 1))
    concat = concatenate([rgb_inputs, nir_inputs], axis=3)
    x = rgb_nir_input_layer(rgb_inputs, nir_inputs)
    outputs = model(x)
    return Model(inputs=[rgb_inputs, nir_inputs], outputs=outputs, name=get_model_name(config))


def rgb_nir_swir_model(base_model: Model, config: Dict[str, Any]) -> Model:
    """
    Create An RGB + NIR Model From A Given Base Model And Configuration
    :param base_model: The base model (U-Net, U2-Net, U-Net++, etc.) that we want to extend
    :param config: The model configuration
    :return: The final RGB + NIR model.
    """
    # Replace Output Layer
    model = replace_output(base_model, config)

    # Replace Model Input
    patch_size = get_patch_size(config)
    rgb_inputs = Input(shape=(patch_size, patch_size, 3))
    nir_inputs = Input(shape=(patch_size, patch_size, 1))
    swir_inputs = Input(shape=(patch_size, patch_size, 1))
    x = rgb_nir_swir_input_layer(rgb_inputs, nir_inputs, swir_inputs)
    outputs = model(x)
    return Model(inputs=[rgb_inputs, nir_inputs, swir_inputs], outputs=outputs, name=get_model_name(config))


def replace_output(base_model: Model, config: Dict[str, Any]) -> Model:
    """
    Replace the output layer of the given base model. 
    :param base_model: The base model (U-Net, U2-Net, U-Net++, etc.) whose output layer we want to replace
    :param config: The model configuration
    :return: The final model.
    """
    x = base_model.layers[-2].output
    outputs = Conv2D(1, kernel_size=(1, 1), name="out", activation='sigmoid', dtype="float32")(x)
    return Model(inputs=base_model.input, outputs=outputs, name=f"{get_model_type(config)}_base")


def get_model_name(config: Dict[str, Any]) -> str:
    """
    Construct a name for the configured model of the format {model_type}.{bands}.{experiment_tag}.{model_id}
    :param config: The model configuration
    :return: The formatted name of the configured model
    """
    # If model_type Is A Checkpointed Model, We Return Its Name As-Is
    model_type = get_model_type(config)
    saved_models = os.listdir("checkpoints")
    if model_type in saved_models:
        return model_type
    
    # Otherwise, We Create A New Unique Name
    partial_name = f"{model_type}.{'_'.join(get_bands(config))}.{get_experiment_tag(config)}".lower()
    existing_ids = [int(model.split(".")[-1]) for model in saved_models if partial_name in model]
    possible_ids = [possible_id for possible_id in range(0, max(existing_ids) + 2) if possible_id not in existing_ids] if existing_ids else [0]
    model_id = possible_ids[0]
    return f"{partial_name}.{model_id}"
