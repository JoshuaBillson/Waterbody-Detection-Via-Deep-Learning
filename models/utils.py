import os
from typing import Dict, Any
from tensorflow.keras.layers import Conv2D, Input, concatenate
from tensorflow.keras.models import Model
from models.layers import rgb_nir_swir_input_layer
from backend.config import get_model_type, get_bands, get_patch_size, get_experiment_tag


def assemble_model(base_model: Model, config: Dict[str, Any]) -> Model:
    """
    Takes a base model and modifies its input and output layers as appropriate for the specified input bands 
    :param base_model: The base model (U-Net, U2-Net, U-Net++, etc.) that we want to extend
    :param config: The model configuration
    :return: The final model.
    """
    model_constructors = {
        "RGB+NIR+SWIR": rgb_nir_swir_model,
        "RGB+NIR": rgb_nir_model,
        "RGB": rgb_model,
        "NIR": nir_model,
        "SWIR": swir_model,
    }
    return model_constructors["+".join(get_bands(config))](base_model, config)


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
    rgb_input = Input(shape=(get_patch_size(config), get_patch_size(config), 3))
    outputs = model(rgb_input)
    return Model(inputs=rgb_input, outputs=outputs, name=get_model_name(config))


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
    nir_input = Input(shape=(get_patch_size(config), get_patch_size(config), 1))
    outputs = model(nir_input)
    return Model(inputs=nir_input, outputs=outputs, name=get_model_name(config))


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
    swir_input = Input(shape=(get_patch_size(config), get_patch_size(config), 1))
    outputs = model(swir_input)
    return Model(inputs=swir_input, outputs=outputs, name=get_model_name(config))


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
    outputs = model(concat)
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
    fusion_head = rgb_nir_swir_input_layer(rgb_inputs, nir_inputs, swir_inputs, config)
    outputs = model(fusion_head)
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


def evaluate_model(model: Model, test_data):
    """
    Evaluate the given model on the provided test data
    :param model: The Keras model we want to evaluate
    :param test_data: The data on which to evaluate the model
    """
    results = model.evaluate(test_data)
    print(f"\nEVALUATION SUMMARY FOR {model.name.upper()}")
    for metric, value in zip(model.metrics_names, results):
        print(metric, value)
    return results
