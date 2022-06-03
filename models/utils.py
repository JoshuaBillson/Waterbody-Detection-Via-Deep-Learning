import os
import numpy as np
import shutil
from typing import Dict, Any, List
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input
from tensorflow.keras.models import Model
from models.layers import rgb_input_layer, nir_input_layer, swir_input_layer, rgb_nir_input_layer
from config import get_model_type, get_bands, get_backbone, get_patch_size, get_experiment_tag
from backend.data_loader import DataLoader
from backend.metrics import MIoU
import matplotlib.pyplot as plt


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
    patch_size = config["patch_size"]
    inputs = Input(shape=(patch_size, patch_size, 3))
    x = rgb_input_layer(inputs)
    outputs = model(x)
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
    patch_size = config["patch_size"]
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
    inputs = swir_input_layer(config)
    outputs = model(inputs)
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
    inputs = rgb_nir_input_layer(config)
    outputs = model(inputs)
    return Model(inputs=inputs, outputs=outputs, name=get_model_name(config))


def replace_output(base_model: Model, config: Dict[str, Any]) -> Model:
    """
    Replace the output layer of the given base model. If the input bands include SWIR, we need to upsample the mask by a factor of 2
    :param base_model: The base model (U-Net, U2-Net, U-Net++, etc.) whose output layer we want to replace
    :param config: The model configuration
    :return: The final model.
    """
    if False:
        x = base_model.layers[-3].output
        up_sample = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
        outputs = Conv2D(1, kernel_size=(1, 1), name="out", activation='sigmoid', dtype="float32")(up_sample)  # create new last layer
        model = Model(inputs=base_model.input, outputs=outputs, name=f"{get_model_type(config)}_base")
        model.summary()
        return model
    x = base_model.layers[-3].output
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
    partial_name = f"{model_type}.{'+'.join(get_bands(config))}.{get_experiment_tag(config)}".lower()
    existing_ids = [int(model.split(".")[-1]) for model in saved_models if partial_name in model]
    possible_ids = [possible_id for possible_id in range(0, max(existing_ids) + 2) if possible_id not in existing_ids] if existing_ids else [0]
    model_id = possible_ids[0]
    return f"{partial_name}.{model_id}"


def predict_batch(batch: List[int], data_loader: DataLoader, model: Model, config: Dict[str, Any], directory: str, threshold) -> None:
    """
    Predict on a batch of feature samples and save the resulting prediction to disk alongside its mask and MIoU performance
    :param batch: A list of patch indexes on which we want to predict
    :param data_loader: An object for reading patches from the disk
    :param model: The model whose prediction we are interested in
    :param config: The script configuration stored as a dictionary; typically read from an external file
    :param directory: The name of the directory in which we want to save the model predictions
    :param threshold: We filter out patches whose water content percentage is below this value
    :return: Nothing
    """
    # Load Batch
    features, masks, indices = data_loader.get_batch(batch, get_bands(config), threshold=threshold)
    
    # Get Predictions
    predictions = []
    for feature_index in range(len(features[0])):
        prediction = model.predict([feature[feature_index:feature_index+1] for feature in features])
        predictions.append(prediction[0, ...])

    # Create Directory To Save Predictions
    if directory not in os.listdir():
        os.mkdir(directory)
    if model.name in os.listdir(directory):
        shutil.rmtree(f"{directory}/{model.name}")
    os.mkdir(f"{directory}/{model.name}")

    # Save Model Predictions To Disk
    for prediction, mask, index in zip(predictions, masks, indices):
        miou = MIoU(mask.astype("float32"), prediction)
        fig, axs = plt.subplots(1, 2)
        fig.tight_layout()
        axs[0].imshow(mask)
        axs[0].set_title("Ground Truth")
        axs[1].imshow(np.where(prediction < 0.5, 0, 1))
        axs[1].set_title(f"{model.name} ({miou.numpy():.3f})")
        plt.savefig(f"{directory}/{model.name}/prediction.{index}.png", dpi=300, bbox_inches='tight')
        plt.cla()
        plt.close()
