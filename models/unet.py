from typing import Dict, Any
from tensorflow.keras.models import Model
from keras_unet_collection.models import unet_2d
from models.utils import assemble_model
from config import get_model_config


def unet(config: Dict[str, Any]) -> Model:
    """
    Construct a U-Net model that takes the input bands and uses the backbone specified in the config
    :param config: The model configuration
    :return: The assembled U-Net model
    """
    # Get Backbone And Input Channels
    input_channels, backbone = get_model_config(config)

    # Construct Base Model
    model = unet_2d(input_size=(config["patch_size"], config["patch_size"], input_channels), filter_num=[64, 128, 256, 512, 1024], n_labels=2, backbone=backbone)
    model.summary()

    # Replace Input And Output Layers
    return assemble_model(model, config)
