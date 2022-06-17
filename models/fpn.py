from typing import Dict, Any
from tensorflow.keras.models import Model
from segmentation_models import FPN
from backend.config import get_model_config
from models.utils import assemble_model


def fpn(config: Dict[str, Any]) -> Model:
    """
    Construct a  V-Net model that takes the input bands and uses the backbone specified in the config
    :param config: The model configuration
    :return: The assembled V-Net model
    """
    # Get Backbone And Input Channels
    input_channels, backbone = get_model_config(config)

    # Construct Base Model
    backbone = backbone if backbone is not None else "efficientnetb0"
    model = FPN(backbone_name=backbone, input_shape=(config["patch_size"], config['patch_size'], input_channels), classes=1, activation='sigmoid', encoder_weights=None, weights=None)
    model.summary()

    # Replace Input And Output Layers
    return assemble_model(model, config)
