from typing import Dict, Any
from tensorflow.keras.models import Model
from keras_unet_collection.models import vnet_2d
from backend.config import get_input_channels
from models.utils import assemble_model


def vnet(config: Dict[str, Any]) -> Model:
    """
    Construct a  V-Net model that takes the input bands and uses the backbone specified in the config
    :param config: The model configuration
    :return: The assembled V-Net model
    """
    # Construct Base Model
    model = vnet_2d((config["patch_size"], config['patch_size'], 125), filter_num=[32, 64, 128, 256, 512],
                    n_labels=1, res_num_ini=1, res_num_max=3, activation='PReLU', output_activation='Sigmoid',
                    batch_norm=True, pool=False, unpool=False, name='vnet')
    model.summary()

    # Replace Input And Output Layers
    return assemble_model(model, config)
