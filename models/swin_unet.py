from typing import Dict, Any
from keras_unet_collection.models import swin_unet_2d
from tensorflow.keras.models import Model
from models.utils import assemble_model
from backend.config import get_input_channels


def swin_unet(config: Dict[str, Any]) -> Model:
    """
    Construct a Swin-UNet model that takes the input bands and uses the backbone specified in the config
    :param config: The model configuration
    :return: The assembled Swin-UNet model
    """
    # Construct Base Model
    model = swin_unet_2d((config['patch_size'], config['patch_size'], get_input_channels(config)), filter_num_begin=64, n_labels=1, depth=4,
                         stack_num_down=2, stack_num_up=2, patch_size=(2, 2), num_heads=[4, 8, 8, 8], window_size=[4, 2, 2, 2],
                         num_mlp=512, output_activation='Sigmoid', shift_window=True, name='swin_unet')
    model.summary()

    # Replace Input And Output Layers
    return assemble_model(model, config)
