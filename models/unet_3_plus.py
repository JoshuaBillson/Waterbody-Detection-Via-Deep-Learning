from keras_unet_collection.models import unet_3plus_2d
from models.utils import assemble_model
from backend.config import get_model_config


def unet_3_plus(config):
    """
    Construct a U-Net 3+ model that takes the input bands and uses the backbone specified in the config
    :param config: The model configuration
    :return: The assembled U-Net 3+ model
    """
    # Get Backbone And Input Channels
    channels, backbone = get_model_config(config)

    # Construct Base Model
    model = unet_3plus_2d((config['patch_size'], config['patch_size'], channels), n_labels=1,
                          filter_num_down=[64, 128, 256, 512], filter_num_skip='auto', filter_num_aggregate='auto',
                          stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid', batch_norm=True,
                          pool='max', unpool=False, deep_supervision=True, name='unet3plus', backbone=backbone)
    model.summary()

    # Replace Input And Output Layers
    return assemble_model(model, config)
