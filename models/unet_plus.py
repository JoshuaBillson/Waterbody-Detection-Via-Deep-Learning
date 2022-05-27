from keras_unet_collection.models import unet_plus_2d
from models.utils import assemble_model
from config import get_model_config


def unet_plus(config):
    """
    Construct a U-Net++ model that takes the input bands and uses the backbone specified in the config
    :param config: The model configuration
    :return: The assembled U-Net++ model
    """
    # Get Backbone And Input Channels
    input_channels, backbone = get_model_config(config)

    # Construct Base Model
    model = unet_plus_2d((config['patch_size'], config['patch_size'], input_channels), [64, 128, 256, 512],
                         n_labels=1, stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid',
                         batch_norm=True, pool='max', unpool='nearest', name='r2unet', backbone=backbone)
    model.summary()

    # Replace Input And Output Layers
    return assemble_model(model, config)
