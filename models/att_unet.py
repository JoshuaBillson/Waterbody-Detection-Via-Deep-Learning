from keras_unet_collection.models import att_unet_2d
from backend.config import get_model_config
from models.utils import assemble_model


def att_unet(config):
    """
    Construct an Attention U-Net model that takes the input bands and uses the backbone specified in the config
    :param config: The model configuration
    :return: The assembled Attention U-Net model
    """
    # Get Backbone And Input Channels
    channels, backbone = get_model_config(config)

    # Construct Base Model
    model = att_unet_2d((config['patch_size'], config['patch_size'], channels), [64, 128, 256, 512], n_labels=1, stack_num_down=2,
                        stack_num_up=2, activation='ReLU', atten_activation='ReLU', attention='add', output_activation="Sigmoid",
                        batch_norm=True, pool=False, unpool='bilinear', name='attunet', backbone=backbone)
    model.summary()

    # Replace Input And Output Layers
    return assemble_model(model, config)
