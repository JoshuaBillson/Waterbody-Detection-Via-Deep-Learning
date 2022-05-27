from keras_unet_collection.models import u2net_2d
from models.utils import assemble_model
from config import get_model_config


def u2net(config):
    """
    Construct a U2-Net model that takes the input bands and uses the backbone specified in the config
    :param config: The model configuration
    :return: The assembled U2-Net model
    """
    # Get Backbone And Input Channels
    input_channels, _ = get_model_config(config)

    # Construct Base Model
    model = u2net_2d((config['patch_size'], config['patch_size'], input_channels), n_labels=1, filter_num_down=[64, 128, 256, 512],
                     filter_num_up=[64, 64, 128, 256], filter_mid_num_down=[32, 32, 64, 128], filter_mid_num_up=[16, 32, 64, 128],
                     filter_4f_num=[512, 512], filter_4f_mid_num=[256, 256], activation='ReLU', output_activation='Sigmoid',
                     batch_norm=True, pool=False, unpool=False, deep_supervision=True, name='u2net')
    model.summary()

    # Replace Input And Output Layers
    return assemble_model(model, config)
