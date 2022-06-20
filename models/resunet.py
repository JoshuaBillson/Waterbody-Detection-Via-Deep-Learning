from keras_unet_collection.models import resunet_a_2d
from models.utils import assemble_model
from backend.config import get_input_channels


def resunet(config):
    """
    Construct a ResUNet model that takes the input bands and uses the backbone specified in the config
    :param config: The model configuration
    :return: The assembled ResUNet model
    """
    # Construct Base Model
    model = resunet_a_2d((config['patch_size'], config['patch_size'], get_input_channels(config)), [32, 64, 128, 256, 512],
                         dilation_num=[1, 3, 15, 31], n_labels=1, aspp_num_down=256, aspp_num_up=128, activation='ReLU',
                         output_activation='Sigmoid', batch_norm=True, pool=False, unpool='nearest', name='resunet')
    model.summary()

    # Replace Input And Output Layers
    return assemble_model(model, config)
