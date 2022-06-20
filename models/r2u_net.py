from keras_unet_collection.models import r2_unet_2d
from models.utils import assemble_model
from backend.config import get_input_channels


def r2_unet(config):
    """
    Construct an R2U-Net model that takes the input bands and uses the backbone specified in the config
    :param config: The model configuration
    :return: The assembled R2U-Net model
    """
    # Construct Base Model
    model = r2_unet_2d((config['patch_size'], config['patch_size'], get_input_channels(config)), [64, 128, 256, 512], n_labels=1,
                       stack_num_down=2, stack_num_up=1, recur_num=2, activation='ReLU', output_activation='Sigmoid',
                       batch_norm=True, pool='max', unpool='nearest', name='r2unet')
    model.summary()

    # Replace Input And Output Layers
    return assemble_model(model, config)
