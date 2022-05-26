from models.utils import rgb_model
from keras_unet_collection.models import resunet_a_2d


def resunet_rgb(config):
    """
        Summary:
            Create VNET from keras unet collection library model object
        Arguments:
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
    model = resunet_a_2d((config['patch_size'], config['patch_size'], 3), [32, 64, 128, 256, 512, 1024],
                         dilation_num=[1, 3, 15, 31], n_labels=1, aspp_num_down=256, aspp_num_up=128, activation='ReLU',
                         output_activation='Sigmoid', batch_norm=True, pool=False, unpool='nearest', name='resunet')
    model.summary()
    return rgb_model(model, config)


def resunet_multispectral(config):
    pass
