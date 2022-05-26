from models.utils import rgb_model
from keras_unet_collection.models import unet_plus_2d


def unet_plus_rgb(config):
    """
        Summary:
            Create VNET from keras unet collection library model object
        Arguments:
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
    model = unet_plus_2d((config['patch_size'], config['patch_size'], 3), [64, 128, 256, 512],
                         n_labels=1, stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid',
                         batch_norm=True, pool='max', unpool='nearest', name='r2unet', backbone="ResNet152")
    model.summary()
    return rgb_model(model, config)


def unet_plus_multispectral(config):
    pass
