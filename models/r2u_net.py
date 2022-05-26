from models.utils import rgb_model
from keras_unet_collection.models import r2_unet_2d


def r2_unet_rgb(config):
    """
        Summary:
            Create VNET from keras unet collection library model object
        Arguments:
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
    model = r2_unet_2d((config['patch_size'], config['patch_size'], 3), [64, 128, 256, 512], n_labels=1,
                       stack_num_down=2, stack_num_up=1, recur_num=2, activation='ReLU', output_activation='Sigmoid',
                       batch_norm=True, pool='max', unpool='nearest', name='r2unet')
    model.summary()
    return rgb_model(model, config)


def r2_unet_multispectral(config):
    pass
