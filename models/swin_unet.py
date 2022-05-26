from models.utils import rgb_model
from keras_unet_collection.models import swin_unet_2d


def swin_unet_rgb(config):
    """
        Summary:
            Create VNET from keras unet collection library model object
        Arguments:
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
    model = swin_unet_2d((config['patch_size'], config['patch_size'], 3), filter_num_begin=64, n_labels=2, depth=4,
                         stack_num_down=2, stack_num_up=2, patch_size=(2, 2), num_heads=[4, 8, 8, 8], window_size=[4, 2, 2, 2],
                         num_mlp=512, output_activation='Sigmoid', shift_window=True, name='swin_unet')
    model.summary()
    return rgb_model(model, config)


def swin_unet_multispectral(config):
    pass
