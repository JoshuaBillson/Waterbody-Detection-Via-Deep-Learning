from models.utils import rgb_model
from keras_unet_collection.models import u2net_2d


def u2net_rgb(config):
    """
        Summary:
            Create VNET from keras unet collection library model object
        Arguments:
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
    model = u2net_2d((config['patch_size'], config['patch_size'], 3), n_labels=1, filter_num_down=[64, 128, 256, 512],
                     filter_num_up=[64, 64, 128, 256], filter_mid_num_down=[32, 32, 64, 128], filter_mid_num_up=[16, 32, 64, 128],
                     filter_4f_num=[512, 512], filter_4f_mid_num=[256, 256], activation='ReLU', output_activation='Sigmoid',
                     batch_norm=True, pool=False, unpool=False, deep_supervision=True, name='u2net')
    model.summary()
    return rgb_model(model, config)


def u2net_multispectral(config):
    pass
