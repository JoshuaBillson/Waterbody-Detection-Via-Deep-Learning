from models.utils import rgb_model
from keras_unet_collection.models import att_unet_2d


def att_unet_rgb(config):
    """
        Summary:
            Create VNET from keras unet collection library model object
        Arguments:
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
    model = att_unet_2d((config['patch_size'], config['patch_size'], 3), [64, 128, 256, 512], n_labels=2, stack_num_down=2,
                        stack_num_up=2, activation='ReLU', atten_activation='ReLU', attention='add', output_activation="Sigmoid",
                        batch_norm=True, pool=False, unpool='bilinear', name='attunet', backbone="ResNet152")
    model.summary()
    return rgb_model(model, config)


def att_unet_multispectral(config):
    pass
