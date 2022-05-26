from models.utils import rgb_model
from keras_unet_collection.models import transunet_2d


def transunet_rgb(config):
    """
        Summary:
            Create VNET from keras unet collection library model object
        Arguments:
            Model configuration from config.yaml
        Return:
            Keras.model object
    """
    model = transunet_2d((config['patch_size'], config['patch_size'], 3), filter_num=[64, 128, 256, 512], n_labels=1,
                         stack_num_down=2, stack_num_up=2, embed_dim=768, num_mlp=3072, num_heads=12, num_transformer=12,
                         activation='ReLU', mlp_activation='GELU', output_activation='Softmax', batch_norm=True, pool=True,
                         unpool='bilinear', name='transunet', backbone="ResNet152")
    model.summary()
    return rgb_model(model, config)


def transunet_multispectral(config):
    pass
