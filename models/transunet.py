from keras_unet_collection.models import transunet_2d
from models.utils import assemble_model
from backend.config import get_model_config


def transunet(config):
    """
    Construct a TransUNet model that takes the input bands and uses the backbone specified in the config
    :param config: The model configuration
    :return: The assembled TransUNet model
    """
    # Get Backbone And Input Channels
    input_channels, backbone = get_model_config(config)

    # Construct Base Model
    model = transunet_2d((config['patch_size'], config['patch_size'], input_channels), filter_num=[64, 128, 256, 512], n_labels=1,
                         stack_num_down=2, stack_num_up=2, embed_dim=768, num_mlp=3072, num_heads=12, num_transformer=12,
                         activation='ReLU', mlp_activation='GELU', output_activation='Softmax', batch_norm=True, pool=True,
                         unpool='bilinear', name='transunet', backbone=backbone)
    model.summary()

    # Replace Input And Output Layers
    return assemble_model(model, config)
