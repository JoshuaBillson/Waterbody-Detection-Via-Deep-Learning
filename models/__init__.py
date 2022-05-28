from typing import Dict, Any
from tensorflow.keras.models import Model
from models.unet import unet
from models.vnet import vnet
from models.unet_plus import unet_plus
from models.unet_3_plus import unet_3_plus
from models.r2u_net import r2_unet
from models.resunet import resunet
from models.u2net import u2net
from models.transunet import transunet
from models.swin_unet import swin_unet
from models.att_unet import att_unet
from config import get_model_type


def get_model(config: Dict[str, Any]) -> Model:
    """
    Takes the project configuration and returns the desired model
    :param config: A dictionary representing the project config which is typically loaded from an external file
    :returns: The model we want to train
    """
    model = get_model_type(config)
    models = {"unet": unet,
              "vnet": vnet,
              "unet_plus": unet_plus,
              "unet_3_plus": unet_3_plus,
              "r2_unet": r2_unet,
              "resunet": resunet,
              "u2net": u2net,
              "transunet": transunet,
              "swin_unet": swin_unet,
              "att_unet": att_unet,
              }
    if model in models:
        return models[model](config)
    raise ValueError(f"Error: Invalid Model Received (Must Be One Of {list(models.keys())})!")
