from models.unet import unet, unet_rgb, unet_multispectral
from models.vnet import vnet_rgb, vnet_multispectral
from models.unet_plus import unet_plus_rgb, unet_plus_multispectral
from models.unet_3_plus import unet_3_plus_multispectral, unet_3_plus_rgb
from models.r2u_net import r2_unet_rgb, r2_unet_multispectral
from models.resunet import resunet_rgb, resunet_multispectral
from models.u2net import u2net_rgb, u2net_multispectral
from models.transunet import transunet_rgb, transunet_multispectral
from models.swin_unet import swin_unet_rgb, swin_unet_multispectral
from models.att_unet import att_unet_rgb, att_unet_multispectral
from models.layers import fusion_head


def get_model(config):
    model = config["hyperparameters"]["model"]
    models = {"unet_multi": unet_multispectral,
              "unet_rgb": unet_rgb,
              "vnet_multi": vnet_multispectral,
              "vnet_rgb": vnet_rgb,
              "unet_plus_multi": unet_plus_multispectral,
              "unet_plus_rgb": unet_plus_rgb,
              "unet_3_plus_multi": unet_3_plus_multispectral,
              "unet_3_plus_rgb": unet_3_plus_rgb,
              "r2_unet_multi": r2_unet_multispectral,
              "r2_unet_rgb": r2_unet_rgb,
              "resunet_multi": resunet_multispectral,
              "resunet_rgb": resunet_rgb,
              "u2net_multi": u2net_multispectral,
              "u2net_rgb": u2net_rgb,
              "transunet_multi": transunet_multispectral,
              "transunet_rgb": transunet_rgb,
              "swin_unet_multi": swin_unet_multispectral,
              "swin_unet_rgb": swin_unet_rgb,
              "att_unet_multi": att_unet_multispectral,
              "att_unet_rgb": att_unet_rgb,
              }
    return models[model](config) if model in models else unet_rgb(config)
