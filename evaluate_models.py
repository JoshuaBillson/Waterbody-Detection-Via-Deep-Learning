import os
import sys
import random
import numpy as np
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt

from backend.utils import adjust_rgb
from backend.data_loader import DataLoader
from models import get_model
from backend.config import get_timestamp, get_bands

BIZ = [ 10202, 10204, 27504, 11101, 16601, 16602, 16603, 16604, 21602, 15804, 26001, 26002, 36601, 36603, 32501, 25401, 25404, 29504, 13501, 8804 , 39301, 26101, 26102, 35401, 35403, 35404, 20301, 20302, 20303, 20304, 10601, 17003, 10801, 15701, 15702, 15704, 28801, 28802, 28803, 28804, 7601 , 20601, 15602, 28702, 28703, 28704, 6104 , 37001, 37004, 19401, 16501, 16503, 9201 , 33201, 33202, 33203, 33204, 30701, 30703, 39402, 39404, 4701 , 24903, 7801 , 26504, 28901, 28902, 28904, 18702, 11302, 11303, 35002, ]
BAZ = [BIZ[i] for i in [5, 7, 18, 19, 21, 23, 24, 25, 26, 28, 46, 50, 51, 68]]
ONE = [10202, 10204, 9002, 27502, 27504, 11101, 11102, 11103, 16601, 16602, 16604, 21601, 21603, 21604, 15801, 15802, 15803, 15804, 20901, 20903, 20904, 26002, 26004] 
TWO = [36602, 24001, 24003, 24004, 32503, 25402, 25404, 29503, 36202, 36204, 13501, 13502, 13504, 8802, 8803, 8804, 27402, 39302, 39304, 28001, 28002, 28003, 28004] 
THREE = [26101, 26102, 26103, 26104, 35402, 35403, 35404, 21803, 21804, 20301, 20302, 20303, 10601, 10603, 17001, 17002, 17003, 17004, 10801, 3603, 20502, 20503, 15702]
FOUR = [15704, 28801, 28803, 28804, 7601, 7602, 27801, 27803, 27804, 17701, 17702, 17703, 37501, 37503, 37504, 17801, 17802, 17804, 20601, 20602, 20603, 20604, 15602, 15603]
FIVE = [31701, 31702, 31703, 1601, 1602, 1603, 13301, 13302, 13304, 28701, 28702, 28704, 6101, 6103, 6104, 37001, 37002, 37003, 37004, 19403, 16501, 16502, 39001, 39002, 39003]
SIX = [39004, 9201, 9203, 9204, 33201, 33202, 33203, 33204, 30702, 39401, 39402, 39403, 39404, 4702, 4703, 24902, 24903, 24904, 7801, 7802, 28901, 28902, 28904, 18701, 18702, 18703, 18704, 11301, 11302, 11303, 35001, 35003, 35004]
FOO = [27402, 17001, 3603, 6104, 13502, 35002]
# FOO = [27804, 27402, 26102, 17001, 3603, 20602, 1601, 6104, 20904, 13502, 36201, 33203, 35002]

FONTSIZE = 4

MODELS = {
    "PCT": [
        {"patch_size": 512, "hyperparameters": {"model": "att_unet.rgb_nir_swir.baseline.1", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "att_unet.rgb_nir_swir.waterbody_transfer_5.1", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        # {"patch_size": 512, "hyperparameters": {"model": "unet.rgb_nir_swir.baseline.3", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        # {"patch_size": 512, "hyperparameters": {"model": "unet.rgb_nir_swir.waterbody_transfer_10.2", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        # {"patch_size": 512, "hyperparameters": {"model": "unet_plus.rgb_nir_swir.baseline.4", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        # {"patch_size": 512, "hyperparameters": {"model": "unet_plus.rgb_nir_swir.waterbody_transfer_5.1", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "deeplab.rgb_nir_swir.baseline.1", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "deeplab.rgb_nir_swir.waterbody_transfer_15.1", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "r2_unet.rgb_nir_swir.baseline.1", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "r2_unet.rgb_nir_swir.waterbody_transfer_15.0", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        ],
    "MODEL_COMPARISON": [
        # {"patch_size": 512, "hyperparameters": {"model": "link_net.rgb_nir_swir.baseline.3", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "fpn.rgb_nir_swir.baseline.2", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "deeplab.rgb_nir_swir.baseline.1", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "swin_unet.rgb_nir_swir.baseline.2", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        # {"patch_size": 512, "hyperparameters": {"model": "vnet.rgb_nir_swir.baseline.1", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "unet.rgb_nir_swir.baseline.3", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "unet_plus.rgb_nir_swir.baseline.4", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "att_unet.rgb_nir_swir.baseline.1", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "deeplab.rgb_nir_swir.baseline_imagenet_backbone.0", "bands": ["RGB", "NIR", "SWIR"], "backbone": "ResNet50", "fusion_head": "prism"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "r2_unet.rgb_nir_swir.baseline.2", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        ], 
    "INPUT_BAND_COMPARISON": [
        {"patch_size": 512, "hyperparameters": {"model": "fpn.rgb.baseline.2", "bands": ["RGB"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "fpn.nir.baseline.1", "bands": ["NIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "fpn.rgb_nir_swir.baseline.2", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "unet.rgb.baseline.3", "bands": ["RGB"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "unet.nir.baseline.2", "bands": ["NIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "unet.rgb_nir_swir.baseline.3", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "r2_unet.rgb.baseline.1", "bands": ["RGB"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "r2_unet.nir.baseline.1", "bands": ["NIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "r2_unet.rgb_nir_swir.baseline.2", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        ],
    "LOSS_COMPARISON": [
        {"patch_size": 512, "hyperparameters": {"model": "r2_unet.rgb_nir_swir.bce.1", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "r2_unet.rgb_nir_swir.weighted_bce.0", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "r2_unet.rgb_nir_swir.focal_loss.0", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "r2_unet.rgb_nir_swir.dice.0", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "r2_unet.rgb_nir_swir.jaccard.0", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "r2_unet.rgb_nir_swir.dice_bce.0", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "r2_unet.rgb_nir_swir.jaccard_bce.0", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "r2_unet.rgb_nir_swir.tversky.0", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        {"patch_size": 512, "hyperparameters": {"model": "r2_unet.rgb_nir_swir.focal_tversky.0", "bands": ["RGB", "NIR", "SWIR"], "backbone": None, "fusion_head": "naive"}}, 
        ],
    }

MODEL_NAMES = {
    "PCT": ["Attention U-Net", "Attention U-Net (PCT)", "DeepLabv3+", "DeepLabv3+ (PCT)", "R2U-Net", "R2U-Net (PCT)"],
    "MODEL_COMPARISON": ["FPN", "DeepLabv3+", "Swin-Unet", "U-Net", "U-Net++", "Attention U-Net", "DeepLabv3+ (ImageNet)", "R2U-Net"], 
    "INPUT_BAND_COMPARISON": ["FPN (RGB)", "FPN (NIR)", "FPN (Multi)", "U-Net (RGB)", "U-Net (NIR)", "U-Net (Multi)", "R2U-Net (RGB)", "R2U-Net (NIR)", "R2U-Net (Multi)"], 
    "LOSS_COMPARISON": ["BCE", "Weighted BCE", "Focal", "Dice", "Jaccard", "Dice + BCE", "Jaccard + BCE", "Tversky", "Focal Tversky"], 
    }

SAMPLES = {
    "PCT": [16602, 20303, 16501, 18702],
    "MODEL_COMPARISON": [20904, 13502, 28901, 17004, 20502, 36201],
    "INPUT_BAND_COMPARISON": [39301, 27802, 19404, 35002, 28903, 24904],
    "LOSS_COMPARISON": FOO,
    }

TIMESTAMP = 1
BANDS = ["RGB", "NIR", "SWIR"]
EXPERIMENT = "MODEL_COMPARISON"

def main():
    # Create Batch To Evaluate On
    samples = SAMPLES[EXPERIMENT]
    models = MODELS[EXPERIMENT]
    model_names = MODEL_NAMES[EXPERIMENT]
    loader = DataLoader(timestamp=TIMESTAMP)
    mc_wbdn_loader = DataLoader(timestamp=TIMESTAMP, upscale_swir=False)

    # Create Directory To Save Predictions
    directory = ""
    for path in ("images", "model_evaluation"):
        if path not in os.listdir() and path not in os.listdir(directory):
            os.mkdir(directory + f"{path}/")
        directory += f"{path}/"

    # Create Figure
    _, axs = plt.subplots(len(samples), len(BANDS) + len(models) + 1, figsize=(len(models) + len(BANDS) + 1, len(samples)))

    # Get Features
    rgb_features = []
    nir_features = []
    multispectral_features = []
    mc_wbdn_features = []
    for sample in samples:
        rgb_features.append(loader.get_features(sample, ["RGB"]))
        nir_features.append(loader.get_features(sample, ["NIR"]))
        multispectral_features.append(loader.get_features(sample, ["RGB", "NIR", "SWIR"]))
        mc_wbdn_features.append(mc_wbdn_loader.get_features(sample, ["RGB", "NIR", "SWIR"]))

    # Get Predictions
    predictions = [[] for _ in samples]
    for model_config in models:
        model = get_model(model_config)
        for row, rgb_feature, nir_feature, multispectral_feature, mc_wbdn_feature in zip(range(len(rgb_features)), rgb_features, nir_features, multispectral_features, mc_wbdn_features):
            if "mc_wbdn" in model.name:
                prediction = model.predict([np.array([DataLoader.normalize_channels(mc_wbdn_feature[band].astype("float32"))]) for band in ["RGB", 'NIR', "SWIR"]])
            elif ".rgb_nir_swir." in model.name:
                prediction = model.predict([np.array([DataLoader.normalize_channels(multispectral_feature[band].astype("float32"))]) for band in ["RGB", 'NIR', "SWIR"]])
            elif ".nir." in model.name:
                prediction = model.predict([np.array([DataLoader.normalize_channels(nir_feature[band].astype("float32"))]) for band in ["NIR"]])
            elif ".rgb." in model.name:
                prediction = model.predict([np.array([DataLoader.normalize_channels(rgb_feature[band].astype("float32"))]) for band in ["RGB"]])
            predictions[row] += [np.where(prediction < 0.5, 0, 1)[0]]

    # Plot Features For Each Sample
    for row, feature in enumerate(multispectral_features):
        for col, band in enumerate(BANDS + ["mask"]):
            axs[row][col].imshow(adjust_rgb(feature[band]) if band == "RGB" else feature[band])
            axs[row][col].axis("off")
            if row == 0:
                axs[0][col].set_title(band if band != "mask" else "Ground Truth", fontsize=FONTSIZE, fontweight="bold")
            if col == 0:
                axs[row][0].set_ylabel(SAMPLES[EXPERIMENT][row], rotation=0, size='large')
    
    # Plot Predictions
    for row, prediction in enumerate(predictions):
        for col, model in enumerate(model_names):
            col_offset = len(BANDS) + 1
            axs[row][col + col_offset].imshow(prediction[col])
            axs[row][col + col_offset].axis("off")
            if row == 0:
                axs[0][col + col_offset].set_title(model, fontsize=FONTSIZE, fontweight="bold")

    # Save Figure To Disk
    plt.subplots_adjust(wspace=0, hspace=0.050)
    plt.savefig(f"{directory}{EXPERIMENT.lower()}.png", dpi=800, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Set Visible GPU
    args = sys.argv
    GPU = args[1] if len(args) > 1 else "0" 
    os.environ["CUDA_VISIBLE_DEVICES"]=GPU

    # Use Mixed Precision
    # mixed_precision.set_global_policy('mixed_float16')

    # Run Script
    main()
