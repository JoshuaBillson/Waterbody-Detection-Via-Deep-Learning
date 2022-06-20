import os
import json
import sys
from time import sleep
from typing import Dict, Sequence
import numpy as np
import cv2
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt

from backend.utils import adjust_rgb
from backend.data_loader import DataLoader
from models import get_model
from backend.config import get_timestamp, get_bands


def get_features(data_loader: DataLoader, patch: int, bands: Sequence[str]) -> Dict[str, np.ndarray]:
    # Get Mask
    features = {"mask": data_loader.get_mask(patch)}

    # Get RGB Features
    if "RGB" in bands:
        features["RGB"] = data_loader.get_rgb_features(patch, preprocess_img=False)

    # Get NIR Features
    if "NIR" in bands:
        features["NIR"] = data_loader.get_nir_features(patch, preprocess_img=False)

    # Get SWIR Features
    if "SWIR" in bands:
        features["SWIR"] = data_loader.get_swir_features(patch, preprocess_img=False)
        features["SWIR"] = np.resize(cv2.resize(features["SWIR"], (512, 512), interpolation = cv2.INTER_AREA), (512, 512, 1))

    return features


def main():
    # Get Project Configuration
    with open('config.json') as f:
        config = json.loads(f.read())
        bands = get_bands(config)

    # Create Batch To Evaluate On
    samples = [2325, 1795, 1749, 1883]
    models = ["vnet.rgb.baseline.1", "vnet.rgb.modified_tanimoto_with_bce.1", "unet_plus.rgb.baseline.1", "unet_plus.rgb.modified_tanimoto_with_bce.1"]
    model_names = ["V-Net (Baseline)", "V-Net (Modified Tanimoto)", "U-Net++ (Baseline)", "U-Net++ (Modified Tanimoto)"]
    loader = DataLoader(timestamp=get_timestamp(config))

    # Create Directory To Save Predictions
    directory = ""
    for path in ("images", "model_evaluation"):
        if path not in os.listdir() and path not in os.listdir(directory):
            os.mkdir(directory + f"{path}/")
        directory += f"{path}/"

    # Create Figure
    _, axs = plt.subplots(len(samples), len(bands) + len(models) + 1)

    # Get Features
    features = []
    for sample in samples:
        features.append(get_features(loader, sample, bands))

    # Get Predictions
    predictions = [[] for _ in samples]
    for model_name in models:
        config["hyperparameters"]["model"] = model_name
        model = get_model(config)
        for row, feature in enumerate(features):
            prediction = model.predict([np.array([DataLoader.normalize_channels(feature[band].astype("float32"))]) for band in bands])
            predictions[row] += [np.where(prediction < 0.5, 0, 1)[0]]

    # Plot Features For Each Sample
    for row, feature in enumerate(features):
        for col, band in enumerate(bands + ["mask"]):
            axs[row][col].imshow(adjust_rgb(feature[band]) if band == "RGB" else feature[band])
            axs[row][col].axis("off")
            if row == 0:
                axs[0][col].set_title(band if band != "mask" else "Ground Truth", fontsize=6)
    
    # Plot Predictions
    for row, prediction in enumerate(predictions):
        for col, model in enumerate(model_names):
            col_offset = len(bands) + 1
            axs[row][col + col_offset].imshow(prediction[col])
            axs[row][col + col_offset].axis("off")
            if row == 0:
                axs[0][col + col_offset].set_title(model, fontsize=6)

    # Save Figure To Disk
    print(f"{directory}comparison.png")
    sleep(5)
    plt.tight_layout()
    plt.savefig(f"{directory}{sys.argv[1]}.png", dpi=800, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Set Visible GPU
    args = sys.argv
    GPU = args[2] if len(args) > 2 else "0" 
    os.environ["CUDA_VISIBLE_DEVICES"]=GPU

    # Use Mixed Precision
    mixed_precision.set_global_policy('mixed_float16')

    # Run Script
    main()
