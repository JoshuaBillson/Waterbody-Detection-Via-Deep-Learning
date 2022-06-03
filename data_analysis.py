import os
import sys
import shutil
import json
from backend.data_loader import DataLoader, analyze_dataset
from config import get_timestamp
import numpy as np
import matplotlib.pyplot as plt
from models import get_model
from config import get_bands


def on_epoch_end(data_loader: DataLoader, model, config):
    # Get All Patches With At Least 10% Water Coverage
    val_data = list(range(2700, 3001))
    masks, features, indices, bands = [], [], [], get_bands(config)
    for patch in val_data:
        mask = data_loader.get_mask(patch)
        if (np.sum(mask) / mask.size * 100.0) >= 5.0:

            # Keep Mask And Index
            masks.append(mask)
            indices.append(patch)

            # Keep RGB Feature
            feature_list = []
            if "RGB" in bands:
                rgb_feature = data_loader.get_rgb_features(patch)
                feature_list.append(np.reshape(rgb_feature, (1, rgb_feature.shape[0], rgb_feature.shape[1], rgb_feature.shape[2])))

            # Keep NIR Feature
            if "NIR" in bands:
                nir_feature = data_loader.get_nir_features(patch)
                feature_list.append(np.reshape(nir_feature, (1, nir_feature.shape[0], nir_feature.shape[1], nir_feature.shape[2])))

            # Keep SWIR Feature
            if "SWIR" in bands:
                swir_feature = data_loader.get_swir_features(patch)
                feature_list.append(np.reshape(swir_feature, (1, swir_feature.shape[0], swir_feature.shape[1], swir_feature.shape[2])))
                
            features.append(feature_list)

    # Create Predictions Directory
    if "predictions" not in os.listdir():
        os.mkdir("predictions")
    if model.name in os.listdir("predictions"):
        shutil.rmtree(f"predictions/{model.name}")
    os.mkdir(f"predictions/{model.name}")

    # Save Model Predictions To Disk
    print(len(masks))
    for mask, feature, index in zip(masks, features, indices):

        # Make Prediction
        print("Index:", index)
        prediction = model.predict(feature)
        
        # Plot Prediction
        fig, axs = plt.subplots(1, 2)
        fig.tight_layout()
        axs[0].imshow(mask)
        axs[0].set_title("Mask")
        axs[1].imshow(np.where(prediction < 0.5, 0, 1)[0, ...])
        axs[1].set_title(model.name)
        plt.savefig(f"predictions/{model.name}/prediction.{index + 1}.png", dpi=300, bbox_inches='tight')
        plt.cla()
        plt.close()

def main():
    # Get Project Configuration
    with open('config.json') as f:
        config = json.loads(f.read())
    
    model = get_model(config)

    # Create Data Loader
    loader = DataLoader(timestamp=get_timestamp(config))

    # Perform Analysis
    #analyze_dataset(loader)
    on_epoch_end(loader, model, config)



if __name__ == "__main__":
    args = sys.argv
    GPU = int(args[1]) if len(args) > 1 and args[1].isdigit() else 0
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{GPU}"
    main()
