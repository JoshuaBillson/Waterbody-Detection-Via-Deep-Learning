import os
import sys
import copy
import shutil
import json
import random
from typing import Dict, List, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
from backend.data_loader import DataLoader
from backend.pipeline import ImgSequence, ImgSequence
from backend.config import get_timestamp_directory
from backend.utils import adjust_rgb

class GenerateTransferImgSequence(ImgSequence):
    """A class to demonstrate the waterbody transfer method."""

    def __init__(self, timestamp: int, tiles: List[int], batch_size: int = 32, bands: Sequence[str] = None, is_train: bool = True, random_subsample: bool = True):
        # Initialize Member Variables
        self.data_loader = DataLoader(timestamp, overlapping_patches=is_train, random_subsample=random_subsample)
        self.batch_size = batch_size
        self.bands = ["RGB"] if bands is None else bands
        self.indices = tiles 
        self.is_train = is_train
        self.transfer_tile_index = 401

        # If We Want To Apply Waterbody Transferrence, Locate All Patches With At Least 10% Water
        self.transfer_patches = []
        if self.is_train:
            for tile_index in self.indices:
                mask = self.data_loader.get_mask(tile_index)
                if 3.5 < self._water_content(mask):
                    print(tile_index, mask.shape, self._water_content(mask))
                    self.transfer_patches.append(tile_index)

    def transfer_waterbody(self, features: Dict[str, np.ndarray], timestamp_directory) -> None:
        """
        Given a destination patch, transfers a waterbody to the destination if the destination has a water content below a given threshold.
        :param features: A dictionary of input features (RGB, NIR, SWIR) and the mask to which we want to transfer a waterbody
        :param threshold: The water content threshold below which we apply waterbody transfer
        """
        # Acquire Probability Of Applying Transfer
        if (self._water_content(features["mask"]) < 2.0):

            # Transfer All Source Patches
            for _ in range(3):

                # Get Source Mask
                assert len(self.transfer_patches) > 0, "Error: Cannot Augment Dataset Without Transfer Patches!"
                source_index = self.transfer_patches[random.randint(0, len(self.transfer_patches) - 1)]
                source_features = self._get_features(source_index, subsample=False)
                source_mask = source_features["mask"]

                # Variables To Plot
                src_mask = copy.deepcopy(source_features["mask"])
                src_features = copy.deepcopy(source_features)
                dst_mask = copy.deepcopy(features["mask"])
                dst_features = copy.deepcopy(features)

                # Apply Waterbody Transfer To Each Feature Map
                for band in self.bands:

                    # Compute Different In Brightness Between Source And Destination
                    src_brightness = np.mean(source_features[band])
                    dst_brightness = np.mean(dst_features[band])
                    brightness_ratio = dst_brightness / src_brightness

                    # Get Source Feature
                    source_feature = (source_features[band] * brightness_ratio).astype("uint16")

                    # Extract Waterbody From Source Feature 
                    waterbody = source_mask * source_feature

                    # Remove Waterbody Region From Destination Feature
                    features[band] *= np.where(source_mask == 1, 0, 1).astype("uint16")

                    # Transfer Waterbody To Destination Feature Map
                    features[band] += waterbody

                    # Save Augmented Patch To Disk
                    filename = f"data/{timestamp_directory}/transplanted_tiles/{band.lower()}/{self.transfer_tile_index}.tif"
                    DataLoader.save_image(features[band], filename)
                    
                # Save Augmented Mask To Disk
                features["mask"] = np.where((features["mask"] + source_mask) >= 1, 1, 0).astype("uint16")
                filename = f"data/{timestamp_directory}/transplanted_tiles/mask/{self.transfer_tile_index}.tif"
                DataLoader.save_image(features["mask"], filename)

                # Plot Transfers
                self.plot_transfer(src_mask, src_features, dst_mask, dst_features, features["mask"], features)

                # Restore Original Destination Features And Mask
                features = dst_features


    def run_waterbody_transfer(self, config):
        """Demonstrate the waterbody tranfer method"""
        # Create Directory Plotting Transferred Images
        if "transfers" in os.listdir("images"):
            shutil.rmtree("images/transfers")
        os.mkdir("images/transfers")

        # Create Directory For Saving Transplanted Images
        if "transplanted_tiles" in os.listdir(f"data/{get_timestamp_directory(config)}"):
            shutil.rmtree(f"data/{get_timestamp_directory(config)}/transplanted_tiles")
        os.mkdir(f"data/{get_timestamp_directory(config)}/transplanted_tiles")

        # Create Directory For Each Band Of Transplanted Images
        for band in ("mask", "nir", "rgb", "swir"):
            os.mkdir(f"data/{get_timestamp_directory(config)}/transplanted_tiles/{band}")

        for patch in self.indices:
            features = self._get_features(patch, subsample=False)
            self.transfer_waterbody(features, get_timestamp_directory(config))

    def plot_transfer(self, src_mask, src_features, dst_mask, dst_features, aug_mask, aug_features) -> None:
        """
        Given a destination patch, transfers a waterbody to the destination if the destination has a water content below a given threshold.
        :param features: A dictionary of input features (RGB, NIR, SWIR) and the mask to which we want to transfer a waterbody
        :param index: The index of the destination patch; used for plotting the resulting transfer
        :param threshold: The water content threshold below which we apply waterbody transfer
        """
        _, axs = plt.subplots(len(self.bands), 6, figsize = (6, len(self.bands)))
        for row, band in enumerate(self.bands):
            print(row, band)

            axs[row][0].imshow(src_mask)
            axs[row][0].set_title("Src. Mask", fontsize=6)
            axs[row][0].axis("off")

            # Plot Source Features
            axs[row][1].imshow(adjust_rgb(src_features[band], gamma=0.5) if band == "RGB" else src_features[band])
            axs[row][1].set_title("Src. Features", fontsize=6)
            axs[row][1].axis("off")

            # Plot Destination Mask
            axs[row][2].imshow(dst_mask)
            axs[row][2].set_title("Dest. Mask", fontsize=6)
            axs[row][2].axis("off")

            # Plot Destination Features
            axs[row][3].imshow(adjust_rgb(dst_features[band], gamma=0.5) if band == "RGB" else dst_features[band])
            axs[row][3].set_title("Dest. Features", fontsize=6)
            axs[row][3].axis("off")

            # Plot Augmented Mask
            axs[row][4].imshow(aug_mask)
            axs[row][4].set_title("Final Mask", fontsize=6)
            axs[row][4].axis("off")
            
            # Plot Augmented Patch
            axs[row][5].imshow(adjust_rgb(aug_features[band], gamma=0.5) if band == "RGB" else aug_features[band])
            axs[row][5].set_title("Final Features", fontsize=6)
            axs[row][5].axis("off")

        # Save Figure
        plt.savefig(f"images/transfers/transfer_{self.transfer_tile_index}.png", dpi=500, bbox_inches='tight')
        self.transfer_tile_index += 1
        plt.close()


def main():
    # Get Project Configuration
    with open('config.json') as f:
        config = json.loads(f.read())
    
    # Get Training Batches
    with open('batches/tiles.json') as f:
        batches = json.loads(f.read())

    # Create Data Loader
    # data = GenerateTransferImgSequence(timestamp=get_timestamp(config), batch_size=1, bands=["RGB", "NIR", "SWIR"], tiles=batches["train"], is_train=True)

    # Run Transfer
    # data.run_waterbody_transfer(config)

    # Add Transplanted Tiles To Train Data
    transplanted_tiles = list(filter(lambda x: ".tif" in x, os.listdir(f"data/{get_timestamp_directory(config)}/transplanted_tiles/mask")))
    transplanted_tile_indices = list(map(lambda x: int(x.split(".")[1]), transplanted_tiles))
    batches["train"] += transplanted_tile_indices
    random.shuffle(batches["train"])

    # Save Batches To Disk
    if "transplanted.json" not in os.listdir("batches"):
        with open("batches/transplanted.json", 'w') as batch_file:
            batch_file.write(json.dumps(batches, indent=2))


if __name__ == "__main__":
    # Set Visible GPU
    args = sys.argv
    GPUS = args[1:] if len(args) > 1 else ["0"] 
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{','.join(GPUS)}"

    # Run Script
    main()

