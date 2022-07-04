import os
import random
import sys
import json
from math import sqrt
import shutil
from typing import List, Dict, Any, Sequence
import matplotlib.pyplot as plt
import numpy as np
from backend.data_loader import DataLoader
from backend.utils import adjust_rgb


def show_patches(loader: DataLoader, x_bounds, y_bounds, directory) -> None:
    """Visualize A Selection Of Patches"""

    # Create Directory Plotting Images
    if directory in os.listdir(f"images/patches"):
        shutil.rmtree(f"images/patches/{directory}")
    os.mkdir(f"images/patches/{directory}")

    # Generate  Patches
    tile = loader.get_features(268, ["mask", "RGB", "NIR", "SWIR"], subsample=False)
    patches = generate_patches(tile, x_bounds, y_bounds)

    # Plot Tile
    _, axs = plt.subplots(1, 4)
    for col, band in enumerate(["mask", "RGB", "NIR", "SWIR"]):
        axs[col].imshow(adjust_rgb(tile[band], gamma=0.8) if band == "RGB" else tile[band])
        axs[col].axis("off")
    plt.savefig(f"images/patches/{directory}/tile.png", dpi=2500, bbox_inches="tight")
    plt.close()

    # Plot Non-Overlapping Patches
    for band in tile.keys():
        i = 0
        size = int(sqrt(len(x_bounds)))
        _, ax = plt.subplots(size, size, figsize=(size, size))
        for row in range(size):
            for col in range(size):
                ax[row][col].imshow(adjust_rgb(patches[i][band], gamma=0.8) if band == "RGB" else patches[i][band])
                ax[row][col].axis("off")
                i += 1
        plt.savefig(f"images/patches/{directory}/{band}.png", dpi=2500, bbox_inches="tight")
        plt.close()


def generate_patches(tile: Dict[str, np.ndarray], x_bounds, y_bounds) -> List[Dict[str, np.ndarray]]:
    """
    Takes a tile and generates four non-overlapping patches
    :param tile: The tile to be cut into patches
    :returns: A list of non-overlapping patches
    """
    patches = []
    for x, y in zip(x_bounds, y_bounds):
        patch = dict()
        for band in tile.keys():
            if band == "SWIR":
                x_swir = x // 2
                y_swir = y // 2
                patch[band] = tile[band][y_swir:y_swir+256, x_swir:x_swir+256, :]
            else:
                patch[band] = tile[band][y:y+512, x:x+512, :]
        patches.append (patch)
    return patches


def main():
    # Create Directory Plotting Images
    if "patches" in os.listdir("images"):
        shutil.rmtree("images/patches")
    os.mkdir("images/patches")
    
    # Create Data Loader
    loader = DataLoader(timestamp=1, overlapping_patches=True, upscale_swir=False)

    # Show Non-Overlapping Patches
    show_patches(loader, (0, 512, 0, 512), (0, 0, 512, 512), "non_overlapping")

    # Show Overlapping Patches
    show_patches(loader, (0, 256, 512, 0, 256, 512, 0, 256, 512), (0, 0, 0, 256, 256, 256, 512, 512, 512), "overlapping")

    # Show Random Patches
    show_patches(loader, [random.randint(0, 512) for _ in range(9)], [random.randint(0, 512) for _ in range(9)], "random")


if __name__ == "__main__":
    # Set Visible GPU
    args = sys.argv
    GPUS = args[1:] if len(args) > 1 else ["0"] 
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{','.join(GPUS)}"

    # Generate Patches
    main()
