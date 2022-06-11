import os
import sys
import json
import shutil
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from backend.config import get_patch_size, get_timestamp_directory, get_timestamp
from backend.data_loader import DataLoader
from backend.utils import adjust_rgb


def create_rgb_and_nir_patches(config: Dict[str, Any], show_img: bool = False) -> None:
    """
    Creates RGB and NIR patches from the original image and saves them to disk
    :param config: The script configuration stored as a dictionary; typically read from an external file
    :param show_img: If this parameter is set to True, we plot the original RGB and NIR image for visualization purposes
    """
    # Read Image
    directory = get_timestamp_directory(config)
    img = DataLoader.read_image(f"data/{directory}/rgb_nir/rgb_nir.tif")

    # Show RGB Image
    if show_img:
        rgb_img = img[..., 0:3]
        img_scaled = adjust_rgb(rgb_img, gamma=0.2)
        plt.imshow(img_scaled)
        plt.savefig(f"images/rgb.{get_timestamp(config)}.png", dpi=5000, bbox_inches='tight')
        plt.show()

        plt.imshow(DataLoader.threshold_channel(img[..., 3:]))
        plt.savefig(f"images/nir.{get_timestamp(config)}.png", dpi=5000, bbox_inches='tight')
        plt.show()

    # Partition Image Into Patches
    patches = segment_image(img, config)
    rgb_patches = patches[..., 0:3]
    nir_patches = patches[..., 3:]
    write_patches(nir_patches, "nir", config)
    write_patches(rgb_patches, "rgb", config)


def create_swir_patches(config: Dict[str, Any], show_img: bool = False) -> None:
    """
    Creates SWIR patches from the original image and saves them to disk
    :param config: The script configuration stored as a dictionary; typically read from an external file
    :param show_img: If this parameter is set to True, we plot the original SWIR image for visualization purposes
    """
    # Open File
    img = DataLoader.read_image(f"data/{get_timestamp_directory(config)}/swir/swir.tif")

    # Plot Image
    if show_img:
        plt.imshow(DataLoader.threshold_channel(img))
        plt.savefig(f"images/swir.{get_timestamp(config)}.png", dpi=5000, bbox_inches='tight')
        plt.show()

    # Partition Image Into Patches
    patches = segment_image(img, config, is_swir=True).astype("uint16")
    write_patches(patches, "swir", config)


def create_mask_patches(config: Dict[str, Any], show_mask: bool = False) -> None:
    """
    Creates mask patches from the original mask and saves them to disk
    :param config: The script configuration stored as a dictionary; typically read from an external file
    :param show_mask: If this parameter is set to True, we plot the mask for visualization purposes
    """
    # Open File
    mask = DataLoader.read_image("data/label.tif")

    # Plot Image
    if show_mask:
        plt.imshow(mask)
        plt.savefig(f"images/mask.{get_timestamp(config)}.png", dpi=5000, bbox_inches='tight')

    # Return Patches
    patches = segment_image(np.clip(mask, 0, 1), config)
    write_patches(patches, "mask", config)
    

def write_patches(patches: List[np.ndarray], image_type: str, config: Dict[str, Any]) -> None:
    """
    Write a list of patches to disk
    :param patches: The patches to be written to disk
    :param image_type: The type of patches; one of 'rgb', 'nir', 'swir', or 'mask'
    :param config: The script configuration stored as a dictionary; typically read from an external file
    :returns: Nothing
    """
    # Create Patches Directory
    timestamp_directory = f"data/{get_timestamp_directory(config)}"
    if "patches" not in os.listdir(timestamp_directory):
        os.mkdir(f"{timestamp_directory}/patches")

    # Create Directory For Patches Of The Appropriate Type (RGB, NIR, SWIR, Mask)hhh
    patches_directory = f"{timestamp_directory}/patches"
    img_directory = f"{patches_directory}/{image_type}"
    if image_type in os.listdir(patches_directory):
        shutil.rmtree(img_directory)
    os.mkdir(img_directory)

    # Save Patches To Disk
    for patch_number, patch in enumerate(patches):
        filename = f"{img_directory}/{image_type}.{patch_number + 1}.tif"
        DataLoader.save_image(patch, filename)


def segment_image(img: np.ndarray, config: Dict[str, Any], is_swir: bool = False):
    """
    Segment an image from the original dataset into patches
    :param img: The image we want to cut into patches
    :param config: The script configuration stored as a dictionary; typically read from an external file
    :param is_swir: If true, we need to generate patches half the size specified in the config file
    :returns: A list of patches generated from the given image
    """
    # Compute Dimentions
    tile_size = (get_patch_size(config) * 2) if not is_swir else get_patch_size(config)
    img_size = min(img.shape[0], img.shape[1])
    num_tiles = img_size // tile_size

    # Construct Patches
    patches = []
    for x in np.arange(start=0, stop=num_tiles*tile_size, step=tile_size):
        for y in np.arange(start=0, stop=num_tiles*tile_size, step=tile_size):
            tile = img[y:y+tile_size, x:x+tile_size, :]
            patches += create_patches(tile)

    # Return Patches
    return np.array(patches)


def create_patches(tile: np.ndarray) -> List[np.array]:
    """
    Segment a tile into 9 partially overlapping patches.
    :param tile: The tile we want to cut into patches.
    :returns: A list of patches generated from the given tile.
    """
    patches = []
    patch_size = tile.shape[0] // 2
    for x in (0, 0.5, 1):
        for y in (0, 0.5, 1):
            top = int(y*patch_size)
            bottom = top + patch_size
            left = int(x*patch_size)
            right = left + patch_size
            patches.append(tile[top:bottom, left:right, :])
    return patches


def _create_patches(config: Dict[str, Any], show_image: bool = False) -> None:
    """
    Generate the patches and save them to disk.
    :param config: The script configuration stored as a dictionary; typically read from an external file
    :param show_image: If True, we will visualize the original images from which the patches are generated
    :returns: Nothing
    """
    create_swir_patches(config, show_img=show_image)
    create_rgb_and_nir_patches(config, show_img=show_image)
    create_mask_patches(config, show_mask=show_image)


def create_batches(config: Dict[str, Any]) -> None:
    """
    Divide patches into training, validation, and test batches. Save the patch indices for each batch to disk to
    ensure that we use the same patches for training, validation, and testing across different experiments.
    :param config: The script configuration stored as a dictionary; typically read from an external file
    :returns: Nothing
    """
    # Get Patch Indices
    patches_directory = f"data/{get_timestamp_directory(config)}/patches"
    patches = list(filter(lambda x: "mask" in x, os.listdir(f"{patches_directory}/mask")))
    num_patches = len(patches)
    patch_indices = np.array(range(1, num_patches + 1))
    np.random.shuffle(patch_indices)

    # Split Patches Into Training, Validation, And Test Batches
    num_train, num_val = (2700, 300) if num_patches == 3600 else (int(0.75 * num_patches), int(0.10 * num_patches))
    train_data = patches[0:num_train]
    val_data = patches[num_train:num_train+num_val]
    test_data = patches[num_train+num_val:]

    # Create Directory For Batches
    batches_directory = "batches"
    if batches_directory not in os.listdir():
        os.mkdir(batches_directory)

    # Write Batches To Disk
    with open(f"batches/{get_patch_size(config)}.json", 'w') as batch_file:
        batch_file.write(json.dumps({"train": list(train_data), "validation": list(val_data), "test": list(test_data)}))


def generate_patches(loader: DataLoader = None, config: Dict[str, Any] = None):
    # Get Project Configuration
    if config is None:
        with open('config.json') as f:
            config = json.loads(f.read())

    # Create Data Loader
    if loader is None:
        loader = DataLoader(timestamp=get_timestamp(config))

    # Generate Patches
    _create_patches(config, True)
    create_batches(config)


if __name__ == "__main__":
    # Set Visible GPU
    args = sys.argv
    GPUS = args[1:] if len(args) > 1 else ["0"] 
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{','.join(GPUS)}"

    # Generate Patches
    generate_patches()
