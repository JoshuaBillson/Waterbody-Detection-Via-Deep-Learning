import os
import sys
import json
import shutil
from typing import List, Dict, Any, Sequence
import matplotlib.pyplot as plt
import numpy as np
from backend.config import get_patch_size, get_timestamp_directory, get_timestamp
from backend.data_loader import DataLoader
from backend.utils import adjust_rgb


def create_rgb_and_nir_tiles(config: Dict[str, Any]) -> None:
    """
    Creates RGB and NIR patches from the original image and saves them to disk
    :param config: The script configuration stored as a dictionary; typically read from an external file
    :param show_img: If this parameter is set to True, we plot the original RGB and NIR image for visualization purposes
    """
    # Read Image
    print("READING RGB")
    directory = get_timestamp_directory(config)
    img = DataLoader.read_image(f"data/{directory}/rgb_nir/rgb_nir.tif", preprocess_img=False)
    print("DONE READING RGB")

    # Plot RGB And NIR Features
    rgb_img = img[..., 0:3]
    img_scaled = adjust_rgb(rgb_img, gamma=0.2)
    plt.imshow(img_scaled)
    plt.savefig(f"images/features/{directory}/rgb_features.png", dpi=2500, bbox_inches='tight')
    plt.close()

    plt.imshow(np.clip(img[..., 3:], a_min=0, a_max=3000))
    plt.savefig(f"images/features/{directory}/nir_features.png", dpi=2500, bbox_inches='tight')
    plt.close()

    # Partition Image Into Tiles
    tiles = image_to_tiles(img, config)
    rgb_tiles = tiles[..., 0:3]
    nir_tiles = tiles[..., 3:]
    write_tiles(nir_tiles, "nir", config)
    write_tiles(rgb_tiles, "rgb", config)


def create_swir_tiles(config: Dict[str, Any]) -> None:
    """
    Creates SWIR patches from the original image and saves them to disk
    :param config: The script configuration stored as a dictionary; typically read from an external file
    :param show_img: If this parameter is set to True, we plot the original SWIR image for visualization purposes
    """
    # Open File
    print("READING SWIR")
    img = DataLoader.read_image(f"data/{get_timestamp_directory(config)}/swir/swir.tif", preprocess_img=False)
    print("DONE READING SWIR")

    # Plot SWIR Features
    plt.imshow(np.clip(img, a_min=0, a_max=3000))
    plt.savefig(f"images/features/{get_timestamp_directory(config)}/swir_features.png", dpi=2500, bbox_inches='tight')
    plt.close()

    # Partition Image Into Patches
    patches = image_to_tiles(img, config, is_swir=True).astype("uint16")
    write_tiles(patches, "swir", config)


def create_mask_tiles(config: Dict[str, Any], show_mask: bool = False) -> None:
    """
    Creates mask patches from the original mask and saves them to disk
    :param config: The script configuration stored as a dictionary; typically read from an external file
    :param show_mask: If this parameter is set to True, we plot the mask for visualization purposes
    """
    # Open File
    print("READING MASK")
    mask = DataLoader.read_image("data/label.tif", preprocess_img=False)
    print("DONE READING MASK")

    # Plot Image
    plt.imshow(mask)
    plt.savefig(f"images/features/{get_timestamp_directory(config)}/mask.png", dpi=2500, bbox_inches='tight')
    plt.close()

    # Return Patches
    patches = image_to_tiles(np.clip(mask, 0, 1), config)
    write_tiles(patches, "mask", config)
    

def write_tiles(tiles: List[np.ndarray], image_type: str, config: Dict[str, Any], subdir: str = "tiles") -> None:
    """
    Write a list of tiles to disk
    :param patches: The tiles to be written to disk
    :param image_type: The type of tiles; one of 'rgb', 'nir', 'swir', or 'mask'
    :param config: The script configuration stored as a dictionary; typically read from an external file
    :returns: Nothing
    """
    # Create Tiles Directory
    create_directory(f"data/{get_timestamp_directory(config)}/{subdir}/{image_type}", delete_old=True)

    # Save Patches To Disk
    for tile_number, tile in enumerate(tiles):
        img_directory = f"data/{get_timestamp_directory(config)}/{subdir}/{image_type}"
        filename = f"{img_directory}/{image_type}.{tile_number + 1}.tif"
        DataLoader.save_image(tile, filename)


def create_tiles(config: Dict[str, Any]) -> None:
    """
    Generate the tiles and save them to disk.
    :param config: The script configuration stored as a dictionary; typically read from an external file
    :param show_image: If True, we will visualize the original images from which the patches are generated
    :returns: Nothing
    """
    create_mask_tiles(config)
    create_swir_tiles(config)
    create_rgb_and_nir_tiles(config)


def create_patches(config: Dict[str, Any]) -> None:
    """
    Generate the tiles and save them to disk.
    :param config: The script configuration stored as a dictionary; typically read from an external file
    :param show_image: If True, we will visualize the original images from which the patches are generated
    :returns: Nothing
    """
    for image_type in ("mask" , "rgb", "nir", "swir"):
        # Load Tile From Disk
        tiles_directory = f"data/{get_timestamp_directory(config)}/tiles/{image_type}"
        for tile in range(1, 401):
            tile = DataLoader.read_image(f"{tiles_directory}/{image_type}.{tile}.tif")


def image_to_tiles(img: np.ndarray, config: Dict[str, Any], is_swir: bool = False):
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
    tiles = []
    for x in np.arange(start=0, stop=num_tiles*tile_size, step=tile_size):
        for y in np.arange(start=0, stop=num_tiles*tile_size, step=tile_size):
            tiles.append(img[y:y+tile_size, x:x+tile_size, :])

    # Return Patches
    return np.array(tiles)


def patches_from_tile(tile: np.ndarray) -> List[np.array]:
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


def create_batches(config: Dict[str, Any]) -> None:
    """
    Divide patches into training, validation, and test batches. Save the patch indices for each batch to disk to
    ensure that we use the same patches for training, validation, and testing across different experiments.
    :param config: The script configuration stored as a dictionary; typically read from an external file
    :returns: Nothing
    """
    # Create Directory For Batches
    batches_directory = "batches"
    if batches_directory not in os.listdir():
        os.mkdir(batches_directory)

    filename = "tiles.json"
    if filename not in os.listdir("batches"):

        # Get Patch Indices
        patch_indices = np.array(range(1, 401))
        np.random.shuffle(patch_indices)

        # Split Patches Into Training, Validation, And Test Batches
        train_data = list(map(int, list(patch_indices[0:300])))
        val_data = list(map(int, list(patch_indices[300:335])))
        test_data = list(map(int, list(patch_indices[335:])))

        # Write Batches To Disk
        with open(f"batches/tiles.json", 'w') as batch_file:
            batch_file.write(json.dumps({"train": train_data, "validation": val_data, "test": test_data}, indent=2))


def create_directory(path: str, delete_old: bool = False) -> None:
    # Create Directory
    dirs, current_path = path.split("/"), [] 
    for directory in dirs:
        contents = os.listdir() if len(current_path) == 0 else os.listdir("/".join(current_path))
        if directory not in contents:
            os.mkdir("/".join(current_path + [directory]))
        current_path.append(directory)

    # Delete Old Files
    if delete_old:
        for file in os.listdir(path):
            os.remove(f"{path}/{file}")


def show_samples(loader: DataLoader, config) -> None:
    """Visualize A Selection Of Patches"""
    rgb_samples, nir_samples, swir_samples, mask_samples = [], [], [], []
    for patch in [26809, 38902, 16801]:
        features = loader.get_features(patch, ["mask", "RGB", "NIR", "SWIR"])
        for band in features.keys():
            if band == "mask":
                mask_samples.append(features[band])
            elif band == "RGB":
                rgb_samples.append(features[band])
            elif band == "NIR":
                nir_samples.append(features[band])
            elif band == "SWIR":
                swir_samples.append(features[band])
    plot_samples(rgb_samples, nir_samples, swir_samples, mask_samples, config)


def plot_samples(rgb_samples: Sequence[np.ndarray], nir_samples: Sequence[np.ndarray], swir_samples: Sequence[np.ndarray], mask_samples: Sequence[np.ndarray], config) -> None:
    """
    Takes some sample patches and plots them
    :param rgb_samples: The samples of RGB patches we want to plot
    :param nir_samples: The samples of NIR patches we want to plot
    :param swir_samples: The samples of SWIR patches we want to plot
    :param mask_samples: The samples of mask patches we want to plot
    """
    # Create Sample Directory
    create_directory(f"images/samples/{get_timestamp_directory(config)}", delete_old=True)

    # Create Subplots
    num_samples = min(len(rgb_samples), len(nir_samples), len(swir_samples), len(mask_samples))
    _, axs = plt.subplots(num_samples, 4, figsize=(4, num_samples))

    # Label Columns
    for ax, col in zip(axs[0], ["Mask", "RGB", "Nir", "Swir"]):
        ax.set_title(col)

    # Plot Samples
    for rgb_sample, nir_sample, swir_sample, mask_sample, ax in zip(rgb_samples, nir_samples, swir_samples, mask_samples, axs):
        ax[0].imshow(mask_sample)
        ax[0].axis("off")
        ax[1].imshow(adjust_rgb(rgb_sample, gamma=0.8))
        ax[1].axis("off")
        ax[2].imshow(np.clip(nir_sample, a_min=0, a_max=3000))
        ax[2].axis("off")
        ax[3].imshow(np.clip(swir_sample, a_min=0, a_max=3000))
        ax[3].axis("off")

    # Save Figure
    plt.savefig(f"images/samples/{get_timestamp_directory(config)}/sample.png", dpi=2500, bbox_inches="tight")
    plt.close()


def generate_patches(loader: DataLoader = None, config: Dict[str, Any] = None):
    # Get Project Configuration
    if config is None:
        with open('config.json') as f:
            config = json.loads(f.read())

    # Create Data Loader
    if loader is None:
        loader = DataLoader(timestamp=get_timestamp(config), overlapping_patches=True)

    # Create Directories To Plot Data
    create_directory(f"images/features/{get_timestamp_directory(config)}")

    # Generate Tiles
    create_tiles(config)

    # Generate Patches
    create_patches(config)

    # Create Batches
    create_batches(config)

    # Show Samples
    show_samples(loader, config)



if __name__ == "__main__":
    # Set Visible GPU
    args = sys.argv
    GPUS = args[1:] if len(args) > 1 else ["0"] 
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{','.join(GPUS)}"

    # Generate Patches
    generate_patches()
