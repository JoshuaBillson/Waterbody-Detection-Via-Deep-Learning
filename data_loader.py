import os
import math
import shutil
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Sequence, List
from cv2 import normalize, NORM_MINMAX, CV_8U, LUT
from tensorflow.keras.utils import Sequence as KerasSequence


class DataLoader:
    """A class to generate patches, load the dataset from disk, and show statistics."""
    def __init__(self, tile_size: int = 1024, timestamp: int = 1):
        self.timestamp = timestamp
        self.tile_size = tile_size
        self.folders = {1: "2018.04", 2: "2018.12", 3: "2019.02"}

    def get_rgb_features(self, patch_number: int) -> np.ndarray:
        """
        Get RGB Features Matching The Given Patch Number
        :param patch_number: The number of the patch we want to retrieve which must be in the range [min_patch, max_patch]
        :return: The RGB features of the matching patch,
        """
        return self.read_image(f"data/{self.folders.get(self.timestamp, 1)}/patches/rgb/rgb.{patch_number}.tif")

    def get_nir_features(self, patch_number: int) -> np.ndarray:
        """
        Get NIR Features Matching The Given Patch Number
        :param patch_number: The number of the patch we want to retrieve which must be in the range [min_patch, max_patch]
        :return: The NIR features of the matching patch,
        """
        return self.read_image(f"data/{self.folders.get(self.timestamp, 1)}/patches/nir/nir.{patch_number}.tif")

    def get_swir_features(self, patch_number: int) -> np.ndarray:
        """
        Get SWIR Features Matching The Given Patch Number
        :param patch_number: The number of the patch we want to retrieve which must be in the range [min_patch, max_patch]
        :return: The SWIR features of the matching patch,
        """
        return self.read_image(f"data/{self.folders.get(self.timestamp, 1)}/patches/swir/swir.{patch_number}.tif")

    def get_mask(self, patch_number: int) -> np.ndarray:
        """
        Get The Mask For The Given Patch Number
        :param patch_number: The number of the patch we want to retrieve which must be in the range [min_patch, max_patch]
        :return: The mark of the matching patch.
        """
        return self.read_image(f"data/{self.folders.get(self.timestamp, 1)}/patches/mask/mask.{patch_number}.tif")

    def get_bounds(self) -> Tuple[int, int]:
        """Returns the indices of the lowest and highest numbered patches respectively."""
        files = os.listdir(f"data/{self.folders.get(self.timestamp, 1)}/patches/mask")
        min_patch = min([int(x.split(".")[1]) for x in files])
        max_patch = max([int(x.split(".")[1]) for x in files])
        return min_patch, max_patch

    def create_rgb_and_nir_patches(self, show_image: bool = False) -> None:
        """
        Creates RGB and NIR patches from the original image and saves them to disk
        :param show_imgage: If this parameter is set to True, we plot the original RGB and NIR image for visualization purposes
        """
        # Read Image
        img = self.read_image(f"data/{self.folders.get(self.timestamp, 1)}/rgb_nir/rgb_nir.tif")

        # Show RGB Image
        if show_image:
            rgb_img = img[..., 0:3]
            img_scaled = normalize(rgb_img, None, 0, 255, NORM_MINMAX, dtype=CV_8U)
            img_scaled = DataLoader.adjust_gamma(img_scaled, 0.2)
            plt.imshow(img_scaled)
            plt.savefig(f"images/rgb.{self.timestamp}.png", dpi=5000, bbox_inches='tight')
            plt.show()

            plt.imshow(self._threshold_channel(img[..., 3:]))
            plt.savefig(f"images/nir.{self.timestamp}.png", dpi=5000, bbox_inches='tight')
            plt.show()

        # Partition Image Into Patches
        patches = self._segment_image(img)
        rgb_patches = patches[..., 0:3]
        nir_patches = patches[..., 3:]
        self._write_patches(nir_patches, "nir")
        self._write_patches(rgb_patches, "rgb")

    def create_swir_patches(self, show_img: bool = False) -> None:
        """
        Creates SWIR patches from the original image and saves them to disk
        :param show_img: If this parameter is set to True, we plot the original SWIR image for visualization purposes
        """
        # Open File
        img = self.read_image(f"data/{self.folders.get(self.timestamp, 1)}/swir/swir.tif")

        # Plot Image
        if show_img:
            plt.imshow(self._threshold_channel(img))
            plt.savefig(f"images/swir.{self.timestamp}.png", dpi=5000, bbox_inches='tight')
            plt.show()

        # Partition Image Into Patches
        patches = self._segment_image(img, is_swir=True).astype("uint16")
        self._write_patches(patches, "swir")

    def create_mask_patches(self, show_mask: bool = False) -> None:
        """
        Creates mask patches from the original mask and saves them to disk
        :param show_mask: If this parameter is set to True, we plot the mask for visualization purposes
        """
        # Open File
        mask = self.read_image("data/label.tif")

        # Plot Image
        if show_mask:
            plt.imshow(mask)
            plt.savefig(f"images/mask.{self.timestamp}.png", dpi=5000, bbox_inches='tight')

        # Return Patches
        patches = self._segment_image(np.clip(mask, 0, 1))
        self._write_patches(patches, "mask")

    @staticmethod
    def read_image(filename: str) -> np.ndarray:
        """
        Reads a raster image from disk and returns it
        :param filename: The name of the file on disk that we want to read
        :returns: The read image as a Numpy array
        """
        with rasterio.open(filename) as dataset:
            # Read Channels
            num_channels = dataset.count
            channels = [dataset.read(i) for i in range(1, num_channels+1)]

            # Reshape Channels
            shape = (channels[0].shape[0], channels[0].shape[1], 1)
            channels = [np.reshape(channel, shape) for channel in channels]

            # Concat Channels If More Than One
            return np.concatenate(channels, axis=2) if len(channels) > 1 else channels[0]

    def plot_samples(self, rgb_samples: Sequence[np.ndarray], nir_samples: Sequence[np.ndarray], swir_samples: Sequence[np.ndarray], mask_samples: Sequence[np.ndarray]) -> None:
        """
        Takes some sample patches and plots them
        :param rgb_samples: The samples of RGB patches we want to plot
        :param nir_samples: The samples of NIR patches we want to plot
        :param swir_samples: The samples of SWIR patches we want to plot
        :param mask_samples: The samples of mask patches we want to plot
        """
        # Create Subplots
        num_samples = min(len(rgb_samples), len(nir_samples), len(swir_samples), len(mask_samples))
        fig, axs = plt.subplots(num_samples, 4)
        fig.tight_layout()

        # Label Columns
        for ax, col in zip(axs[0], ["Mask", "RGB", "Nir", "Swir"]):
            ax.set_title(col)

        # Plot Samples
        for rgb_sample, nir_sample, swir_sample, mask_sample, ax in zip(rgb_samples, nir_samples, swir_samples, mask_samples, axs):
            ax[0].imshow(mask_sample)
            ax[1].imshow(DataLoader.adjust_gamma(normalize(rgb_sample, None, 0, 255, NORM_MINMAX, dtype=CV_8U), 0.4))
            ax[2].imshow(DataLoader._threshold_channel(nir_sample))
            ax[3].imshow(DataLoader._threshold_channel(swir_sample))

        # Save Figure
        plt.savefig(f"images/samples.{self.timestamp}.png", dpi=2500, bbox_inches='tight')
        plt.show()

    @staticmethod
    def adjust_gamma(image: np.ndarray, gamma: float = 1.0):
        """
        Perform gamma correction on the provided image
        :param image: The image to which we want to apply gamme correction
        :param gamma: The ammount of gamma correction to apply
        :returns: The gamma corrected image
        """
        invGamma = 1 / gamma
        table = np.array([((i / 255.0) * invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return LUT(image, table)
    
    @staticmethod
    def save_image(img: np.ndarray, filename: str) -> None:
        """
        Save a raster image to disk
        :param img: The image we want to save encoded as a numpy array
        :param filename: The name of the destination file at which we want to save the image
        :returns: Nothing
        """
        height, width, count, dtype = img.shape[0], img.shape[1], img.shape[2], img.dtype
        with rasterio.open(filename, 'w', driver='GTiff', height=height, width=width, count=count, dtype=dtype) as dst:
            dst.write(np.moveaxis(img, -1, 0))

    @staticmethod
    def _threshold_channel(channel, threshold=3000):
        return np.clip(channel, 0, threshold)

    def _write_patches(self, patches, image_type):
        # Create Patches Directory
        timestamp_directory = f"data/{self.folders.get(self.timestamp, 1)}"
        if "patches" not in os.listdir(timestamp_directory):
            os.mkdir(f"{timestamp_directory}/patches")

        # Create Directory For Patches Of The Appropriate Type (RGB, NIR, SWIR, Mask)
        patches_directory = f"{timestamp_directory}/patches"
        img_directory = f"{patches_directory}/{image_type}"
        if image_type in os.listdir(patches_directory):
            shutil.rmtree(img_directory)
        os.mkdir(img_directory)

        # Save Patches To Disk
        for patch_number, patch in enumerate(patches):
            filename = f"{img_directory}/{image_type}.{patch_number + 1}.tif"
            self.save_image(patch, filename)

    def _segment_image(self, img, is_swir=False):
        # Compute Dimentions
        tile_size = self.tile_size if not is_swir else self.tile_size // 2
        img_size = min(img.shape[0], img.shape[1])
        num_tiles = img_size // tile_size

        # Construct Patches
        patches = []
        for x in np.arange(start=0, stop=num_tiles*tile_size, step=tile_size):
            for y in np.arange(start=0, stop=num_tiles*tile_size, step=tile_size):
                tile = img[y:y+tile_size, x:x+tile_size, :]
                patches += self._create_patches(tile)

        # Return Patches
        return np.array(patches)

    @staticmethod
    def _create_patches(tile: np.ndarray):
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


class ImgSequence(KerasSequence):
    def __init__(self, data_loader: DataLoader, lower_bound: int, upper_bound: int, batch_size: int = 32, bands: Sequence[str] = None):
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.bands = ["RGB"] if bands is None else bands
        self.indices = np.array(range(lower_bound, upper_bound + 1))
        np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, idx):
        # Create Batch
        rgb_batch, nir_batch, swir_batch, mask_batch = [], [], [], []
        batch = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        for b in batch:

            # Get Mask
            mask_batch.append(self.data_loader.get_mask(b))

            # Get RGB Features
            if "RGB" in self.bands:
                rgb_batch.append(self.data_loader.get_rgb_features(b))

            # Get NIR Features
            if "NIR" in self.bands:
                nir_batch.append(self.data_loader.get_nir_features(b))

            # Get SWIR Features
            if "SWIR" in self.bands:
                swir_batch.append(self.data_loader.get_swir_features(b))

        # Return Batch
        if all([band in self.bands for band in ["RGB", "NIR", "SWIR"]]):
            return [np.array(x).astype("float32") for x in [rgb_batch, nir_batch, swir_batch]], np.array(mask_batch).astype("float32")
        elif all([band in self.bands for band in ["RGB", "NIR"]]):
            return [np.array(x).astype("float32") for x in [rgb_batch, nir_batch]], np.array(mask_batch).astype("float32")
        elif all([band in self.bands for band in ["RGB", "SWIR"]]):
            return [np.array(x).astype("float32") for x in [rgb_batch, swir_batch]], np.array(mask_batch).astype("float32")
        elif "RGB" in self.bands:
            return np.array(rgb_batch).astype("float32"), np.array(mask_batch).astype("float32")
        elif "NIR" in self.bands:
            return np.array(nir_batch).astype("float32"), np.array(mask_batch).astype("float32")
        return np.array(swir_batch).astype("float32"), np.array(mask_batch).astype("float32")
    
    def get_patch_indices(self) -> List[int]:
        """
        Get the patch indices for all patches in this dataset
        :returns: The list of all patch indices in this dataset.
        """
        return list(self.indices)


def create_patches(loader: DataLoader, show_image: bool = False) -> None:
    """
    Generate the patches and save them to disk.
    :param loader: The DataLoader that will be used to read the patches from disk
    :param show_image: If True, we will visualize the original images from which the patches are generated
    :returns: Nothing
    """
    loader.create_rgb_and_nir_patches(show_image=show_image)
    loader.create_swir_patches(show_img=show_image)
    loader.create_mask_patches(show_mask=show_image)


def load_dataset(loader: DataLoader, config) -> Tuple[ImgSequence, ImgSequence, ImgSequence]:
    """
    Load the training, validation, and test datasets
    :param loader: The DataLoader that will be used to read the patches from disk
    :param config: The script configuration encoded as a Python dictionary; typically read from an external file
    :returns: The training, validation, and test data as a tuple of the form (train, validation, test)
    """
    bands = config["hyperparameters"]["bands"]
    batch_size = config["hyperparameters"]["batch_size"]
    lower_bound, upper_bound = loader.get_bounds()
    assert lower_bound == 1 and upper_bound == 3600, f"Error: Bounds Must Be Between 1 and 3600 (Got [{lower_bound}, {upper_bound}])"
    train_data = ImgSequence(loader, 1, 2700, bands=bands, batch_size=batch_size)
    val_data = ImgSequence(loader, 2701, 3000, bands=bands, batch_size=batch_size)
    test_data = ImgSequence(loader, 3001, 3600, bands=bands, batch_size=batch_size)
    return train_data, val_data, test_data


def show_samples(loader: DataLoader) -> None:
    """Visualize A Selection Of Patches"""
    rgb_samples, nir_samples, swir_samples, mask_samples = [], [], [], []
    for patch in range(1003, 1006):
        rgb, nir, swir = loader.get_rgb_features(patch), loader.get_nir_features(patch), loader.get_swir_features(patch)
        mask = loader.get_mask(patch)
        rgb_samples.append(rgb)
        nir_samples.append(nir)
        swir_samples.append(swir)
        mask_samples.append(mask)
    loader.plot_samples(rgb_samples, nir_samples, swir_samples, mask_samples)


def analyze_dataset(loader: DataLoader) -> None:
    """
    Perform statistical analysis of the dataset
    :param loader: The DataLoader that will be used to read the patches from disk
    :returns: Nothing
    """
    # Analyze Initial Mask
    mask = np.clip(loader.read_image("data/label.tif"), a_min=0, a_max=1)
    total_pixels = mask.size
    water_pixels = np.sum(mask)
    print("\nMASK\n")
    print(f"Total Pixels: {total_pixels}\nWater Pixels: {water_pixels}\nWater Percentage: {round(water_pixels / total_pixels * 100.0, ndigits=2)}%")

    # Analyze Patches
    water_pixels_hist = []
    lower_bound, upper_bound = loader.get_bounds()
    for patch in range(lower_bound, upper_bound+1):
        mask_patch = loader.get_mask(patch)
        total_pixels = mask_patch.size
        water_pixels = np.sum(mask_patch)
        water_percentage = water_pixels / total_pixels * 100.0
        water_pixels_hist.append(water_percentage)
    
    # Generate Histogram For All Patches
    stats = plt.hist(water_pixels_hist, bins=np.arange(0.0, 51.0, 2.5))
    plt.title("All Patches")
    plt.xlabel("Water Pixels (%)")
    plt.ylabel("Patches (Count)")
    plt.savefig("images/histogram_all.png", bbox_inches='tight')
    plt.cla()
    
    # Summarize Histogram Statistics
    print("\nSUMMARIZE PATCH HISTOGRAM\n")
    for count, b in zip(stats[0], stats[1]):
        print(f"[{b}, {b+2.5}): {int(count)}")
    

    # Generate Histogram For Patches With At Least 5% Water
    stats = plt.hist(list(filter(lambda x: x >= 5.0, water_pixels_hist)), bins=np.arange(5.0, 51.0, 2.5))
    plt.title("Patches With At Least 5% Water")
    plt.xlabel("Water Pixels (%)")
    plt.ylabel("Patches (Count)")
    plt.savefig("images/histogram_over5.png", bbox_inches='tight')
    plt.cla()

    # Additional Statistics
    print("\nADDITIONAL STATISTICS\n")
    print(f"Patches With No Water: {len(list(filter(lambda x: x == 0.0, water_pixels_hist)))}")
    print(f"Patches With Water: {len(list(filter(lambda x: x > 0.0, water_pixels_hist)))}")
    print(f"Patches With Less Than 5% Water: {len(list(filter(lambda x: x < 5.0, water_pixels_hist)))}")
    print(f"Patches With Over 5% Water: {len(list(filter(lambda x: x >= 5.0, water_pixels_hist)))}")
    print(f"Patches With Less Than 10% Water: {len(list(filter(lambda x: x < 10.0, water_pixels_hist)))}")
    print(f"Patches With Over 10% Water: {len(list(filter(lambda x: x >= 10.0, water_pixels_hist)))}")