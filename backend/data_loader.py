import random
from typing import Tuple, List, Dict
import cv2
import rasterio
import numpy as np


class DataLoader:
    """A class to save and load images from disk"""
    def __init__(self, timestamp: int = 1, overlapping_patches: bool = False, random_subsample: bool = False):
        self.timestamp = timestamp
        self.folders = {1: "2018.04", 2: "2018.12", 3: "2019.02"}
        self.overlapping_patches = overlapping_patches
        self.random_subsample = random_subsample

    def get_rgb_features(self, tile_number: int, coords: Tuple[int, int] = (0, 0), preprocess_img: bool = True) -> np.ndarray:
        """
        Get RGB Features Matching The Given Patch Number
        :param patch_number: The number of the patch we want to retrieve which must be in the range [min_patch, max_patch]
        :return: The RGB features of the matching patch,
        """
        tile = self.read_image(f"data/{self.folders.get(self.timestamp, 1)}/tiles/rgb/rgb.{tile_number}.tif", preprocess_img=preprocess_img)
        return self.subsample_tile(tile, coords=coords) if coords is not None else tile

    def get_nir_features(self, tile_number: int, coords: Tuple[int, int] = None, preprocess_img: bool = True) -> np.ndarray:
        """
        Get NIR Features Matching The Given Patch Number
        :param patch_number: The number of the patch we want to retrieve which must be in the range [min_patch, max_patch]
        :return: The NIR features of the matching patch,
        """
        tile = self.read_image(f"data/{self.folders.get(self.timestamp, 1)}/tiles/nir/nir.{tile_number}.tif", preprocess_img=preprocess_img)
        return self.subsample_tile(tile, coords=coords) if coords is not None else tile

    def get_swir_features(self, tile_number: int, coords: Tuple[int, int] = None, preprocess_img: bool = True) -> np.ndarray:
        """
        Get SWIR Features Matching The Given Patch Number
        :param patch_number: The number of the patch we want to retrieve which must be in the range [min_patch, max_patch]
        :return: The SWIR features of the matching patch,
        """
        tile = self.read_image(f"data/{self.folders.get(self.timestamp, 1)}/tiles/swir/swir.{tile_number}.tif", preprocess_img=preprocess_img)
        tile = np.resize(cv2.resize(tile, (1024, 1024), interpolation = cv2.INTER_AREA), (1024, 1024, 1))
        return self.subsample_tile(tile, coords=coords) if coords is not None else tile

    def get_mask(self, tile_number: int, coords: Tuple[int, int] = None, preprocess_img: bool = True) -> np.ndarray:
        """
        Get The Mask For The Given Patch Number
        :param patch_number: The number of the patch we want to retrieve which must be in the range [min_patch, max_patch]
        :return: The mark of the matching patch.
        """
        tile = self.read_image(f"data/{self.folders.get(self.timestamp, 1)}/tiles/mask/mask.{tile_number}.tif")
        return self.subsample_tile(tile, coords=coords) if coords is not None else tile

    def get_features(self, patch: int, bands: List[str], subsample: bool = True) -> Dict[str, np.ndarray]:
        # Get Coords Of Patch Inside Tile
        coords, tile_index = None, patch
        if subsample:
            tile_index = patch // 100
            patch_index = patch - (tile_index * 100)
            coords = self.get_patch_coords(patch_index)

        # Get Mask
        features = {"mask": self.get_mask(tile_index, coords=coords, preprocess_img=False)}

        # Get RGB Features
        if "RGB" in bands:
            features["RGB"] = self.get_rgb_features(tile_index, preprocess_img=False, coords=coords)

        # Get NIR Features
        if "NIR" in bands:
            features["NIR"] = self.get_nir_features(tile_index, preprocess_img=False, coords=coords)

        # Get SWIR Features
        if "SWIR" in bands:
            features["SWIR"] = self.get_swir_features(tile_index, preprocess_img=False, coords=coords)

        return features
    
    def subsample_tile(self, tile: np.ndarray, coords: Tuple[int, int] = (0, 0)) -> np.ndarray:
        """
        Take a 512X512 sub-patch from a 1024X12024 tile.
        """
        return tile[coords[1]:coords[1]+512, coords[0]:coords[0]+512, :]
    
    def get_patch_coords(self, patch_index: int = 0) -> Tuple[int, int]:
        """Get the coordinates for a patch inside a tile from a given patch_index. If random_subsample is True, the coords will be selected randomly."""
        assert (self.overlapping_patches and (1 <= patch_index <= 9)) or (not self.overlapping_patches and (1 <= patch_index <= 4)) or (self.random_subsample)
        overlapping_coords = {1: (0, 0), 2: (256, 0), 3: (512, 0), 4: (0, 256), 5: (256, 256), 6: (512, 256), 7: (0, 512), 8: (256, 512), 9: (512, 512)}
        non_overlapping_coords = {1: (0, 0), 2: (512, 0), 3: (0, 512), 4: (512, 512)}
        if self.random_subsample:
            return (random.randint(0, 512), random.randint(0, 512))
        elif self.overlapping_patches:
            return overlapping_coords[patch_index]
        return non_overlapping_coords[patch_index]

    @staticmethod
    def read_image(filename: str, preprocess_img: bool = False) -> np.ndarray:
        """
        Reads a raster image from disk and returns it
        :param filename: The name of the file on disk that we want to read
        :param preprocess_img: If True, the image is normalized before being returned to the caller
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
            img = np.concatenate(channels, axis=2) if len(channels) > 1 else channels[0]
            return DataLoader.normalize_channels(img.astype("float32")) if preprocess_img else img

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
    def normalize_channels(img: np.ndarray) -> np.ndarray:
        # First We Threshold SWIR and NIR Patches
        if img.shape[-1] == 1:
            img = np.clip(img, a_min=0.0, a_max=3000.0)

        # Next We Normalize Each Channel By Subtracting The Mean And Scaling By The Inverse Of The Standard Deviation
        for channel_index in range(img.shape[-1]):
            channel = img[:, :, channel_index]
            channel_mean = np.mean(channel)
            channel_stdev = np.std(channel)
            channel -= channel_mean
            channel *= (1.0 / channel_stdev)
        return img
