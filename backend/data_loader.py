import os
import math
import json
import random
import shutil
import cv2
import pandas
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Tuple, Sequence, List, Dict, Any
from tensorflow.image import flip_up_down, flip_left_right, rot90
from tensorflow.keras.utils import Sequence as KerasSequence
from tensorflow.keras.models import Model
from backend.utils import adjust_rgb
from backend.metrics import MIOU
from backend.config import get_patch_size, get_waterbody_transfer
from models.utils import evaluate_model


class DataLoader:
    """A class to generate patches, load the dataset from disk, and show statistics."""
    def __init__(self, tile_size: int = 1024, timestamp: int = 1):
        self.timestamp = timestamp
        self.tile_size = tile_size
        self.folders = {1: "2018.04", 2: "2018.12", 3: "2019.02"}

    def get_rgb_features(self, patch_number: int, preprocess_img: bool = True) -> np.ndarray:
        """
        Get RGB Features Matching The Given Patch Number
        :param patch_number: The number of the patch we want to retrieve which must be in the range [min_patch, max_patch]
        :return: The RGB features of the matching patch,
        """
        return self.read_image(f"data/{self.folders.get(self.timestamp, 1)}/patches/rgb/rgb.{patch_number}.tif", preprocess_img=preprocess_img)

    def get_nir_features(self, patch_number: int, preprocess_img: bool = True) -> np.ndarray:
        """
        Get NIR Features Matching The Given Patch Number
        :param patch_number: The number of the patch we want to retrieve which must be in the range [min_patch, max_patch]
        :return: The NIR features of the matching patch,
        """
        return self.read_image(f"data/{self.folders.get(self.timestamp, 1)}/patches/nir/nir.{patch_number}.tif", preprocess_img=preprocess_img)

    def get_swir_features(self, patch_number: int, preprocess_img: bool = True) -> np.ndarray:
        """
        Get SWIR Features Matching The Given Patch Number
        :param patch_number: The number of the patch we want to retrieve which must be in the range [min_patch, max_patch]
        :return: The SWIR features of the matching patch,
        """
        return self.read_image(f"data/{self.folders.get(self.timestamp, 1)}/patches/swir/swir.{patch_number}.tif", preprocess_img=preprocess_img)

    def get_mask(self, patch_number: int) -> np.ndarray:
        """
        Get The Mask For The Given Patch Number
        :param patch_number: The number of the patch we want to retrieve which must be in the range [min_patch, max_patch]
        :return: The mark of the matching patch.
        """
        return self.read_image(f"data/{self.folders.get(self.timestamp, 1)}/patches/mask/mask.{patch_number}.tif")

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
            ax[1].imshow(adjust_rgb(rgb_sample))
            ax[2].imshow(DataLoader.threshold_channel(nir_sample))
            ax[3].imshow(DataLoader.threshold_channel(swir_sample))

        # Save Figure
        plt.savefig(f"images/samples.{self.timestamp}.png", dpi=2500, bbox_inches='tight')
        plt.show()

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

    @staticmethod
    def threshold_channel(channel, threshold=3000):
        return np.clip(channel, 0, threshold)


class ImgSequence(KerasSequence):
    def __init__(self, data_loader: DataLoader, patches: List[int], batch_size: int = 32, bands: Sequence[str] = None, augment_data: bool = False, shuffle: bool = True):
        # Initialize Member Variables
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.bands = ["RGB"] if bands is None else bands
        self.indices = patches
        self.augment_data = augment_data
        self.shuffle = shuffle

        # Shuffle Patches
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # Confirm Waterbody Transfer
        self._apply_water_transfer()

    def __len__(self) -> int:
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, idx):
        # Create Batch
        feature_batches = {"RGB": [], "NIR": [], "SWIR": [], "mask": []}
        batch = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        for b in batch:

            # Get Mask And Features For Patch b
            features = self._get_features(b)

            # Augment Data
            if self.augment_data:
                self.augment_features(features)
            
            # Add Features To Batch
            for key, val in features.items():
                feature_batches[key].append(DataLoader.normalize_channels(val.astype("float32")) if key != "mask" else val)

        # Return Batch
        return [np.array(feature_batches[band]).astype("float32") for band in ("RGB", "NIR", "SWIR") if len(feature_batches[band]) > 0], np.array(feature_batches["mask"]).astype("float32")
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def get_patch_indices(self) -> List[int]:
        """
        Get the patch indices for all patches in this dataset
        :returns: The list of all patch indices in this dataset.
        """
        return list(self.indices)

    def predict_batch(self, model: Model, directory: str):
        """
        Predict on a batch of feature samples and save the resulting prediction to disk alongside its mask and MIoU performance
        :param batch: A list of patch indexes on which we want to predict
        :param data_loader: An object for reading patches from the disk
        :param model: The model whose prediction we are interested in
        :param config: The script configuration stored as a dictionary; typically read from an external file
        :param directory: The name of the directory in which we want to save the model predictions
        :param threshold: We filter out patches whose water content percentage is below this value
        :return: Nothing
        """
        # Create Directory To Save Predictions
        if directory not in os.listdir():
            os.mkdir(directory)
        model_directory = f"{directory}/{model.name}"
        if model.name in os.listdir(directory):
            shutil.rmtree(model_directory)
        os.mkdir(model_directory)

        # Iterate Over All Patches In Batch
        MIoUs, MIoU = [], MIOU()
        for patch_index in self.indices:

            # Load Features And Mask
            features = self._get_features(patch_index)
            mask = features["mask"]
        
            # Get Prediction
            prediction = model.predict([np.array([DataLoader.normalize_channels(features[band].astype("float32"))]) for band in self.bands])
            MIoUs.append([patch_index, MIoU(mask.astype("float32"), prediction).numpy()])

            # Plot Features
            i = 0
            _, axs = plt.subplots(1, len(self.bands) + 2)
            for band in self.bands:
                axs[i].imshow(adjust_rgb(features[band], gamma=0.5) if band == "RGB" else features[band])
                axs[i].set_title(band, fontsize=6)
                axs[i].axis("off")
                i += 1
            
            # Plot Ground Truth
            axs[i].imshow(mask)
            axs[i].set_title("Ground Truth", fontsize=6)
            axs[i].axis("off")
            i += 1

            # Plot Prediction
            axs[i].imshow(np.where(prediction < 0.5, 0, 1)[0])
            axs[i].set_title(f"Prediction ({MIoUs[-1][1]:.3f})", fontsize=6)
            axs[i].axis("off")
            i += 1

            # Save Prediction To Disk
            plt.tight_layout()
            plt.savefig(f"{model_directory}/prediction.{patch_index}.png", dpi=300, bbox_inches='tight')
            plt.cla()
            plt.close()
        
        # Save MIoU For Each Patch
        summary = np.array(MIoUs)
        df = pandas.DataFrame(summary[:, 1:], columns=["MIoU"], index=summary[:, 0].astype("int32"))
        df.to_csv(f"{model_directory}/Evaluation.csv", index_label="patch")

        # Evaluate Final Performance
        results = evaluate_model(model, self)
        df = pandas.DataFrame(np.reshape(np.array(results), (1, len(results))), columns=model.metrics_names)
        df.to_csv(f"{model_directory}/Overview.csv", index=False)
        return results
    
    def augment_features(self, features: Dict[str, np.ndarray]) -> None:
        return None

    def _apply_water_transfer(self):
        print("WATERBODY TRANSFER: FALSE")

    def _water_content(self, mask: np.ndarray) -> float:
        return np.sum(mask) / mask.size * 100.0
    
    def _get_features(self, patch: int) -> Dict[str, np.ndarray]:
        # Get Mask
        features = {"mask": self.data_loader.get_mask(patch)}

        # Get RGB Features
        if "RGB" in self.bands:
            features["RGB"] = self.data_loader.get_rgb_features(patch, preprocess_img=False)

        # Get NIR Features
        if "NIR" in self.bands:
            features["NIR"] = self.data_loader.get_nir_features(patch, preprocess_img=False)

        # Get SWIR Features
        if "SWIR" in self.bands:
            features["SWIR"] = self.data_loader.get_swir_features(patch, preprocess_img=False)
            features["SWIR"] = np.resize(cv2.resize(features["SWIR"], (512, 512), interpolation = cv2.INTER_AREA), (512, 512, 1))

        return features


class TransferImgSequence(ImgSequence):
    """A class to demonstrate the waterbody transfer method."""

    def __init__(self, data_loader: DataLoader, patches: List[int], batch_size: int = 32, bands: Sequence[str] = None, augment_data: bool = False, shuffle: bool = True):
        # If We Want To Apply Waterbody Transferrence, Locate All Patches With At Least 10% Water
        super().__init__(data_loader, patches, batch_size, bands, augment_data, shuffle)

        # If We Want To Apply Waterbody Transferrence, Locate All Patches With At Least 10% Water
        self.transfer_patches = []
        if self.augment_data:
            for source_index in self.indices:
                source_mask = self.data_loader.get_mask(source_index)
                if 2.0 < self._water_content(source_mask):
                    print(source_index, self._water_content(source_mask))
                    self.transfer_patches.append(source_index)

    def augment_features(self, features: Dict[str, np.ndarray]) -> None:
        """
        Applies data augmentation to a given patch
        :param features: A dictionary of input features (RGB, NIR, SWIR) and the mask to which we want to apply augmentation
        """
        # Transfer Waterbody
        self.transfer_waterbody(features)

        # Apply Random Horizontal/Vertical Flip with 25% Probability
        outcome_1 = random.randint(1, 100)
        outcome_2 = random.randint(1, 100)
        if outcome_1 <= -1:
            for feature_index in features.keys():
                features[feature_index] = flip_up_down(features[feature_index]).numpy() if outcome_2 <= 50 else flip_left_right(features[feature_index]).numpy()

        # Apply Random Counter Clockwise Rotation Of 90, 180, or 270 Degrees With 25% Probability
        outcome_1 = random.randint(1, 100)
        outcome_2 = random.randint(1, 3)
        if outcome_1 <= -1:
            for feature_index in features.keys():
                features[feature_index] = rot90(features[feature_index], k=outcome_2).numpy()
        

    def transfer_waterbody(self, features: Dict[str, np.ndarray], threshold: float = 0.0) -> None:
        """
        Given a destination patch, transfers a waterbody to the destination if the destination has a water content below a given threshold.
        :param features: A dictionary of input features (RGB, NIR, SWIR) and the mask to which we want to transfer a waterbody
        :param threshold: The water content threshold below which we apply waterbody transfer
        """
        # Acquire Probability Of Applying Transfer
        if self._water_content(features["mask"]) == 0.0:

            # Get Source Mask
            assert len(self.transfer_patches) > 0, "Error: Cannot Augment Dataset Without Transfer Patches!"
            source_index = self.transfer_patches[random.randint(0, len(self.transfer_patches) - 1)]
            source_features = self._get_features(source_index)
            source_mask = source_features["mask"]

            # Apply Waterbody Transfer To Each Feature Map
            for band in self.bands:

                # Get Source Feature
                source_feature = source_features[band]

                # Extract Waterbody From Source Feature 
                waterbody = source_mask * source_feature

                # Remove Waterbody Region From Destination Feature
                features[band] *= np.where(source_mask == 1, 0, 1).astype("uint16")

                # Transfer Waterbody To Destination Feature Map
                features[band] += waterbody
                
            features["mask"] = np.where((features["mask"] + source_mask) >= 1, 1, 0).astype("uint16")

    def _apply_water_transfer(self):
        if self.augment_data:
            print("WATERBODY TRANSFER: TRUE")
        else:
            print("WATERBODY TRANSFER: FALSE")


def load_dataset(loader: DataLoader, config) -> Tuple[ImgSequence, ImgSequence, ImgSequence]:
    """
    Load the training, validation, and test datasets
    :param loader: The DataLoader that will be used to read the patches from disk
    :param config: The script configuration encoded as a Python dictionary; typically read from an external file
    :returns: The training, validation, and test data as a tuple of the form (train, validation, test)
    """
    # Read Parameters From Config File
    bands = config["hyperparameters"]["bands"]
    batch_size = config["hyperparameters"]["batch_size"]
    patch_size = get_patch_size(config)

    # Read Batches From JSON File
    with open(f"batches/{patch_size}.json") as f:
        batch_file = json.loads(f.read())
    
    # Create Train, Validation, And Test Data
    Constructor = ImgSequence if not get_waterbody_transfer(config) else TransferImgSequence
    train_data = Constructor(loader, batch_file["train"], bands=bands, batch_size=batch_size, augment_data=True)
    val_data = Constructor(loader, batch_file["validation"], bands=bands, batch_size=batch_size, shuffle=False)
    test_data = Constructor(loader, batch_file["test"], bands=bands, batch_size=batch_size, shuffle=False)
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
