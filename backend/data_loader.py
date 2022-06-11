import os
import math
import random
import shutil
import pandas
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Sequence, List, Dict, Any
from tensorflow.keras.utils import Sequence as KerasSequence
from tensorflow.keras.models import Model
from backend.utils import adjust_rgb
from backend.metrics import MIoU


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

    def get_bounds(self) -> Tuple[int, int]:
        """Returns the indices of the lowest and highest numbered patches respectively."""
        files = os.listdir(f"data/{self.folders.get(self.timestamp, 1)}/patches/mask")
        min_patch = min([int(x.split(".")[1]) for x in files])
        max_patch = max([int(x.split(".")[1]) for x in files])
        return min_patch, max_patch

    def get_batch(self, batch: List[int], bands: List[str], threshold: float = 0.0) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        masks, rgb_features, nir_features, swir_features, indices = [], [], [], [], []
        for patch in batch:
            mask = self.get_mask(patch)
            
            # Return Patches Whose Water Content Meets The Threshold 
            if (np.sum(mask) / mask.size * 100.0) >= threshold:

                # Read Mask And Index
                masks.append(mask)

                # Store Patch Index
                indices.append(patch)

                # Read RGB Features
                if "RGB" in bands:
                    rgb_features.append(self.get_rgb_features(patch))

                # Read NIR Features
                if "NIR" in bands:
                    nir_features.append(self.get_nir_features(patch))

                # Read SWIR Features
                if "SWIR" in bands:
                    swir_features.append(self.get_swir_features(patch))
        
        # Return Batch
        features = []
        if "RGB" in bands:
            features.append(np.array(rgb_features).astype("float32"))
        if "NIR" in bands:
            features.append(np.array(nir_features).astype("float32"))
        if "SWIR" in bands:
            features.append(np.array(swir_features).astype("float32"))
        return features, np.array(masks), np.array(indices)
                


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

        # If We Want To Apply Waterbody Transferrence, Locate All Patches With At Least 10% Water
        self.transfer_patches = []
        if self.augment_data:
            for source_index in self.indices:
                source_mask = self.data_loader.get_mask(source_index)
                if self._water_content(source_mask) > 10:
                    print(source_index, self._water_content(source_mask))
                    self.transfer_patches.append(source_index)

    def __len__(self) -> int:
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, idx):
        # Create Batch
        rgb_batch, nir_batch, swir_batch, mask_batch = [], [], [], []
        batch = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        for b in batch:

            # Get Mask And Features For Patch b
            features, mask = self._get_features(b)

            # Augment Data
            if self.augment_data:
                features, mask = self.augment_patch(features, mask)
            
            # Add Features To Batch
            mask_batch.append(mask)
            if "RGB" in self.bands:
                rgb_batch.append(DataLoader.normalize_channels(features[0].astype("float32")))
            if "NIR" in self.bands:
                nir_batch.append(DataLoader.normalize_channels(features[1 if "RGB" in self.bands else 0].astype("float32")))
            if "SWIR" in self.bands:
                swir_batch.append(DataLoader.normalize_channels(features[len(self.bands) - 1].astype("float32")))

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
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def get_patch_indices(self) -> List[int]:
        """
        Get the patch indices for all patches in this dataset
        :returns: The list of all patch indices in this dataset.
        """
        return list(self.indices)

    def predict_batch(self, model: Model, directory: str) -> None:
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
        MIoUs = []
        for patch_index in self.indices:

            # Load Features And Mask
            features, mask = self._get_features(patch_index)
        
            # Get Prediction
            prediction = model.predict([np.array([DataLoader.normalize_channels(feature.astype("float32"))]) for feature in features])

            # Plot Prediction And Save To Disk
            MIoUs.append([patch_index, MIoU(mask.astype("float32"), prediction).numpy()])
            _, axs = plt.subplots(1, 3)
            axs[0].imshow(adjust_rgb(features[0], gamma=0.5) if self.bands[0] == "RGB" else features[0])
            axs[0].set_title(self.bands[0])
            axs[1].imshow(mask)
            axs[1].set_title("Ground Truth")
            axs[2].imshow(np.where(prediction < 0.5, 0, 1)[0])
            axs[2].set_title(f"Prediction ({MIoUs[-1][1]:.3f})")
            plt.tight_layout()
            plt.savefig(f"{model_directory}/prediction.{patch_index}.png", dpi=300, bbox_inches='tight')
            plt.cla()
            plt.close()
        
        # Save MIoU For Each Patch
        summary = np.array(MIoUs)
        df = pandas.DataFrame(summary[:, 1:], columns=["MIoU"], index=summary[:, 0].astype("int32"))
        df.to_csv(f"{model_directory}/Evaluation.csv", index_label="patch")

        # Evaluate Final Performance
        results = model.evaluate(self)
        for metric, value in zip(model.metrics_names, results):
            print(metric, value)
        df = pandas.DataFrame(np.reshape(np.array(results), (1, len(results))), columns=model.metrics_names)
        df.to_csv(f"{model_directory}/Overview.csv", index=False)
    
    def show_agumentation(self):
        for patch in self.indices:
            features, mask = self._get_features(patch)
            self.augment_patch(features, mask, index=patch, show_examples=True)

    def augment_patch(self, patches: np.ndarray, mask: np.ndarray, index: int = 1, show_examples: bool = False, threshold: float = 0.0) -> Tuple[np.ndarray]:
        # while self._water_content(mask) < threshold:
        if self._water_content(mask) <= threshold:

            # Get Source Mask
            assert len(self.transfer_patches) > 0, "Error: Cannot Augment Dataset Without Transfer Patches!"
            source_index = self.transfer_patches[random.randint(0, len(self.transfer_patches) - 1)]
            source_mask = self.data_loader.get_mask(source_index)

            # Get Source Features
            methods = {"RGB": self.data_loader.get_rgb_features, "NIR": self.data_loader.get_nir_features, "SWIR": self.data_loader.get_swir_features}
            for patch, band in zip(patches, self.bands):
                source_feature = methods[band](source_index, preprocess_img=False)

                # Extract Waterbody From Source Feature 
                waterbody = source_mask * source_feature

                # Remove Waterbody Region From Destination Feature
                augmented_patch = patch * np.where(source_mask == 1, 0, 1).astype("uint16")

                # Transfer Waterbody To Destination Feature Map
                augmented_patch += waterbody
                
                # Plot Augmented Patch
                if show_examples:
                    _, axs = plt.subplots(1, 6)
                    axs[0].imshow(source_mask)
                    axs[0].set_title("Src. Mask", fontsize=6)
                    axs[1].imshow(adjust_rgb(source_feature, gamma=0.5) if band == "RGB" else source_feature)
                    axs[1].set_title("Src. Features", fontsize=6)
                    axs[2].imshow(mask)
                    axs[2].set_title("Dest. Mask", fontsize=6)
                    axs[3].imshow(adjust_rgb(patch, gamma=0.5) if band == "RGB" else patch)
                    axs[3].set_title("Dest. Features", fontsize=6)
                    axs[4].imshow(np.where((mask + source_mask) >= 1, 1, 0).astype("uint16"))
                    axs[4].set_title("Final Mask", fontsize=6)
                    axs[5].imshow(adjust_rgb(augmented_patch, gamma=0.5) if band == "RGB" else augmented_patch)
                    axs[5].set_title("Final Features", fontsize=6)
                    plt.tight_layout()
                    plt.savefig(f"faz/transfer_{index}_{band}.png", dpi=1000)
                    plt.close()

            mask = np.where((mask + source_mask) >= 1, 1, 0).astype("uint16")

        return patches, mask

    def _water_content(self, mask: np.ndarray) -> float:
        return np.sum(mask) / mask.size * 100.0
    
    def _get_features(self, patch: int) -> Tuple[List[np.ndarray], np.ndarray]:
        # Get Mask
        mask = self.data_loader.get_mask(patch)

        # Get RGB Features
        rgb_feature = self.data_loader.get_rgb_features(patch, preprocess_img=False) if "RGB" in self.bands else None

        # Get NIR Features
        nir_feature = self.data_loader.get_nir_features(patch, preprocess_img=False) if "NIR" in self.bands else None

        # Get SWIR Features
        swir_feature = self.data_loader.get_swir_features(patch, preprocess_img=False) if "SWIR" in self.bands else None

        # Collect Features 
        features = list(filter(lambda x: x is not None, (rgb_feature, nir_feature, swir_feature)))

        return features, mask


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
    patches = np.array(range(1, 3601))
    np.random.shuffle(patches)
    train_data = ImgSequence(loader, patches[0:2700], bands=bands, batch_size=batch_size)
    val_data = ImgSequence(loader, patches[2700:3000], bands=bands, batch_size=batch_size, shuffle=False)
    test_data = ImgSequence(loader, patches[3000:3600], bands=bands, batch_size=batch_size, shuffle=False)
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
