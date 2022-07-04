import gc
import os
import math
import statistics
import json
import random
import shutil
import pandas
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Sequence, List, Dict
from tensorflow.image import flip_up_down, flip_left_right, rot90
from tensorflow.keras.utils import Sequence as KerasSequence
from tensorflow.keras.models import Model
from tensorflow.keras.backend import clear_session
from backend.utils import adjust_rgb
from backend.metrics import MIOU
from backend.config import get_timestamp, get_waterbody_transfer, get_random_subsample, get_fusion_head, get_water_threshold
from models.utils import evaluate_model
from backend.data_loader import DataLoader


class ImgSequence(KerasSequence):
    def __init__(self, timestamp: int, tiles: List[int], batch_size: int = 32, bands: Sequence[str] = None, is_train: bool = False, random_subsample: bool = False, upscale_swir: bool = True):
        # Initialize Member Variables
        self.data_loader = DataLoader(timestamp, overlapping_patches=is_train, random_subsample=(random_subsample and is_train), upscale_swir=upscale_swir)
        self.batch_size = batch_size
        self.bands = ["RGB"] if bands is None else bands
        self.indices = []
        self.is_train = is_train

        # Generate Patch Indices From Tiles
        for tile in tiles:
            self.indices += [((tile * 100) + patch_index) for patch_index in range(1, (10 if is_train else 5))]

        # Shuffle Patches
        if self.is_train:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, idx, normalize_data=True):
        # Create Batch
        feature_batches = {"RGB": [], "NIR": [], "SWIR": [], "mask": []}
        batch = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        for b in batch:

            # Get Mask And Features For Patch b
            features = self._get_features(b)

            # Augment Data
            if self.is_train:
                self.augment_features(features)
            
            # Add Features To Batch
            for key, val in features.items():
                if normalize_data:
                    feature_batches[key].append(DataLoader.normalize_channels(val.astype("float32")) if key != "mask" else val)
                else:
                    feature_batches[key].append(val)

        # Return Batch
        return [np.array(feature_batches[band]).astype("float32") for band in ("RGB", "NIR", "SWIR") if len(feature_batches[band]) > 0], np.array(feature_batches["mask"]).astype("float32")
    
    def on_epoch_end(self):
        if self.is_train:
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
        MIoUs, MIoU, i = [], MIOU(), 0
        for batch in range(len(self)):

            # Get Batch
            features, masks = self.__getitem__(batch, normalize_data=False)
            normalized_features, _ = self.__getitem__(batch)
            rgb_features = features[0] if "RGB" in self.bands else None
            nir_features = features[1 if "RGB" in self.bands else 0] if "NIR" in self.bands else None
            swir_features = features[2] if "SWIR" in self.bands else None

            # Get Prediction
            predictions = model.predict(normalized_features)

            # Iterate Over Each Prediction In The Batch
            for p in range(predictions.shape[0]):

                mask = masks[p, ...]
                prediction = predictions[p, ...]
                MIoUs.append([self.indices[i], MIoU(mask, prediction).numpy()])

                # Plot Features
                col = 0
                _, axs = plt.subplots(1, len(self.bands) + 2)
                for band, feature in zip(["RGB", "NIR", "SWIR"], [rgb_features, nir_features, swir_features]):
                    if feature is not None:
                        axs[col].imshow(adjust_rgb(feature[p, ...], gamma=0.5) if feature.shape[-1] == 3 else feature[p, ...])
                        axs[col].set_title(band, fontsize=6)
                        axs[col].axis("off")
                        col += 1
                
                # Plot Ground Truth
                axs[col].imshow(mask)
                axs[col].set_title("Ground Truth", fontsize=6)
                axs[col].axis("off")
                col += 1

                # Plot Prediction
                axs[col].imshow(np.where(prediction < 0.5, 0, 1))
                axs[col].set_title(f"Prediction ({MIoUs[-1][1]:.3f})", fontsize=6)
                axs[col].axis("off")
                col += 1

                # Save Prediction To Disk
                plt.tight_layout()
                plt.savefig(f"{model_directory}/prediction.{self.indices[i]}.png", dpi=300, bbox_inches='tight')
                plt.cla()
                plt.close()

                # Housekeeping
                gc.collect()
                clear_session()
                i += 1
        
        # Save MIoU For Each Patch
        # summary = np.array(MIoUs)
        # df = pandas.DataFrame(summary[:, 1:], columns=["MIoU"], index=summary[:, 0].astype("int32"))
        # df.to_csv(f"{model_directory}/Evaluation.csv", index_label="patch")

        # Evaluate Final Performance
        results = evaluate_model(model, self)
        df = pandas.DataFrame(np.reshape(np.array(results), (1, len(results))), columns=model.metrics_names)
        df.to_csv(f"{model_directory}/Overview.csv", index=False)
        return results
    
    def augment_features(self, features: Dict[str, np.ndarray]) -> None:
        # Apply Random Horizontal/Vertical Flip with 50% Probability
        self._rotate_patch(features)
        self._flip_patch(features)
    
    def _rotate_patch(self, patch: Dict[str, np.ndarray]) -> None:
        """Apply 90 degree clockwise rotation to a given patch with 25% probability"""
        outcome = random.randint(1, 100)
        if outcome <= 25:
            for band in patch.keys():
                patch[band] = rot90(patch[band], k=1).numpy()

    def _flip_patch(self, patch: Dict[str, np.ndarray]) -> None:
        """Apply a horizontal flip with 50% probability and a vertical flip with 50% probability to a given patch"""
        outcome_1 = random.randint(1, 100)
        outcome_2 = random.randint(1, 100)
        for band in patch.keys():
            patch[band] = flip_up_down(patch[band]).numpy() if outcome_1 <= 50 else patch[band]
            patch[band] = flip_left_right(patch[band]).numpy() if outcome_2 <= 50 else patch[band]

    def _water_content(self, mask: np.ndarray) -> float:
        return np.sum(mask) / mask.size * 100.0
    
    def _get_features(self, patch: int, subsample: bool = True) -> Dict[str, np.ndarray]:
        return self.data_loader.get_features(patch, self.bands, subsample=subsample)


class WaterbodyTransferImgSequence(ImgSequence):
    def __init__(self, timestamp: int, tiles: List[int], batch_size: int = 32, bands: Sequence[str] = None, is_train: bool = False, random_subsample: bool = False, upscale_swir: bool = True, water_threshold: int = 5):
        super().__init__(timestamp, tiles, batch_size, bands, is_train, random_subsample, upscale_swir)
        self.water_threshold = water_threshold

    """A data pipeline that returns tiles with transplanted waterbodies"""
    def _get_features(self, patch: int, subsample: bool = True) -> Dict[str, np.ndarray]:
        tile_index = patch // 100
        return self.data_loader.get_features(patch, self.bands, tile_dir="tiles" if tile_index <= 400 else f"transplanted_tiles_{self.water_threshold}")


def load_dataset(config) -> Tuple[ImgSequence, ImgSequence, ImgSequence]:
    """
    Load the training, validation, and test datasets
    :param config: The script configuration encoded as a Python dictionary; typically read from an external file
    :returns: The training, validation, and test data as a tuple of the form (train, validation, test)
    """
    # Read Parameters From Config File
    bands = config["hyperparameters"]["bands"]
    batch_size = config["hyperparameters"]["batch_size"]

    # Read Batches From JSON File
    water_threshold = get_water_threshold(config)
    batch_filename = f"batches/transplanted_tiles_{water_threshold}.json" if get_waterbody_transfer(config) else "batches/tiles.json"
    with open(batch_filename) as f:
        batch_file = json.loads(f.read())
    
    # Choose Type Of Data Pipeline Based On Project Config
    Constructor = WaterbodyTransferImgSequence if get_waterbody_transfer(config) else ImgSequence

    # Create Train, Validation, And Test Data
    upscale_swir = get_fusion_head(config) != "paper"
    if get_waterbody_transfer(config):
        train_data = WaterbodyTransferImgSequence(get_timestamp(config), batch_file["train"], batch_size=batch_size, bands=bands, is_train=True, random_subsample=get_random_subsample(config), upscale_swir=upscale_swir, water_threshold=water_threshold)
    else:
        train_data = ImgSequence(get_timestamp(config), batch_file["train"], batch_size=batch_size, bands=bands, is_train=True, random_subsample=get_random_subsample(config), upscale_swir=upscale_swir)
    val_data = ImgSequence(get_timestamp(config), batch_file["validation"], batch_size=12, bands=bands, is_train=False, upscale_swir=upscale_swir)
    test_data = ImgSequence(get_timestamp(config), batch_file["test"], batch_size=12, bands=bands, is_train=False, upscale_swir=upscale_swir)
    return train_data, val_data, test_data