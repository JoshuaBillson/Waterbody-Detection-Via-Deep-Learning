import shutil
import math
import os
from typing import Dict, Any, List
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, Callback, LearningRateScheduler
import numpy as np
from matplotlib import pyplot as plt
from data_loader import DataLoader
from config import get_bands


class PredictionCallback(Callback):
    def __init__(self, val_data: List[int], model: Model, data_loader: DataLoader, config: Dict[str, Any]):
        super().__init__()
        self.val_data = val_data
        self.model = model
        self.data_loader = data_loader
        self.config = config
        
    def on_epoch_end(self, epoch, logs=None):
        """
        Summary:
            call after every epoch to predict mask
        Arguments:
            epoch (int): current epoch
        Output:
            save predict mask
        """
        # Filter Out patches With Less Than 5% Water Coverage
        masks, features, indices, bands = [], [], [], get_bands(self.config)
        for patch in self.val_data:
            mask = self.data_loader.get_mask(patch)
            if (np.sum(mask) / mask.size * 100.0) >= 5.0:

                # Keep Mask And Index
                masks.append(mask)
                indices.append(patch)

                # Keep RGB Feature
                feature_list = []
                if "RGB" in bands:
                    rgb_feature = self.data_loader.get_rgb_features(patch)
                    feature_list.append(np.reshape(rgb_feature, (1, rgb_feature.shape[0], rgb_feature.shape[1], rgb_feature.shape[2])))

                # Keep NIR Feature
                if "NIR" in bands:
                    nir_feature = self.data_loader.get_nir_features(patch)
                    feature_list.append(np.reshape(nir_feature, (1, nir_feature.shape[0], nir_feature.shape[1], nir_feature.shape[2])))

                # Keep SWIR Feature
                if "SWIR" in bands:
                    swir_feature = self.data_loader.get_swir_features(patch)
                    feature_list.append(np.reshape(swir_feature, (1, swir_feature.shape[0], swir_feature.shape[1], swir_feature.shape[2])))
                
                features.append(feature_list)

        # Create Predictions Directory
        if "predictions" not in os.listdir():
            os.mkdir("predictions")
        if self.model.name in os.listdir("predictions"):
            shutil.rmtree(f"predictions/{self.model.name}")
        os.mkdir(f"predictions/{self.model.name}")

        # Save Model Predictions To Disk
        print(len(masks))
        for mask, feature, index in zip(masks, features, indices):

            # Make Prediction
            prediction = self.model.predict(feature)
            
            # Plot Prediction
            fig, axs = plt.subplots(1, 2)
            fig.tight_layout()
            axs[0].imshow(mask)
            axs[0].set_title("Mask")
            axs[1].imshow(np.where(prediction < 0.5, 0, 1)[0, ...])
            axs[1].set_title(self.model.name)
            plt.savefig(f"predictions/{self.model.name}/prediction.{index + 1}.png", dpi=300, bbox_inches='tight')
            plt.cla()
            plt.close()


def lr_scheduler(epoch, learning_rate):
        """
        learning rate decrease according to the model performance
        :param epoch: The current epoch
        :returns: The new learning rate
        """
        return learning_rate * math.pow(0.5, epoch // 10)


def get_callbacks(config: Dict[str, Any], val_data: List[int], model: Model, data_loader: DataLoader) -> List[Callback]:
    """
    Get the callbacks to be used when training the model
    :param config: A dictionary storing the script configuration
    :returns: A list of Keras callbacks
    """
    tensorboard = TensorBoard(log_dir=f"logs/tensorboard/{model.name}")
    csv = CSVLogger(filename=f"logs/csv/{model.name}.csv", append=True)
    checkpoint = ModelCheckpoint(f"checkpoints/{model.name}", save_best_only=False)
    prediction_logger = PredictionCallback(val_data, model, data_loader, config)
    learning_rate_scheduler = LearningRateScheduler(lr_scheduler)
    return [tensorboard, csv, checkpoint, prediction_logger, learning_rate_scheduler]


def create_callback_dirs() -> None:
    """Create the directories needed by our callbacks if they don't already exist"""
    # Create Logs Directory
    if "logs" not in os.listdir():
        os.mkdir("logs")

    # Create Tensorboard Directory
    if "tensorboard" not in os.listdir("logs"):
        os.mkdir("logs/tensorboard")

    # Create CSV Directory
    if "csv" not in os.listdir("logs"):
        os.mkdir("logs/csv")

    # Create Checkpoint Directory
    if "checkpoints" not in os.listdir():
        os.mkdir("checkpoints")
