import math
import os
from typing import Dict, Any, List
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, Callback, LearningRateScheduler
from models.utils import predict_batch
from backend.data_loader import DataLoader
from config import get_create_logs


class PredictionCallback(Callback):
    def __init__(self, val_data: List[int], model: Model, data_loader: DataLoader, config: Dict[str, Any]):
        super().__init__()
        self.val_data = val_data
        self.model = model
        self.data_loader = data_loader
        self.config = config
        
    def on_epoch_end(self, epoch, logs=None):
        """
        Call after every epoch to predict mask
        :param epoch: Current epoch
        :returns: Nothing
        """
        predict_batch(self.val_data, self.data_loader, self.model, self.config, "validation", 5.0)


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
    checkpoint = ModelCheckpoint(f"checkpoints/{model.name}", save_best_only=False, monitor='val_loss', mode='min')
    prediction_logger = PredictionCallback(val_data, model, data_loader, config)
    learning_rate_scheduler = LearningRateScheduler(lr_scheduler)
    return [tensorboard, csv, checkpoint, prediction_logger, learning_rate_scheduler] if get_create_logs(config) else [learning_rate_scheduler]


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
