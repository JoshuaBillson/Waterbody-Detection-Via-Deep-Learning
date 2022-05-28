import os
from typing import Dict, Any, List
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, Callback
from models.utils import get_model_name


def get_callbacks(config: Dict[str, Any]) -> List[Callback]:
    """
    Get the callbacks to be used when training the model
    :param config: A dictionary storing the script configuration
    :returns: A list of Keras callbacks
    """
    tensorboard = TensorBoard(log_dir=f"logs/tensorboard/{get_model_name(config)}")
    csv = CSVLogger(filename=f"logs/csv/{get_model_name(config)}.csv", append=True)
    checkpoint = ModelCheckpoint(f"checkpoints/{get_model_name(config)}", save_best_only=False)
    return [tensorboard, csv, checkpoint]


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
