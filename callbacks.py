from typing import Dict, Any
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from models.utils import get_model_name


def get_callbacks(config: Dict[str, Any]):
    tensorboard = TensorBoard(log_dir=f"logs/tensorboard/{get_model_name(config)}")
    return [tensorboard]
