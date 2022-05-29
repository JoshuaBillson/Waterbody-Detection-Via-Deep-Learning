import os
import json
from tensorflow.keras.metrics import MeanIoU, Recall, Precision
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from models.losses import DiceBCELoss
from data_loader import DataLoader, create_patches, show_samples, load_dataset
from models.layers import preprocessing_layer
from models import get_model
from config import get_epochs, get_model_type, get_timestamp, get_learning_rate
from callbacks import get_callbacks, create_callback_dirs


GPU = 5

def main():
    # Get Project Configuration
    with open('config.json') as f:
        config = json.loads(f.read())

    # Create Data Loader
    loader = DataLoader(timestamp=get_timestamp(config))

    # Generate Patches
    if config["generate_patches"]:
        create_patches(loader, config["show_data"])

    # Show Samples Data
    if config["show_samples"]:
        show_samples(loader)

    # Load Dataset
    train_data, val_data, test_data = load_dataset(loader, config)
    print(len(train_data))

    # Get Callbacks
    create_callback_dirs()
    callbacks = get_callbacks(config)

    # Create Model
    model = get_model(config)
    model.summary()
    model.compile(loss=DiceBCELoss, optimizer=Adam(learning_rate=get_learning_rate(config)), metrics=[MeanIoU(num_classes=2), Precision(), Recall()])

    # If Model Is Loaded From Checkpoint, Find The Last Epoch
    initial_epoch = 0
    if get_model_type(config) in os.listdir("checkpoints"):
        with open(f"logs/csv/{get_model_type(config)}.csv") as csvfile:
            last_line = csvfile.readlines()[-1]
            initial_epoch = int(last_line.split(",")[0]) + 1


    # Train Model
    print(f"EPOCH: {initial_epoch}")
    if config["train"]:
        model.fit(train_data, epochs=get_epochs(config)+initial_epoch, verbose=1, callbacks=callbacks, validation_data=val_data, initial_epoch=initial_epoch)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{GPU}"
    main()
