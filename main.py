import os
import sys
import json
import numpy as np
from tensorflow.keras.metrics import MeanIoU, Recall, Precision
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import mixed_precision

from backend.metrics import MIoU
from models.losses import DiceBCELoss, JaccardBCELoss
from models.utils import predict_batch
from backend.data_loader import DataLoader, create_patches, show_samples, load_dataset
from models import get_model
from config import get_epochs, get_model_type, get_timestamp, get_learning_rate
from backend.callbacks import get_callbacks, create_callback_dirs


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
    batch, masks = train_data[0]
    sample = batch[0]
    mask = masks[0]
    print(sample.shape, mask.shape)
    # train_data.augment_patch(sample, mask)
    # print(batch.shape, sample.shape, np.mean(sample[:, :, 0]), np.std(sample[:, :, 0]))
    # sample = batch[1]
    # print(batch.shape, sample.shape, np.mean(sample[:, :, 0]), np.std(sample[:, :, 0]))

    # Create Callback Directories
    create_callback_dirs()

    # Create Model
    model = get_model(config)
    model.summary()
    model.compile(loss=JaccardBCELoss, optimizer=Adam(learning_rate=get_learning_rate(config)), metrics=[MIoU, Precision(), Recall()])

    # Get Callbacks
    callbacks = get_callbacks(config, val_data.get_patch_indices(), model, loader)

    # If Model Is Loaded From Checkpoint, Find The Last Epoch
    initial_epoch = 0
    if get_model_type(config) in os.listdir("checkpoints"):
        with open(f"logs/csv/{get_model_type(config)}.csv") as csvfile:
            last_line = csvfile.readlines()[-1]
            initial_epoch = int(last_line.split(",")[0]) + 1

    # predict_batch(val_data.get_patch_indices(), loader, model, config, "foo")

    # Train Model
    print(f"EPOCH: {initial_epoch}")
    if config["train"]:
        model.fit(train_data, epochs=get_epochs(config)+initial_epoch, verbose=1, callbacks=callbacks, validation_data=val_data, initial_epoch=initial_epoch)
        # model.fit(train_data, epochs=10, verbose=1, callbacks=callbacks, validation_data=val_data, initial_epoch=initial_epoch)
    
    if config["test"]:
        test_data.predict_batch(model, "test")


if __name__ == '__main__':
    # Set Visible GPU
    args = sys.argv
    GPU = int(args[1]) if len(args) > 1 and args[1].isdigit() else 0
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{GPU}"

    # Use Mixed Precision
    mixed_precision.set_global_policy('mixed_float16')

    # Run Script
    main()
