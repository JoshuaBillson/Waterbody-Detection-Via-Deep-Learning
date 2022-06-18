import os
import sys
import json
from typing import Dict, Any
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from backend.metrics import MIOU 
from backend.data_loader import DataLoader, load_dataset
from generate_patches import generate_patches
from models import get_model
from models.utils import evaluate_model
from backend.config import get_epochs, get_model_type, get_timestamp, get_learning_rate, get_timestamp_directory, get_num_experiments
from models.losses import JaccardBCELoss, DiceBCELoss, JaccardLoss, focal_tversky, WeightedBCE, TanimotoLoss, TanimotoBCELoss, TanimotoLossWithComplement, TanimotoWithComplementBCELoss, ScheduledTanimoto
from backend.callbacks import get_callbacks, create_callback_dirs


def get_loss_function(config: Dict[str, Any]):
    """
    Get the loss function from the project config
    :param config: A dictionary storing the project configuration; typically loaded from an external file
    :returns: The loss function to use during model optimization
    """
    losses = {
        "dice_bce": DiceBCELoss,
        "jaccard": JaccardLoss,
        "jaccard_bce": JaccardBCELoss,
        "tanimoto": TanimotoLoss,
        "tanimoto_with_complement": TanimotoLossWithComplement,
        "modified_tanimoto_with_bce": TanimotoWithComplementBCELoss,
        "tanimoto_bce": TanimotoBCELoss,
        "scheduled_tanimoto": ScheduledTanimoto(config),
        "weighted_bce": WeightedBCE(0.1, 0.9),
        "focal_tversky": focal_tversky,
        }
    return losses[config["hyperparameters"]["loss"]]


def main():
    # Get Project Configuration
    with open('config.json') as f:
        config = json.loads(f.read())

    # Create Data Loader
    loader = DataLoader(timestamp=get_timestamp(config))

    # Generate Patches
    if "patches" not in os.listdir(f"data/{get_timestamp_directory(config)}") or config["generate_patches"]:
        generate_patches(loader=loader, config=config)

    # Load Dataset
    train_data, val_data, test_data = load_dataset(loader, config)

    # Create Callback Directories
    create_callback_dirs()

    # Train A Model For Each Experiment Specified In The Project Config
    trained_models = []
    for _ in range(get_num_experiments(config)):

        # Create Model
        model = get_model(config)
        model.summary()
        model.compile(loss=get_loss_function(config), optimizer=Adam(learning_rate=get_learning_rate(config)), metrics=[MIOU(), Precision(), Recall()])

        # Get Callbacks
        callbacks = get_callbacks(config, val_data, model)

        # If Model Is Loaded From Checkpoint, Find The Last Epoch
        initial_epoch = 0
        if get_model_type(config) in os.listdir("checkpoints"):
            with open(f"logs/csv/{get_model_type(config)}.csv") as csvfile:
                last_line = csvfile.readlines()[-1]
                initial_epoch = int(last_line.split(",")[0]) + 1

        # Train Model
        if config["train"]:
            model.fit(train_data, epochs=get_epochs(config)+initial_epoch, verbose=1, callbacks=callbacks, validation_data=val_data, initial_epoch=initial_epoch)
            trained_models.append(model)
        
    # Evaluate Performance Of All Models
    if config["test"]:
        results = [test_data.predict_batch(m, "test") for m in trained_models]
        print("\nSUMMARY OF MODEL AVERAGES")
        for result in zip(trained_models[0].metrics_names, *results):
            metric = result[0]
            average_value = sum(result[1:]) / len(result[1:])
            print(f"Model Average For {metric}:", average_value)


if __name__ == '__main__':
    # Set Visible GPU
    args = sys.argv
    GPUS = args[1:] if len(args) > 1 else ["0"] 
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{','.join(GPUS)}"

    # Use Mixed Precision
    mixed_precision.set_global_policy('mixed_float16')

    # Run Script
    main()
