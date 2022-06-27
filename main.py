import os
import sys
import json
from typing import Dict, Any
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from backend.metrics import MIOU 
from backend.pipeline import load_dataset
from generate_patches import generate_patches
from models import get_model
from models.utils import evaluate_model
from backend.config import get_epochs, get_model_type, get_timestamp, get_learning_rate, get_timestamp_directory, get_num_experiments
from models.losses import JaccardBCELoss, DiceBCELoss, JaccardLoss, focal_tversky, WeightedBCE, TanimotoLoss, TanimotoBCELoss, TanimotoLossWithComplement, TanimotoWithComplementBCELoss, ScheduledTanimoto, tversky
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
        "tversky": tversky,
        }
    return losses[config["hyperparameters"]["loss"]]


def main():
    # Get Project Configuration
    with open('config.json') as f:
        config = json.loads(f.read())

    # Generate Patches
    if "tiles" not in os.listdir(f"data/{get_timestamp_directory(config)}"):
        generate_patches(config=config)

    # Load Dataset
    train_data, val_data, test_data = load_dataset(config)

    # Create Callback Directories
    create_callback_dirs()

    # Train A Model For Each Experiment Specified In The Project Config
    results, num_experiments = [], get_num_experiments(config)
    assert num_experiments > 0, "Error: Value for 'experiments' must be greater than 0!"
    for _ in range(num_experiments):

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

        # Evaluate Model On Test Set
        if config["test"]:
            results.append(test_data.predict_batch(model, "test"))
        
        # Save Experiment Configuration For Future Reference
        if "experiments" not in os.listdir():
            os.mkdir("experiments")
        with open(f"experiments/{model.name}.json", 'w') as config_file:
            config_file.write(json.dumps(config, indent=2))
        
    # Evaluate Performance Of All Models
    if config["test"]:
        print("\nSUMMARY OF MODEL AVERAGES")
        for result in zip(model.metrics_names, *results):
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
