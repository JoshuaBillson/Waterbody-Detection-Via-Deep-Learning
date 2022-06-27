from typing import Dict, Any, List, Tuple


def get_timestamp_directory(config: Dict[str, Any]) -> str:
    """
    Get the input bands from the project config
    :param config: A dictionary storing the project configuration; typically loaded from an external file
    :returns: The list of bands we want to use
    """
    folders = {1: "2018.04", 2: "2018.12", 3: "2019.02"}
    return folders[get_timestamp(config)]


def get_bands(config: Dict[str, Any]) -> List[str]:
    """
    Get the input bands from the project config
    :param config: A dictionary storing the project configuration; typically loaded from an external file
    :returns: The list of bands we want to use
    """
    return [band for band in ("RGB", "NIR", "SWIR") if band in config["hyperparameters"]["bands"]]


def get_batch_size(config: Dict[str, Any]) -> int:
    """
    Get the batch size from the project config
    :param config: A dictionary storing the project configuration; typically loaded from an external file
    :returns: Batch size to be used by the training loop
    """
    return config["hyperparameters"]["batch_size"]


def get_patch_size(config: Dict[str, Any]) -> int:
    """
    Get the patch size from the project config
    :param config: A dictionary storing the project configuration; typically loaded from an external file
    :returns: Patch size to be generated by the script
    """
    return config["patch_size"]


def get_input_channels(config: Dict[str, Any]) -> int:
    """
    Get the number of input channels based on the inputs bands from the project config
    :param config: A dictionary storing the project configuration; typically loaded from an external file
    :returns: The number of input channels that will be fed into a model
    """
    channels = {
        "RGB": 3,
        "NIR": 1,
        "SWIR": 1,
        "RGB+NIR": 4,
        "RGB+SWIR": 4,
        "RGB+NIR+SWIR": {"naive": 5, "depthwise": 128, "3D": 125, "paper": 128, "grayscale": 3}[get_fusion_head(config)],
    }
    return channels["+".join(get_bands(config))]


def get_waterbody_transfer(config: Dict[str, Any]) -> bool:
    """
    Get the configuration for applying waterbody transfer
    :param config: A dictionary storing the project configuration; typically loaded from an external file
    :returns: Boolean indicating whether or not we want to apply waterbody transfer
    """
    return config["hyperparameters"]["apply_transfer"]


def get_experiment_tag(config: Dict[str, Any]) -> str:
    """
    Get the experiment tag
    :param config: A dictionary storing the project configuration; typically loaded from an external file
    :returns: Experiment tag we can use to associate models with experiments
    """
    return config["experiment_tag"]


def get_model_type(config: Dict[str, Any]) -> int:
    """
    Get the model type from the project config
    :param config: A dictionary storing the project configuration; typically loaded from an external file
    :returns: The type of model we want to train (U-Net, V-Net, etc.)
    """
    return config["hyperparameters"]["model"]


def get_epochs(config: Dict[str, Any]) -> int:
    """
    Get the number of epochs from the project config
    :param config: A dictionary storing the project configuration; typically loaded from an external file
    :returns: The number of epochs to be used by the training loop
    """
    return config["hyperparameters"]["epochs"]


def get_timestamp(config: Dict[str, Any]) -> int:
    """
    Get the timestamp from the project config
    :param config: A dictionary storing the project configuration; typically loaded from an external file
    :returns: The timestamp for which we want to fetch data
    """
    return config["timestamp"]


def get_backbone(config: Dict[str, Any]) -> str:
    """
    Get the backbone for the model we want to build from the project config
    :param config: A dictionary storing the project configuration; typically loaded from an external file
    :returns: The backbone (ResNet151, VGG16, etc.) to be used by the training loop
    """
    return config["hyperparameters"]["backbone"]


def get_learning_rate(config: Dict[str, Any]) -> float:
    """
    Get the learning rate from the project config
    :param config: A dictionary storing the project configuration; typically loaded from an external file
    :returns: The learning rate to be used by the training loop
    """
    return config["hyperparameters"]["learning_rate"]


def get_create_logs(config: Dict[str, Any]) -> bool:
    """
    Determines if our training loop should generate logs based on the project config
    :param config: A dictionary storing the project configuration; typically loaded from an external file
    :returns: True if we want to create logs, false otherwise
    """
    return config["create_logs"]


def get_model_config(config: Dict[str, Any]) -> Tuple[int, str]:
    """
    Get the number of input channels and backbone from the project config
    :param config: A dictionary storing the project configuration; typically loaded from an external file
    :returns: The number of input channels our base model will receive and the backbone returned as (input_channels, backbone)
    """
    return get_input_channels(config), get_backbone(config)


def get_fusion_head(config: Dict[str, Any]) -> str:
    """
    Get the type of fusion head for combining multi-spectral features from the project config
    :param config: A dictionary storing the project configuration; typically loaded from an external file
    :returns: The name of the type of fusion head we want our model to use
    """
    return config["hyperparameters"]["fusion_head"]


def get_num_experiments(config: Dict[str, Any]) -> int:
    """
    Get the number of experiments to run from the project config
    :param config: A dictionary storing the project configuration; typically loaded from an external file
    :returns: The number of experiments to run
    """
    return config["experiments"]


def get_water_threshold(config: Dict[str, Any]) -> float:
    """
    Get the water threshold that patches must meet to avoid being discarded
    :param config: A dictionary storing the project configuration; typically loaded from an external file
    :returns: The water threshold as a percentage that must be met by a patch to avoid being discarded
    """
    return config["hyperparameters"]["water_threshold"]


def get_random_subsample(config: Dict[str, Any]) -> bool:
    """
    Get the setting for whether or not the data pipeline should sub-sample 512x512 patches
    :param config: A dictionary storing the project configuration; typically loaded from an external file
    :returns: Whether or not to randomly sub-sample patches
    """
    return config["hyperparameters"]["random_subsample"]
