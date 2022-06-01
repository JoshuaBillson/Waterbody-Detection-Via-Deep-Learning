from typing import Dict, Any, List, Tuple


def get_bands(config: Dict[str, Any]) -> List[str]:
    """
    Get the input bands from the project config
    :param config: A dictionary storing the project configuration; typically loaded from an external file
    :returns: The list of bands we want to use
    """
    return config["hyperparameters"]["bands"]

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


def get_optimizer(config: Dict[str, Any]) -> str:
    """
    Get the optimizer type from the project config
    :param config: A dictionary storing the project configuration; typically loaded from an external file
    :returns: The type of optimizer to be used by the training loop
    """
    return config["hyperparameters"]["optimizer"]


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
    bands = get_bands(config)
    backbone = get_backbone(config)
    input_channels = len(bands) + 2 if "RGB" in bands else len(bands)
    model_type = get_model_type(config)
    assert not (backbone is not None and input_channels != 3 and model_type not in ["fpn", "pspnet", "linknet"]), "Error: Cannot Use Backbone For Input Channels Other Than 3!"
    return input_channels, backbone
