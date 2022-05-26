from keras.backend import flatten, sum, dot
from keras.losses import binary_crossentropy


def DiceBCELoss(targets, inputs, smooth=1e-6, alpha=0.5):
    """
    Adapted From The Following:
        Source: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
        Author: https://www.kaggle.com/bigironsphere

    :param targets: The ground-truth
    :param inputs: The predictions made by our model
    :param smooth: A constant to prevent division by zero
    :param alpha: A float in the range (0, 1) which denotes the relative weight of the BCE loss with respect to the Dice loss.
    :return:The loss on the given sample.
    """
    inputs = flatten(inputs)
    targets = flatten(targets)
    BCE = binary_crossentropy(targets, inputs)
    # return BCE
    intersection = sum(targets * inputs)
    dice_loss = 1 - (2 * intersection + smooth) / (sum(targets) + sum(inputs) + smooth)
    # return dice_loss
    return (alpha * BCE) + ((1 - alpha) * dice_loss)


def JaccardLoss(targets, inputs, smooth=1e-6):
    """
    Adapted From The Following:
        Source: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
        Author: https://www.kaggle.com/bigironsphere

    :param targets: The ground-truth
    :param inputs: The predictions made by our model
    :param smooth: A constant to prevent division by zero
    :return:The loss on the given sample
    """
    inputs = flatten(inputs)
    targets = flatten(targets)
    intersection = sum(targets * inputs)
    total = sum(targets) + sum(inputs)
    union = total - intersection
    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU


def JaccardBCELoss(targets, inputs, smooth=1e-6, alpha=0.5):
    """
    Adapted From The Following:
        Source: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
        Author: https://www.kaggle.com/bigironsphere

    :param targets: The ground-truth
    :param inputs: The predictions made by our model
    :param smooth: A constant to prevent division by zero
    :param alpha: A float in the range (0, 1) which denotes the relative weight of the BCE loss with respect to the Jaccard loss.
    :return:The loss on the given sample
    """
    BCE = binary_crossentropy(flatten(targets), flatten(inputs))
    jaccard = JaccardLoss(targets, inputs, smooth=smooth)
    return (alpha * BCE) + ((1 - alpha) * jaccard)
