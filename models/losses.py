from math import ceil
from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import flatten, sum, dot, pow, mean, ones_like, log
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from backend.config import get_epochs, get_batch_size
from scipy.ndimage import distance_transform_edt as distance


def DiceLoss(targets, inputs, smooth=1e-6):
    """
    Adapted From The Following:
        Source: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
        Author: https://www.kaggle.com/bigironsphere

    :param targets: The ground-truth
    :param inputs: The predictions made by our model
    :param smooth: A constant to prevent division by zero
    :return:The loss on the given sample.
    """
    inputs, targets = flatten(inputs), flatten(targets)
    intersection = sum(targets * inputs)
    total = sum(targets) + sum(inputs)
    IoU = (2 * intersection + smooth) / (total + smooth)
    return 1 - IoU


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


def WeightedBCE(w0, w1):
    def loss(truth, prediction):
        truth, prediction = K.clip(flatten(truth), 1e-6, 1 - 1e-6), K.clip(flatten(prediction), K.epsilon(), 1 - K.epsilon())
        truth_complement, prediction_complement = tf.ones_like(truth) - truth, tf.ones_like(prediction) - prediction
        entropies = (w1 * K.log(prediction) * truth) + (w0 * K.log(prediction_complement) * (truth_complement))
        return K.mean(-1.0 * entropies)
    return loss


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
    BCE = binary_crossentropy(flatten(targets), flatten(inputs))
    dice = DiceLoss(targets, inputs, smooth=smooth)
    return (alpha * BCE) + ((1 - alpha) * dice)


def JaccardWeightedBCELoss(w0, w1, smooth=1e-6, alpha=0.5):
    BCE = WeightedBCE(w0, w1)
    def jaccard_weighted_bce_loss(targets, inputs):
        bce = BCE(flatten(targets), flatten(inputs))
        jaccard = JaccardLoss(targets, inputs, smooth=smooth)
        return (alpha * bce) + ((1 - alpha) * jaccard)
    return jaccard_weighted_bce_loss


def Tversky(smooth=1e-6, alpha=0.7):
    def tversky(y_true, y_pred):
        y_true_pos = flatten(y_true)
        y_pred_pos = flatten(y_pred)
        y_true_neg = ones_like(y_true_pos) - y_true_pos
        y_pred_neg = ones_like(y_pred_pos) - y_pred_pos
        true_pos = sum(y_true_pos * y_pred_pos)
        false_neg = sum(y_true_pos * y_pred_neg)
        false_pos = sum(y_true_neg * y_pred_pos)
        tversky_index = (true_pos + smooth) / (true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
        return 1 - tversky_index
    return tversky


def Focal_Tversky(smooth=1e-6, alpha=0.7, gamma=1.33):
    tversky = Tversky(smooth=smooth, alpha=alpha)
    def focal_tversky(y_true, y_pred):
        tversky_loss = tversky(y_true, y_pred)
        return pow(tversky_loss, (1.0 / gamma))
    return focal_tversky


def FocalLoss(alpha=0.25, gamma=2.0):
    def focal_loss(y_true, y_pred):
        y_true, y_pred = flatten(y_true), K.clip(flatten(y_pred), K.epsilon(), 1 - K.epsilon())
        y_true_complement, y_pred_complement = ones_like(y_true) - y_true, ones_like(y_pred) - y_pred
        p = (y_true * y_pred) + (y_true_complement * y_pred_complement)
        focal_loss_total = -alpha * (pow(ones_like(p) - p, gamma) * K.log(p))
        return K.mean(focal_loss_total)
    return focal_loss