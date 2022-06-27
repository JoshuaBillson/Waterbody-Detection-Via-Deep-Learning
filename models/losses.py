from math import ceil
from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import flatten, sum, dot, pow, mean, ones_like
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from backend.config import get_epochs, get_batch_size
from scipy.ndimage import distance_transform_edt as distance


class LossScheduler:
    def __init__(self, start_epoch, end_epoch, steps_per_epoch) -> None:
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.steps_per_epoch = steps_per_epoch
        self.current_step = steps_per_epoch * start_epoch

    def update(self):
        self.current_step += 1

    def get_weights(self) -> Tuple[float, float]:
        current_epoch = self.current_step // self.steps_per_epoch
        w_ce = min(current_epoch / self.end_epoch, 1.0)
        return w_ce, (1.0 - w_ce)


def FocalLoss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
	return focal_loss_fixed


def TanimotoLoss(truth, prediction, smooth=1e-6):
    truth, prediction = flatten(truth), flatten(prediction)
    intersection = sum(truth * prediction)
    denominator = sum(pow(truth, 2) + pow(prediction, 2)) - intersection
    return 1 - ((intersection + smooth) / (denominator + smooth))


def TanimotoLossWithComplement(truth, prediction, smooth=1e-6):
    tanimoto = TanimotoLoss(truth, prediction, smooth=smooth)
    complement = TanimotoLoss(tf.ones_like(truth) - truth, tf.ones_like(prediction) - prediction, smooth=smooth)
    return (tanimoto + complement) / 2

def TanimotoWithComplementBCELoss(truth, prediction, smooth=1e-6):
    tanimoto = TanimotoLossWithComplement(truth, prediction, smooth=smooth)
    BCE = WeightedBCE(0.1, 0.9)(truth, prediction)
    return (0.5 * tanimoto) + (0.5 * BCE)


def ScheduledTanimoto(config, smooth=1e-6):
    scheduler = LossScheduler(0, get_epochs(config), ceil(2700.0 / get_batch_size(config)))
    def scheduled_tanimoto(truth, prediction):
        scheduler.update()
        tanimoto = TanimotoLoss(truth, prediction, smooth=smooth)
        BCE = WeightedBCE(0.1, 0.9)(truth, prediction)
        w_ce, w_t = scheduler.get_weights()
        return (w_t * tanimoto) + (w_ce * BCE)
    return scheduled_tanimoto

def TanimotoBCELoss(truth, prediction, smooth=1e-6):
    tanimoto = TanimotoLoss(truth, prediction, smooth=smooth)
    BCE = WeightedBCE(0.1, 0.9)(truth, prediction)
    return (0.5 * tanimoto) + (0.5 * BCE)


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


def WeightedBCE(w0, w1):
    def loss(truth, prediction):
        truth, prediction = K.clip(flatten(truth), 1e-6, 1 - 1e-6), K.clip(flatten(prediction), 1e-6, 1 - 1e-6)
        entropies = (w1 * K.log(prediction) * truth) + (w0 * K.log(tf.ones_like(prediction) - prediction) * (tf.ones_like(truth) - truth))
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

def JaccardBCELoss2(smooth=1e-6, alpha=0.5, gamma=2., alpha2=.25):
    focal_loss = FocalLoss(alpha=alpha2, gamma=gamma)
    def f(targets, inputs):
        BCE = focal_loss(flatten(targets), flatten(inputs))
        jaccard = JaccardLoss(targets, inputs, smooth=smooth)
        return (alpha * BCE) + ((1 - alpha) * jaccard)
    return f


def tversky(y_true, y_pred, smooth=1e-6):
    y_true_pos = flatten(y_true)
    y_pred_pos = flatten(y_pred)
    true_pos = sum(y_true_pos * y_pred_pos)
    false_neg = sum(y_true_pos * (ones_like(y_pred_pos) - y_pred_pos))
    false_pos = sum((ones_like(y_true_pos) - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return 1 - ((true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth))


def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return pow((1-pt_1), gamma)
