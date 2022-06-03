import tensorflow as tf
from tensorflow.keras.backend import flatten, sum, dot
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K


def FocalLoss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
	return focal_loss_fixed


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

def JaccardBCELoss2(smooth=1e-6, alpha=0.5, gamma=2., alpha2=.25):
    focal_loss = FocalLoss(alpha=alpha2, gamma=gamma)
    def f(targets, inputs):
        BCE = focal_loss(flatten(targets), flatten(inputs))
        jaccard = JaccardLoss(targets, inputs, smooth=smooth)
        return (alpha * BCE) + ((1 - alpha) * jaccard)
    return f