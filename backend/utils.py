import math
import numpy as np
from cv2 import normalize, NORM_MINMAX, CV_8U, LUT, cvtColor, COLOR_BGR2GRAY


def adjust_rgb(rgb_img: np.ndarray, gamma: float = 0.45) -> np.ndarray:
    return adjust_gamma(normalize(rgb_img, None, 0, 255, NORM_MINMAX, dtype=CV_8U), gamma)


def compute_gamma_correction(img):
    # convert img to gray
    gray = cvtColor(img, COLOR_BGR2GRAY)

    # compute gamma = log(mid*255)/log(mean)
    mid = 0.5
    mean = np.mean(gray)
    gamma = math.log(mid*255)/math.log(mean)
    return gamma


def adjust_gamma(image: np.ndarray, gamma: float = 1.0):
    """
    Perform gamma correction on the provided image
    :param image: The image to which we want to apply gamme correction
    :param gamma: The ammount of gamma correction to apply
    :returns: The gamma corrected image
    """

    gamma = compute_gamma_correction(image)
    invGamma = (1 / gamma) + 0.2
    table = np.array([((i / 255.0) * invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return LUT(image, table)
