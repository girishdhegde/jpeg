import os
import sys
import glob
import json
import time
import copy

import numpy as np


__author__ = "__Girish_Hegde__"


def to_image(img, norm=False, save=None, show=True, delay=0, rgb=True, bg=0):
    """ Function to show/save image
        author: girish d. hegde
        contact: girish.dhc@gmail.com

    Args:
        img (np.ndarray): [h, w, ch] image(grayscale/rgb)
        norm (bool, optional): min-max normalize image. Defaults to False.
        save (str, optional): path to save image. Defaults to None.
        show (bool, optional): show image. Defaults to True.
        delay (int, optional): cv2 window delay. Defaults to 0.

    Returns:,
        (np.ndarray): [h, w, ch] - image.
    """
    if rgb:
        img = img[..., ::-1]
    if norm:
        img = (img - img.min())/(img.max() - img.min())
    if save is not None:
        if img.max() <= 1:
            img *=255
        cv2.imwrite(save, img.astype(np.uint8))
    if show:
        cv2.imshow('img', img)
        cv2.waitKey(delay)
        cv2.destroyAllWindows()
    return img