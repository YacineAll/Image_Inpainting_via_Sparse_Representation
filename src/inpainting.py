# !/usr/bin/env python3
# -*- coding=utf-8 -*-

import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt

from io import BytesIO
from PIL import Image

from sklearn.preprocessing import MinMaxScaler


import urllib
import requests

from sklearn.linear_model import Lasso




def is_out_of_bounds(pixels, x, y):
    return not (0 <= x < pixels.shape[0] and 0 <= y < pixels.shape[1])


def noise(img, prc):
    l, w, d = img.shape
    new_img = img.copy().ravel()
    new_img[np.random.choice([False, True], l*w*d, p = [1-prc, prc])] = -100
    return new_img.reshape((l, w, d))


def delete_rect(img, i, j, height, width):
    new_img = img.copy()
    new_img[i:i+height, j:j+width] = np.ones((height, width, 3))*-100
    return new_img


def change_color(pixels, color_src: str, color_dest: str):

    if color_src == color_dest:
        return pixels

    l, w, d = pixels.shape
    new_pixels = pixels.copy()

    if color_dest == "HSV":
        new_pixels = colors.rgb_to_hsv(new_pixels)
    if color_dest == "RGB":

        new_pixels = colors.hsv_to_rgb(new_pixels).astype(int)

    return new_pixels


def read_im(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.asarray(img)


def show_im(arr):
    arr3 = arr.copy()
    arr3[arr3 == -100] = 0
    plt.grid(False)
    plt.imshow(arr3)
    plt.show()
    
class Picture(object):
    """docstring for Picture."""

    def __init__(self, pixels: np.ndarray, size: int, step: int = None):
        super(Picture, self).__init__()
        self.pixels = pixels
        self.size = size
        if step is None:
            self.step = size

    def get_patch(self, x, y):
        if not self.is_out_of_bounds_patch(x, y):
            return self.pixels[x - (self.size // 2):x + (self.size // 2) + 1, y - (self.size // 2): y + (self.size // 2) + 1]
        else:
            patch = []
            for index_x in range(x - (self.size // 2), x + (self.size // 2) + 1):
                new_line = []
                for index_y in range(y - (self.size // 2), y + (self.size // 2) + 1):
                    if not is_out_of_bounds(self.pixels, index_x, index_y):
                        new_line.append(self.pixels[index_x, index_y])
                    else:
                        new_line.append(np.ones(3)*-1000)
                patch.append(np.array(new_line))
            return np.array(patch)

    def is_out_of_bounds_patch(self, x, y):
        return (x - (self.size // 2) <= 0) or \
            (x + (self.size // 2) + 1 < self.pixels.shape[0]) or \
            (y - (self.size // 2) <= 0) or \
            (y + (self.size // 2) + 1 < self.pixels.shape[1])

    def iter_patch(self, x: int, y: int):
        for index_y in range(y - (self.size // 2), y + (self.size // 2) + 1):
            for index_x in range(x - (self.size // 2), x + (self.size // 2) + 1):
                yield index_x, index_y

    def iter_patch_empty(self, x: int, y: int):
        for index_x, index_y in iter_patch(x, y, size):
            if not is_out_of_bounds(self.pixels, index_x, index_y):
                if all(self.pixels[index_x, index_y] == -100):
                    yield index_x, index_y

    def get_dictionary(self):
        result = []
        for i in range(0, self.pixels.shape[0], self.step):
            for j in range(0, self.pixels.shape[1], self.step):
                if not is_out_of_bounds(self.pixels, i, j):
                    patch = self.get_patch(i, j)
                    if(np.all(patch != -100) and np.all(patch != -1000)):
                        result.append(patch)

        return np.array(result)


class Inpainting(object):
    """docstring for Inpainting."""

    def __init__(self, pixels: np.ndarray,patch_size: int = 5, step: int = None, alpha: float = 0.001, tolerance: float = 1e-5, max_iterations: int = 1e+3):
        super(Inpainting, self).__init__()
        self.patch_size = patch_size
        self.step = step
        self.alpha = alpha
        self.pixels = pixels
        self.patch_object = (pixels, patch_size, step)
        
        classifiers_kwaargs = {"alpha": alpha, "copy_X": True, "fit_intercept": True, "max_iter": max_iterations,
                               "normalize": False, "positive": False, "precompute": False, "random_state": None,
                               "selection": 'cyclic', "tol": tolerance, "warm_start": False}
        self.classifier_H = Lasso(**classifiers_kwaargs)
        self.classifier_S = Lasso(**classifiers_kwaargs)
        self.classifier_V = Lasso(**classifiers_kwaargs)
        
        
    
    def fit(self, dictionary, patch):
        X_H, X_S, X_V, y_H, y_S, y_V = preprocess_training_data(
            patch, dictionary, self.patch_size)
        self.classifier_H.fit(X_H, y_H)
        self.classifier_S.fit(X_S, y_S)
        self.classifier_V.fit(X_V, y_V)

    def preprocess_training_data(self, patch, dictionary):
        X_H, X_S, X_V, y_H, y_S, y_V = [
        ], [], [], [], [], []
        for x in range(self.patch_size):
            for y in range(self.patch_size):
                if np.all(patch[x, y] != -100) and np.all(patch[x, y] != -1000):
                    X_H.append(dictionary[:, x, y, 0])
                    X_S.append(dictionary[:, x, y, 1])
                    X_V.append(dictionary[:, x, y, 2])
                    y_H.append(patch[x, y, 0])
                    y_S.append(patch[x, y, 1])
                    y_V.append(patch[x, y, 2])
        return np.array(X_H), np.array(X_S), np.array(X_V), np.array(y_H), np.array(y_S), np.array(y_V)

    def predict(self, x, y, dictionary):
        H = self.classifier_H.predict(dictionary[:, x, y, 0].reshape(1, -1))
        S = self.classifier_S.predict(dictionary[:, x, y, 1].reshape(1, -1))
        V = self.classifier_V.predict(dictionary[:, x, y, 2].reshape(1, -1))
        return np.hstack((H, S, V))

    def inpaint(self, patch_size, step):
        new_img = self.pixels.copy()
        dictionary = self.patch_object.get_dictionary()
        missing_pixels_x, missing_pixels_y, *_ = np.where(new_img == -100)
        for i, j in zip(missing_pixels_x, missing_pixels_y):
            next_patch = self.patch_object.get_patch()
            
            fit(dictionary, next_patch,patch_size)
            for x, y in self.patch_object.iter_patch_empty( i, j):
                next_pixel_value = predict(x - i + (patch_size // 2),
                                              y - j + (patch_size // 2),
                                              dictionary)
                new_img[x, y] = next_pixel_value
            missing_pixels_x, missing_pixels_y, *_ = np.where(new_img == -100)
        
        return new_img



