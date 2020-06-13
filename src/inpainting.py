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
from tqdm import tqdm

import random


def add_noise(img: np.ndarray, threshold: float = 0.05):
    """ Add random noise to the picture.
    :param threshold: Chance of each pixel to be noisyfied.
    """
    pixels = img.copy()
    height, width, d = pixels.shape

    for x in range(height):
        for y in range(width):
            pixels[x, y] = np.ones(
                3)*-100 if random.random() < threshold else pixels[x, y]

    return pixels


def is_out_of_bounds(pixels, x, y):
    return not (0 <= x < pixels.shape[0] and 0 <= y < pixels.shape[1])


def noise(img, prc):
    l, w, d = img.shape
    new_img = img.copy().ravel()
    new_img[np.random.choice(
        [False, True], l*w, p=[1-prc, prc]):] = np.ones(3)*-100
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


class Patch(object):
    """docstring for Patch."""

    def __init__(self, pixels: np.ndarray, size: int, step: int = None):
        super(Patch, self).__init__()
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
        for index_x, index_y in self.iter_patch(x, y):
            if not is_out_of_bounds(self.pixels, index_x, index_y):
                if any(self.pixels[index_x, index_y] == -100):
                    yield index_x, index_y

    def get_dictionary(self):
        shape_p = (self.size,self.size,3)
        result = []
        for i in range(0, self.pixels.shape[0], self.step):
            for j in range(0, self.pixels.shape[1], self.step):
                if not is_out_of_bounds(self.pixels, i, j):
                    patch = self.get_patch(i, j)
                    if(np.all(patch != -100) and np.all(patch != -1000) and patch.shape == shape_p):
                        result.append(patch)

        return np.array(result)

    def get_next_patch(self):
        missing_pixels_x, missing_pixels_y, * \
            _ = np.where(self.pixels[:, :, 0] == -100)
        return zip(missing_pixels_x, missing_pixels_y)


class Inpainting(object):
    """docstring for Inpainting."""

    def __init__(self, pixels: np.ndarray, patch_size: int = 5, step: int = None, alpha: float = 0.001, tolerance: float = 1e-5, max_iterations: int = 1e+3):
        super(Inpainting, self).__init__()
        self.patch_size = patch_size
        self.step = step
        self.alpha = alpha
        self.pixels = pixels
        self.patch_object = Patch(pixels, patch_size, step)

        classifiers_kwaargs = {"alpha": alpha, "copy_X": True, "fit_intercept": True, "max_iter": max_iterations,
                               "normalize": False, "positive": False, "precompute": False, "random_state": None,
                               "selection": 'cyclic', "tol": tolerance, "warm_start": False}
        self.classifier_H = Lasso(**classifiers_kwaargs)
        self.classifier_S = Lasso(**classifiers_kwaargs)
        self.classifier_V = Lasso(**classifiers_kwaargs)

    def preprocess_training_data(self, patch):
        boolean = False
        X_H = []
        X_S = []
        X_V = []
        y_H = []
        y_S = []
        y_V = []
        for x in range(self.patch_size):
            for y in range(self.patch_size):
                if np.all(patch[x, y] != -100) and np.all(patch[x, y] != -1000):
                    if boolean:
                        X_H = np.hstack((X_H, self.dictionary[:, x, y, 0]))
                        X_S = np.hstack((X_S, self.dictionary[:, x, y, 1]))
                        X_V = np.hstack((X_V, self.dictionary[:, x, y, 2]))
                        y_H = np.hstack((y_H, patch[x, y, 0]))
                        y_S = np.hstack((y_S, patch[x, y, 1]))
                        y_V = np.hstack((y_V, patch[x, y, 2]))
                    else:
                        X_H = self.dictionary[:, x, y, 0]
                        X_S = self.dictionary[:, x, y, 1]
                        X_V = self.dictionary[:, x, y, 2]
                        y_H = patch[x, y, 0]
                        y_S = patch[x, y, 1]
                        y_V = patch[x, y, 2]
                        boolean = True

        return X_H, X_S, X_V, y_H, y_S, y_V
    
    
    def preprocess_training_data2(self, patch):
        datax_hue, datax_saturation, datax_value, datay_hue, datay_saturation, datay_value = [], [], [], [], [], []

        # Iterate trough each pixels of the patch to inpaint
        for x in range(self.patch_size):
            for y in range(self.patch_size):
                # Ignore missing-pixels in the patch, we cannot learn from them
                if np.all(patch[x, y] != -100) and np.all(patch[x, y] != -1000):
                    datax_hue.append(self.dictionary[:, x, y, 0])
                    datax_saturation.append(self.dictionary[:, x, y, 1])
                    datax_value.append(self.dictionary[:, x, y, 2])
                    datay_hue.append(patch[x, y, 0])
                    datay_saturation.append(patch[x, y, 1])
                    datay_value.append(patch[x, y, 2])

        return np.array(datax_hue), np.array(datax_saturation), \
            np.array(datax_value), np.array(datay_hue), \
            np.array(datay_saturation), np.array(datay_value)
    
    
    def X(self, patch):
        X_h, X_s, X_v = [], [], []
        datay_hue, datay_saturation, datay_value = [], [], []
        
        boolean = False
        for x in range(self.patch_size):
            for y in range(self.patch_size):
                try:
                    if np.all(patch[x, y] != -100) and np.all(patch[x, y] != -1000):
                        datay_hue.append(patch[x, y, 0])
                        datay_saturation.append(patch[x, y, 1])
                        datay_value.append(patch[x, y, 2])

                        if(boolean):                
                            X_h = np.vstack((X_h,self.dictionary[:, x, y, 0]))
                            X_s = np.vstack((X_s,self.dictionary[:, x, y, 1])) 
                            X_v = np.vstack((X_v,self.dictionary[:, x, y, 2])) 
                        else:
                            X_h = self.dictionary[:, x, y, 0]
                            X_s = self.dictionary[:, x, y, 1]
                            X_v = self.dictionary[:, x, y, 2]
                            boolean = True
                except IndexError:
                    pass
                
        return X_h, X_s, X_v, np.array(datay_hue), np.array(datay_saturation), np.array(datay_value)

    def predict(self, x, y):
        H = self.classifier_H.predict(
            self.dictionary[:, x, y, 0].reshape(1, -1))
        S = self.classifier_S.predict(
            self.dictionary[:, x, y, 1].reshape(1, -1))
        V = self.classifier_V.predict(
            self.dictionary[:, x, y, 2].reshape(1, -1))
        return np.hstack((H, S, V))

    def inpaint(self):

        out = self.pixels.copy()

        X_h, X_s, X_v, y_h, y_s, y_v = self.get_training_data()
        print(f'Donnees d\'entrainement recolté...')

        self.classifier_H.fit(X_h, y_h)
        print(f'Model 1 entrainé...')
        self.classifier_S.fit(X_s, y_s)
        print(f'Model 2 entrainé...')
        self.classifier_V.fit(X_v, y_v)
        print(f'Model 3 entrainé...')

        print(f'Les models sont entranés...')

        print(f'Predictions...')

    def predictions(self):
        out = self.pixels.copy()
        for next_pixel in self.patch_object.get_next_patch():
            for x, y in self.patch_object.iter_patch_empty(*next_pixel):
                next_pixel_value = self.predict(x - next_pixel[0] + (self.patch_size // 2),
                                                y - next_pixel[1] +
                                                (self.patch_size // 2))
                out[x, y] = next_pixel_value
        return out

    def get_training_data(self):
        self.dictionary = self.patch_object.get_dictionary()
        X_h, X_s, X_v, y_h, y_s, y_v = [], [], [], [], [], []
        
        
        boolean = False
        next_patchs = self.patch_object.get_next_patch()
        for i, j in tqdm(next_patchs):
            patch = self.patch_object.get_patch(i, j)
            if(boolean):
                
                X_H, X_S, X_V, y_H, y_S, y_V = self.X(patch)
                
                
                X_h = np.vstack((X_h, X_H))
                X_s = np.vstack((X_s, X_S))
                X_v = np.vstack((X_v, X_V))

                y_h = np.hstack((y_h, y_H))
                y_s = np.hstack((y_s, y_S))
                y_v = np.hstack((y_v, y_V))
            else:
                X_h, X_s, X_v, y_h, y_s, y_v = self.preprocess_training_data2(patch)
                boolean = True
        return X_h, X_s, X_v, y_h, y_s, y_v 

if __name__ == "__main__":
    from sklearn.datasets import load_sample_image
    china_image = load_sample_image("china.jpg")
    img = change_color(china_image, "RGB", "HSV")
    new_img = add_noise(img, threshold=0.0001)
    obj = Inpainting(new_img, patch_size=5)
    obj.inpaint()
    out = obj.predictions()
