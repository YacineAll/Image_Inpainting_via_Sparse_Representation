import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from sklearn.linear_model import Lasso

import numpy as np

from tqdm import tqdm


def read_im(fn):
    """Function to read an RGB image and convert it to HSV

    Args:
        fn (str): path to the image file

    Returns:
        np.ndarray: return a 3D array that represent the image.
    """
    img = rgb_to_hsv(plt.imread(fn))/255
    return 2*img - 1


def noise(img, prc):
    """Method o add noise to the image

    Args:
        img (np.ndarray): the image to transform.
        prc (float): perent of noise to add [0,100].
    """
    m, n = (img.shape[0], img.shape[1])
    noise = np.random.randint(100, size=(m, n))
    img[noise < prc, 2] = np.inf


def delete_rect(img, i, j, h, w):
    """Method to clear a rectangle on given image. 

    Args:
        img (np.ndaraay): [description]
        i (int): The first coordinate of the point to locate the rectangle.
        j (int): The second coordinate of the point to locate the rectangle.
        h (int): Height of the rectangle.
        w (int): Width of the rectangle. 
    """
    assert h % 2 == 1, "Please choose an odd height"
    assert w % 2 == 1, "Please choose an odd width"
    img[i-h//2:i+h//2, j-w//2:j+w//2, 2] = np.inf


def get_img(img):
    img_copy = img.copy()
    img_copy[img_copy == np.inf] = -1
    img_copy = 255 * (0.5 * img_copy + 0.5)
    img_copy = np.int32(hsv_to_rgb(img_copy))
    return img_copy


def show_im(img):
    """Method to show the given image.

    Args:
        img (np.ndarray): the image.
    """
    img_copy = img.copy()
    img_copy[img_copy == np.inf] = -1
    img_copy = 255 * (0.5 * img_copy + 0.5)
    img_copy = np.int32(hsv_to_rgb(img_copy))
    plt.imshow(img_copy)


def patch_to_vect(patch):
    """Function to give a vector from an np.ndarray with 3d (h,h,3).

    Args:
        patch (np.ndarray): A patch with shape (h,h,3)

    Returns:
        np.ndarray: Vecteur with shape (h*h*3)
    """
    h = patch[:, :, 0].flatten()
    s = patch[:, :, 1].flatten()
    v = patch[:, :, 2].flatten()
    return np.concatenate((h, s, v))


def fill_patch(vect, center, img):
    """Method to fill a given img patch centred on center with vect values. 

    Args:
        vect (np.ndarray): np.ndarray with shape (h*h*3), represent values to fill on given patch.
        center (tuple): (x,y) patch center coordinate. 
        img (np.ndarray): the image.
    """
    sz = len(vect) // 3
    h = int(np.sqrt(sz))
    patch = np.zeros((h, h, 3))

    patch[:, :, 0] = np.reshape(vect[0:sz], (h, h))
    patch[:, :, 1] = np.reshape(vect[sz:2*sz], (h, h))
    patch[:, :, 2] = np.reshape(vect[2*sz:3*sz], (h, h))

    (i, j) = center
    img[i-h//2:i+h//2, j-h//2:j+h//2] = patch


def get_patch(i, j, h, img):
    """Functiion to get a patch centred on (i,j) with size h from the image img.

    Args:
        i (int): the first coordinate of the patch center. 
        j (int): the second coordinate of the patch center. 
        h (int): patch size.
        img (np.ndarray): the image where we extract the patch.

    Returns:
        np.ndarray: patch with shape (h,h,3) centred on pixel (i,j)
    """
    assert h % 2 == 1, "Please choose an odd height"
    return patch_to_vect(img[i-h//2:i+h//2, j-h//2:j+h//2])


# def build_dict(img, h, step):
#     dictionnary = []
#     missing_patches = []

#     m, n, d = img.shape
#     limits = (h-1)//2

#     for i in range(limits, m - limits, step):
#         for j in range(limits, n - limits, step):
#             patch = get_patch(i, j, h, img)
#             if np.sum(patch) != np.inf:
#                 dictionnary.append(patch)
#             else:
#                 missing_patches.append((patch, (i, j)))

#     return np.array(dictionnary).T, missing_patches

def build_dict(img, h, step):
    """Function to build the dictionary and patchs containing the missing pixels. 

    Args:
        img (np.ndarray): The image.
        h (int): the patch size.
        step (int): step to iterate over the patchs.

    Returns:
        tuple: tuple, a np.ndarray represent dictionary vecteur of patchs, a dictionary with key reprsent coordinate of the center of the patch and value the patch with the misising pixels
    """
    boolean = False

    dictionnary = None
    missing_patches = dict()

    m, n, d = img.shape
    limits = (h-1)//2

    for i in range(limits, m - limits, step):
        for j in range(limits, n - limits, step):
            patch = get_patch(i, j, h, img)
            if np.sum(patch) != np.inf:
                if(boolean):
                    dictionnary = np.vstack((dictionnary, patch))
                else:
                    dictionnary = patch
                    boolean = True
            else:
                missing_patches[(i, j)] = patch

    return dictionnary.T, missing_patches


# def inpainting(img, h, step, alpha=1e-4, max_iter=1e8):
#     clf = Lasso(alpha=1e-4, max_iter=max_iter)
#     X, Z = build_dict(img, h, step)
#     assert len(X) > 0, 'Please choose a smaller step'

#     for z in Z:
#         y = z[0]
#         center = z[1]

#         nonzero = np.argwhere(y != np.inf).flatten()
#         clf.fit(X[nonzero], y[nonzero])

#         missing = np.argwhere(y == np.inf).flatten()
#         y[missing] = clf.predict(X[missing])

#         fill_patch(y, center, img)

def inpainting(img, h, step, alpha=1e-4, max_iter=5000000):
    """Function to inpainting the given image and predict the missing pixels values with Lasso regression.

    Args:
        img (np.ndarray): img to reconstruct.  
        h (int): the patch size.
        step (int): step to built dictionary.
        alpha (float, optional): alpha regularization to the Lasso regression. Defaults to 1e-4.
        max_iter (int, optional): max iteration to fix for the Lasso algorithm. Defaults to 5000000.
    """
    clf = Lasso(alpha=1e-4, max_iter=max_iter)
    X, Z = build_dict(img, h, step)
    assert len(X) > 0, 'Please choose a smaller step'

    for z in tqdm(Z, total=len(Z)):
        y = Z[z]
        center = z
        nonzero = np.argwhere(y != np.inf).flatten()
        clf.fit(X[nonzero], y[nonzero])

        missing = np.argwhere(y == np.inf).flatten()
        y[missing] = clf.predict(X[missing])

        fill_patch(y, center, img)


def run_bruit(img, prc, patch_size=7, step=7, alpha=1e-4, max_iter=1e8):
    """Method to test the algorithm for a noised image. 

    Args:
        img (np.ndarray): the image.
        prc (float): percent of the noise to add to the image.
        patch_size (int, optional): the patch size. Defaults to 7.
        step (int, optional): step to built the dictionary. Defaults to 7.
        alpha (float, optional): alpha penalization. Defaults to 1e-4.
        max_iter (int, optional): max iteration for lasso regression. Defaults to 1e8.

    Returns:
        np.ndarray: predict image.
    """
    img_original = img.copy()

    noise(img, prc)
    img_noised = img.copy()

    inpainting(img, patch_size, step, alpha, max_iter)

    plt.subplot(131)
    show_im(img_original)
    plt.title('Original image')

    plt.subplot(132)
    show_im(img_noised)
    plt.title('Damaged image')

    plt.subplot(133)
    show_im(img)
    plt.title('Restored image')

    plt.show()
    return img_original, img_noised, img


def run_rect(img, rect, patch_size=7, step=7, alpha=1e-4, max_iter=1e8):
    """Method to test the algorithm for a noised image. 

    Args:
        img (np.ndarray): the image.
        rect (tuple): (i,j,h,w) rectangle information. 
        patch_size (int, optional): the patch size. Defaults to 7.
        step (int, optional): step to built the dictionary. Defaults to 7.
        alpha (float, optional): alpha penalization. Defaults to 1e-4.
        max_iter (int, optional): max iteration for lasso regression. Defaults to 1e8.

    Returns:
        np.ndarray: predict image.
    """

    img_original = img.copy()

    delete_rect(img, *rect)

    img_noised = img.copy()

    inpainting(img, patch_size, step, alpha, max_iter)

    plt.subplot(131)
    show_im(img_original)
    plt.title('Original image')

    plt.subplot(132)
    show_im(img_noised)
    plt.title('Damaged image')

    plt.subplot(133)
    show_im(img)
    plt.title('Restored image')

    plt.show()
    return img_original, img_noised, img
