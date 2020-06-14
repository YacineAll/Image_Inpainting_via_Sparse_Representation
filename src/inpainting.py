import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from sklearn.linear_model import Lasso

import numpy as np

from tqdm import tqdm

def read_im(fn):
    img = rgb_to_hsv(plt.imread(fn))/255
    return 2*img - 1


def noise(img, prc):
    m, n = (img.shape[0], img.shape[1])
    noise = np.random.randint(100, size=(m, n))
    img[noise < prc, 2] = np.inf


def delete_rect(img, i, j, h, w):
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
    img_copy = img.copy()
    img_copy[img_copy == np.inf] = -1
    img_copy = 255 * (0.5 * img_copy + 0.5)
    img_copy = np.int32(hsv_to_rgb(img_copy))
    plt.imshow(img_copy)


def patch_to_vect(patch):
    h = patch[:, :, 0].flatten()
    s = patch[:, :, 1].flatten()
    v = patch[:, :, 2].flatten()
    return np.concatenate((h, s, v))


def fill_patch(vect, center, img):
    sz = len(vect) // 3
    h = int(np.sqrt(sz))
    patch = np.zeros((h, h, 3))

    patch[:, :, 0] = np.reshape(vect[0:sz], (h, h))
    patch[:, :, 1] = np.reshape(vect[sz:2*sz], (h, h))
    patch[:, :, 2] = np.reshape(vect[2*sz:3*sz], (h, h))

    (i, j) = center
    img[i-h//2:i+h//2, j-h//2:j+h//2] = patch


def get_patch(i, j, h, img):
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
                    dictionnary = np.vstack((dictionnary,patch))
                else:
                    dictionnary = patch
                    boolean = True
            else:
                missing_patches[(i, j)]  = patch 
               

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
    clf = Lasso(alpha=1e-4, max_iter=max_iter)
    X, Z = build_dict(img, h, step)
    assert len(X) > 0, 'Please choose a smaller step'
    
    for z in tqdm(Z,total=len(Z)):
        y = Z[z]
        center = z
        nonzero = np.argwhere(y != np.inf).flatten()
        clf.fit(X[nonzero], y[nonzero])

        missing = np.argwhere(y == np.inf).flatten()
        y[missing] = clf.predict(X[missing])

        fill_patch(y, center, img)
        
def run_bruit(img, prc, patch_size=7, step=7, alpha=1e-4, max_iter=1e8):
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