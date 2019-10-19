from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import range
import urllib.request, urllib.error, urllib.parse, os, tempfile

import numpy as np
from scipy.misc import imread, imresize

"""
Utility funkcije koje služe za prikazivanje i obradu slika.
"""

def blur_image(X):
    """
    Veoma blago zamućivanje slike koje će se koristiti kao regularizator za
    generisanje slika.

    Ulazi:
    - X: Slike dimenzija (N, 3, H, W)

    Vraća:
    - X_blur: Zamućene verzije slika iz X, dimenzija (N, 3, H, W)
    """
    from funkcije.fast_layers import conv_forward_fast
    w_blur = np.zeros((3, 3, 3, 3))
    b_blur = np.zeros(3)
    blur_param = {'stride': 1, 'pad': 1}
    for i in range(3):
        w_blur[i, i] = np.asarray([[1, 2, 1], [2, 188, 2], [1, 2, 1]],
                                  dtype=np.float32)
    w_blur /= 200.0
    return conv_forward_fast(X, w_blur, b_blur, blur_param)[0]


SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(img):
    """Obrada slika za squeezenet.
    
    Oduzimanje srednje vrijednosti i dijeljenje sa standardnom devijacijom piksela.
    """
    return (img.astype(np.float32)/255.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD


def deprocess_image(img, rescale=False):
    """Operacija koja vraća denormalizuje sliku kako bi se mogla prikazati."""
    img = (img * SQUEEZENET_STD + SQUEEZENET_MEAN)
    if rescale:
        vmin, vmax = img.min(), img.max()
        img = (img - vmin) / (vmax - vmin)
    return np.clip(255 * img, 0.0, 255.0).astype(np.uint8)


def image_from_url(url):
    """
    Učitavanje slike sa određenog URL-a. Funkcija vraća numpy niz koji sadrži
    vrijednosti piksela. Slika se upisuje u fajl kako bi se kasnije čitala direktno sa diska.
    """
    try:
        f = urllib.request.urlopen(url)
        _, fname = tempfile.mkstemp()
        with open(fname, 'wb') as ff:
            ff.write(f.read())
        img = imread(fname)
        os.remove(fname)
        return img
    except urllib.error.URLError as e:
        print('URL Greška: ', e.reason, url)
    except urllib.error.HTTPError as e:
        print('HTTP Greška: ', e.code, url)


def load_image(filename, size=None):
    """Učitavanje i predimenzionisanje slike sa diska.

    Inputs:
    - filename: putanja do fajla
    - size: veličina najmanje dimenzije nakon predimenzionisanja
    """
    img = imread(filename)
    if size is not None:
        orig_shape = np.array(img.shape[:2])
        min_idx = np.argmin(orig_shape)
        scale_factor = float(size) / orig_shape[min_idx]
        new_shape = (orig_shape * scale_factor).astype(int)
        img = imresize(img, scale_factor)
    return img
