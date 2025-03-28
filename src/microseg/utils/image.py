'''
Image utilities
'''

import numpy as np 
import cv2
from typing import Tuple
import math
import skimage 
import skimage.exposure
from PIL import Image

from matgeo import PlanarPolygon

def img_apply_affine(img: np.ndarray, aff: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Return new image and affine transform, accounting for pixel bounds
    '''
    img_bdry = PlanarPolygon.from_shape(img.shape).apply_affine(aff)
    xmin, ymin = np.min(img_bdry.vertices, axis=0)
    xmax, ymax = np.max(img_bdry.vertices, axis=0)
    wnew, hnew = math.ceil(xmax-xmin), math.ceil(ymax-ymin)
    trans = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
    aff = trans @ aff
    img = cv2.warpAffine(img, aff[:2], (wnew, hnew))
    return img, aff

def rescale_intensity(img: np.ndarray, i0=2, i1=98) -> np.ndarray:
    '''
    Autoscale intensity per channel
    '''
    if img.ndim == 2:
        rng = tuple(np.percentile(img, (i0, i1)))    
        img = skimage.exposure.rescale_intensity(img, in_range=rng)
    elif img.ndim == 3:
        img = np.stack((
            skimage.exposure.rescale_intensity(img[:, :, c], in_range=tuple(np.percentile(img[:, :, c], (i0, i1))))
            for c in range(img.shape[2])
        ), axis=-1)
    return img

def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    '''
    Convert RGB image to grayscale
    ''' 
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.dtype == np.uint8
    pil_img = Image.fromarray(img.astype(np.uint8)).convert('L')
    return np.array(pil_img)