'''
Utilities for manipulating binary & integer masks
'''

import pdb
import numpy as np
import cv2
from scipy.ndimage import find_objects
from typing import List
import shapely
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from scipy.ndimage import binary_erosion, binary_dilation, find_objects
import upolygon
from skimage.morphology import convex_hull_image

from matgeo.plane import *

def mask_to_adjacency(mask: np.ndarray, nb_buffer: float=0.1, return_indices: bool=False, use_chull: bool=True) -> dict:
    '''
    Extract the adjacency structure from an integer mask of multiple objects. 
    Arguments:
    - mask: integer mask of shape (H, W)
    - nb_buffer: additional number of radii to buffer each object by
    - return_indices: whether to return indices of the labels (in sorted order), or the labels themselves (if false, default)
    '''
    assert mask.ndim == 2, 'Mask must be 2D'
    assert nb_buffer >= 0, 'nb_buffer must be non-negative'
    adj = {}
    elems = np.unique(mask)[1:] # Skip zero
    elem_indices = dict(zip(elems, range(len(elems))))
    for elem_index, elem in enumerate(elems): # Skip zero
        if nb_buffer > 0:
            mask_ = mask == elem
            if use_chull:
                mask_ = convex_hull_image(mask_)
            mask_ = mask_.astype(np.uint8)
            radius = np.sqrt(mask_.sum() / np.pi) # Approx radius
            buf = int(np.ceil(nb_buffer * radius))
            mask_ = binary_dilation(mask_, iterations=buf)
            nbs = set(np.unique(mask_ * mask))
            nbs.remove(0)
            nbs.remove(elem)
        else:
            nbs = set()
        if return_indices:
            adj[elem_index] = [elem_indices[nb] for nb in nbs]
        else:
            adj[elem] = list(nbs)
    return adj

def mask_to_com(mask: np.ndarray, as_dict: bool=False, use_chull_if_invalid=False) -> np.ndarray:
    '''
    Enumerate centers of mask in mask in same order as labels (zero label missing)
    '''
    assert mask.ndim == 2, 'Mask must be 2D'
    elems = np.unique(mask)[1:] # Skip zero
    polygons = mask_to_polygons(mask, rdp_eps=0, erode=0, dilate=0, use_chull_if_invalid=use_chull_if_invalid)
    com = np.array([PlanarPolygon(p).centroid() for p in polygons])
    assert len(elems) == len(com)
    return dict(zip(elems, com)) if as_dict else com

def delete_label(mask: np.ndarray, label: int) -> np.ndarray:
    mask = mask.copy()
    mask[mask == label] = 0
    return mask
