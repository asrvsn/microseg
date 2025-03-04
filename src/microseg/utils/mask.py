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

def mask_to_polygons(mask: np.ndarray, rdp_eps: float=0., erode: int=0, dilate: int=0, use_chull_if_invalid=False) -> List[np.ndarray]:
    '''
    Compute outlines of objects in mask as a list of polygon coordinates
    Arguments:
    - mask: integer mask of shape (H, W)
    - rdp_eps: epsilon parameter in the Ramer-Douglas-Peucker algorithm for polygon simplification
    - erode: number of pixels to erode the mask by before computing the polygon to get rid of single-pixel artifacts
    '''
    assert mask.ndim == 2, 'Mask must be 2D'
    assert rdp_eps >= 0
    assert erode >= 0 and dilate >= 0
    polygons = []

    # slices = find_objects(mask)
    # for i,si in enumerate(slices):
    #         if si is not None:
    #             sr, sc = si
    #             submask = mask[sr, sc] == (i+1) # Select ROI
    #             if erode > 0:
    #                 submask = binary_erosion(submask, iterations=erode) # Get rid of single-pixel artifacts
    #             if dilate > 0:
    #                 submask = binary_dilation(submask, iterations=dilate) # Correct area lost in erosion
    #             contours = cv2.findContours(submask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    #             # Take contour with largest coord count
    #             n_coords = [c.size for c in contours]
    #             pvc, pvr = contours[np.argmax(n_coords)].squeeze().T
    #             vr, vc = pvr + sr.start, pvc + sc.start
    #             coords = np.stack([vc, vr], axis=1)
    #             # Construct polygon
    #             poly = Polygon(coords)
    #             if rdp_eps > 0: # Simplify polygon using RDP algorithm
    #                 poly = poly.simplify(rdp_eps, preserve_topology=True)
    #             poly = orient(poly, sign=1.0)
    #             coords = np.array(poly.exterior.coords) # Exterior coordinates oriented counter-clockwise
    #             polygons.append(coords)

    elems = np.unique(mask)
    elems = elems[elems != 0] # Ignore background
    for elem in elems:
        submask = mask == elem
        if erode > 1:
            submask = binary_erosion(submask, iterations=erode) # Get rid of single-pixel artifacts
        if dilate > 1:
            submask = binary_dilation(submask, iterations=dilate) # Correct area lost in erosion
        _, external_paths, internal_paths = upolygon.find_contours(submask.astype(np.uint8))
        # Take contour with largest coord count
        n_coords = [len(c) for c in external_paths]
        contour = external_paths[np.argmax(n_coords)] # in X, Y, X, Y, ... format
        contour = np.array(contour).reshape(-1, 2) # (N, 2) in X, Y format
        p = Polygon(contour)
        # Ensure valid polygons are extracted
        if not p.is_valid:
            if use_chull_if_invalid:
                p = p.convex_hull
            else:
                # Try buffer(0) trick
                p = p.buffer(0)
                if type(p) == shapely.geometry.MultiPolygon:
                    # Extract polygon with largest area
                    p = max(p.geoms, key=lambda x: x.area)
            assert type(p) == shapely.geometry.Polygon
            assert p.is_valid
        contour = np.array(p.exterior.coords)  
        polygons.append(contour)

    return polygons

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
