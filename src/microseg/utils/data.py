'''
Utils for reading data and metadata
'''

import numpy as np
import os
from aicsimageio import AICSImage
from PIL import Image
import tifffile
from typing import Tuple, Optional
from skimage.io import imread
import pdb
import pickle
import sys
import pymupdf
import io

def get_voxel_size(path: str, fmt: str='XYZ') -> np.ndarray:
    '''
    Get physical pixel sizes
    '''
    assert os.path.exists(path), f'File not found: {path}'
    _, fext = os.path.splitext(path)
    if fext in ['.czi', '.tif', '.tiff']:
        sizes = AICSImage(path).physical_pixel_sizes
        z = sizes.Z
        if z is None:
            z = 1.0 # TODO: hack
        return np.array([{
            'Z': z, 'Y': sizes.Y, 'X': sizes.X
        }[c] for c in fmt.upper()])
    else:
        raise NotImplementedError

def load_XY_image(path: str, gray: bool=True, imscale: Optional[Tuple[float, float]]=None) -> np.ndarray:
    '''
    Return an XY image.
    gray: if True, convert to grayscale
    imscale: optional rescaling factor for images
    '''
    assert os.path.exists(path), f'File not found: {path}'
    _, fext = os.path.splitext(path)
    def convert_PIL_to_numpy(im):
        if not imscale is None:
            # Rescale x/y axes 
            r1, r2 = round(im.size[0] * imscale[0]), round(im.size[1] * imscale[1])
            print(f'Rescaling image from {im.size} to {(r1, r2)}')
            im = im.resize((r1, r2))
        if gray and im.mode in ['RGB', 'RGBA']:
            im = im.convert('L')
        return np.array(im)

    if fext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        img = convert_PIL_to_numpy(Image.open(path))
    elif fext in ['.pdf']:
        # Extract first image from pdf
        doc = pymupdf.open(path)
        img = None
        for page in doc:
            imlist = page.get_images(full=True)
            if imlist:
                xref = imlist[0][0]
                iminfo = doc.extract_image(xref)
                imbytes = iminfo['image']
                im = Image.open(io.BytesIO(imbytes))
                img = convert_PIL_to_numpy(im)
                break
        if img is None:
            raise ValueError(f'No image found in {path}')
    else:
        raise NotImplementedError(f'File type {fext} not supported')
    if gray:
        assert img.ndim == 2, f'Expected 2d image, got {img.ndim}d image'
    return img

def load_stack(path: str, imscale: Optional[Tuple[float, float]]=None, fmt='ZXYC') -> np.ndarray:
    '''
    Return data in ZXYC format
    imscale: optional rescaling factor for slices
    '''
    assert os.path.exists(path), f'File not found: {path}'
    fmt = fmt.upper()
    assert len(fmt) == 4 and set(fmt) == {'Z', 'X', 'Y', 'C'}, 'fmt must be permutation of ZXYC'
    is_formatted = False
    _, fext = os.path.splitext(path)
    if fext == '.czi':
        img = AICSImage(path).get_image_data(fmt)
        assert imscale is None, 'Rescaling not supported for CZI files yet'
        is_formatted = True # Format achieved by AICSImage call
    elif fext in ['.tif', '.tiff']:
        # Read tiff data
        img = imread(path)
        # Rectify shape based on available metadata
        if img.ndim < 4:
            with tifffile.TiffFile(path) as tif:
                if img.ndim == 2:
                    img = img[np.newaxis, :, :, np.newaxis]
                elif tif.is_imagej:
                    if not ('channels' in tif.imagej_metadata):
                        img = img[:, :, :, np.newaxis] # Image is ZXY
                    elif not ('slices' in tif.imagej_metadata):
                        img = np.array([img]) # Image is XYC
                    else:
                        raise ValueError('Image is <4D yet has both slices and channels, I dont get it.')
                elif img.ndim == 3:
                    img = img[:, :, :, np.newaxis] # Assume image is ZXY
                else:
                    raise NotImplementedError('Cant read metadata from non-ImageJ tiff files yet')
        if not imscale is None:
            # Resize each slice (now in ZXYC)
            r1, r2 = round(img.shape[1] * imscale[0]), round(img.shape[2] * imscale[1])
            print(f'Rescaling image from {img.shape[1:3]} to {(r1, r2)}')
            # Restack to ZCXY for rescaling
            img = img.transpose(0, 3, 1, 2)
            img = np.array([
                np.array([
                    np.array(Image.fromarray(chan).resize((r1, r2)))
                    for chan in frame
                ])
                for frame in img
            ])
            # Restack to ZXYC
            img = img.transpose(0, 2, 3, 1)
    elif fext in ['.seg']: # Support Segmentation2D files, uses the z-projected version
        # Monkey-patch modules
        from microseg.data import seg_2d
        from matgeo import plane
        sys.modules['seg_2d'] = seg_2d 
        sys.modules['plane'] = plane
        seg = pickle.load(open(path, 'rb'))
        img = seg.zproj.copy()
        # Reshape CXY to ZXYC
        img = np.array([img.transpose(1, 2, 0)])
        return img
    else:
        img = load_XY_image(path, imscale=imscale, gray=False)
        img = np.array([img])
    # data = itk.array_view_from_image(itk.imread(args.data))
    assert img.ndim == 4, f'Expected 4d image, got {img.ndim}d image'
    if not is_formatted and fmt != 'ZXYC':
        img = img.transpose(*['ZXYC'.index(ax) for ax in fmt])
    return img