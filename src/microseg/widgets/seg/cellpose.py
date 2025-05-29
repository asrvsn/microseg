'''
Cellpose3-based segmentor
'''
import numpy as np
from qtpy.QtWidgets import QCheckBox, QComboBox, QLabel, QPushButton, QRadioButton, QButtonGroup, QLineEdit, QSpinBox
from qtpy.QtGui import QIntValidator
import cellpose
import cellpose.models
import upolygon
from scipy.ndimage import find_objects
import cv2
import pdb
import skimage
import skimage.restoration as skrest

from matgeo import PlanarPolygon, Circle, Ellipse
from microseg.widgets.pg import ImagePlotWidget
from microseg.utils.image import rescale_intensity
import microseg.utils.mask as mutil
from .base import *
from .auto import *

class CellposeSegmentorWidget(AutoSegmentorWidget):
    USE_GPU: bool=True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        ## Cellpose settings
        self._cp_diam_sld = FloatSlider(label='Diam scaling',step=0.01)
        self._cp_diam_sld.setData(0.01, 2, 0.5)
        self._auto_wdg.addWidget(self._cp_diam_sld)

        self._cp_cellprob_sld = FloatSlider(label='Cellprob', step=0.1)
        self._cp_cellprob_sld.setData(-3, 4, 0.)
        self._auto_wdg.addWidget(self._cp_cellprob_sld)

        # State
        self._set_cp_model()

    ''' Overrides '''

    def name(self) -> str:
        return 'Cellpose'
    
    def auto_name(self) -> str:
        return 'Cellpose'

    def reset_state(self):
        super().reset_state()
        if hasattr(self, '_cp_cellprob_sld'):
            self._cp_cellprob_sld.setValue(0.)

    def produces_mask(self) -> bool:
        return True

    def recompute_auto(self, img: np.ndarray, poly: PlanarPolygon) -> np.ndarray:
        '''
        Compute cellpose polygons using with possible downscaling
        Returns mask in the original (un-downscaled) coordinate system
        '''
        # diam = poly.circular_radius() * 2
        diam = poly.diameter() * self._cp_diam_sld.value()
        cellprob = self._cp_cellprob_sld.value()
        mask = self._cp_model.eval(
            img,
            diameter=diam,
            cellprob_threshold=cellprob,
        )[0]
        print(f'Cellpose mask computed with diameter {diam}, cellprob {cellprob}')
        assert mask.shape == img.shape[:2]
        return mask

    def keyPressEvent(self, evt):
        if evt.key() == QtCore.Qt.Key_Return and evt.modifiers() & Qt.ShiftModifier:
            self.recompute_auto()
        else:
            super().keyPressEvent(evt)

    ''' Private methods '''

    def _set_cp_model(self):
        '''
        Sets the cellpose model
        '''
        self._cp_model = cellpose.models.CellposeModel(
            gpu=self.USE_GPU
        )
        # Print GPU information
        device = self._cp_model.device
        print(f"Cellpose model using device: {device}")

        
## TODO: add spectral clustering for single-segment
## https://scikit-learn.org/dev/auto_examples/cluster/plot_segmentation_toy.html