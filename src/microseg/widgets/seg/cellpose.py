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
    USE_GPU: bool=False
    MODELS: List[str] = [
        'cyto3',
        'nuclei',
    ]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        ## Cellpose settings
        mod_wdg = HLayoutWidget()
        mod_wdg.addWidget(QLabel('Model:'))
        self._cp_mod_drop = QComboBox()
        self._cp_mod_drop.addItems(self.MODELS)
        mod_wdg.addWidget(self._cp_mod_drop)
        self._auto_wdg.addWidget(mod_wdg)
        cellprob_wdg = HLayoutWidget()
        cellprob_wdg.addWidget(QLabel('Cellprob:'))
        self._cp_cellprob_sld = FloatSlider(step=0.1)
        cellprob_wdg.addWidget(self._cp_cellprob_sld)
        self._auto_wdg.addWidget(cellprob_wdg)

        # State
        self._set_cp_model(0)
        self._cp_cellprob_sld.setData(-3, 4, 0.)

        # Listeners
        self._cp_mod_drop.currentIndexChanged.connect(self._set_cp_model)

    ''' Overrides '''

    def name(self) -> str:
        return 'Cellpose'
    
    def auto_name(self) -> str:
        return 'Cellpose'

    def reset_state(self):
        super().reset_state()
        if hasattr(self, '_cp_cellprob_sld'):
            self._cp_cellprob_sld.setValue(0.)

    def recompute_auto(self, img: np.ndarray, poly: PlanarPolygon) -> List[PlanarPolygon]:
        '''
        Compute cellpose polygons using with possible downscaling
        Returns polygons in the original (un-downscaled) coordinate system
        '''
        # diam = poly.circular_radius() * 2
        diam = poly.diameter()
        cellprob = self._cp_cellprob_sld.value()
        mask = self._cp_model.eval(
            img,
            diameter=diam,
            cellprob_threshold=cellprob,
        )[0]
        print(f'Cellpose mask computed with diameter {diam}, cellprob {cellprob}')
        assert mask.shape == img.shape[:2]
        cp_polys = mask_to_polygons(mask)
        print(f'Cellpose found {len(cp_polys)} valid polygons')
        return cp_polys


    def keyPressEvent(self, evt):
        if evt.key() == QtCore.Qt.Key_Return and evt.modifiers() & Qt.ShiftModifier:
            self.recompute_auto()
        else:
            super().keyPressEvent(evt)

    ''' Private methods '''

    def _set_cp_model(self, idx: int):
        '''
        Sets the cellpose model
        '''
        self._cp_model = cellpose.models.Cellpose(
            model_type=self.MODELS[idx],
            gpu=self.USE_GPU
        )

        
## TODO: add spectral clustering for single-segment
## https://scikit-learn.org/dev/auto_examples/cluster/plot_segmentation_toy.html