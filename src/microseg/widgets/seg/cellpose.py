'''
Cellpose3-based segmentor
'''
import numpy as np
from qtpy.QtWidgets import QCheckBox, QComboBox, QLabel, QPushButton, QRadioButton, QButtonGroup
import cellpose
import cellpose.models
import upolygon

from matgeo import PlanarPolygon, Circle, Ellipse
from .base import *
from .manual import ROICreatorWidget

class CellposeSegmentorWidget(SegmentorWidget):
    USE_GPU: bool=False
    MODELS: List[str] = [
        'cyto3',
        'nuclei',
    ]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        ## For Cellpose settings
        self._cp_wdg = VGroupBox('Cellpose settings')
        self._main.addWidget(self._cp_wdg)
        mod_wdg = HLayoutWidget()
        mod_wdg.addWidget(QLabel('Model:'))
        self._cp_mod_drop = QComboBox()
        self._cp_mod_drop.addItems(self.MODELS)
        mod_wdg.addWidget(self._cp_mod_drop)
        self._cp_wdg.addWidget(mod_wdg)
        cellprob_wdg = HLayoutWidget()
        cellprob_wdg.addWidget(QLabel('Cellprob:'))
        self._cp_cellprob_sld = FloatSlider(step=0.1)
        cellprob_wdg.addWidget(self._cp_cellprob_sld)
        self._cp_wdg.addWidget(cellprob_wdg)
        self._cp_btn = QPushButton('Recompute')
        self._cp_wdg.addWidget(self._cp_btn)

        ## For mask -> ROI postprocessing
        self._roi_wdg = VGroupBox('ROI settings')
        self._main.addWidget(self._roi_wdg)
        self._roi_creator = ROICreatorWidget()
        self._roi_wdg.addWidget(self._roi_creator)

        # State
        self._set_cp_model(0)
        self._cp_cellprob_sld.setData(-3, 4, 0.)

        # Listeners
        self._cp_mod_drop.currentIndexChanged.connect(self._set_cp_model)
        self._cp_btn.clicked.connect(self._recompute)
        self._roi_creator.edited.connect(self.propose.emit) # Bubble from the editor

    ''' Overrides '''

    def name(self) -> str:
        return 'Cellpose'

    def make_proposals(self, img: np.ndarray, poly: PlanarPolygon) -> List[ROI]:
        ''' 
        Recomputes only the mask/poly post-processing step if no existing cellpose mask exists.
        Cellpose mask is re-computed only on explicit user request.
        '''
        if self._cp_polys is None:
            self._update_cp_polys(img, poly)
        self._roi_creator.setData(self._cp_polys)

    def reset_state(self):
        super().reset_state()
        self._cp_polys = None

    ''' Private methods '''

    def _set_cp_model(self, idx: int):
        '''
        Sets the cellpose model
        '''
        self._cp_model = cellpose.models.Cellpose(
            model_type=self.MODELS[idx],
            gpu=self.USE_GPU
        )

    def _update_cp_polys(self, img: np.ndarray, poly: PlanarPolygon):
        '''
        Computes & sets cellpose mask
        '''
        diam = poly.circular_radius() * 2
        cellprob = self._cp_cellprob_sld.value()
        mask = self._cp_model.eval(
            img,
            diameter=diam,
            cellprob_threshold=cellprob,
        )[0]
        assert mask.shape == img.shape[:2]
        self._cp_polys = []
        labels = np.unique(mask)
        for l in labels:
            if l == 0:
                continue
            l_mask = mask == l
            _, contours, __ = upolygon.find_contours(l_mask.astype(np.uint8))
            contours = [np.array(c).reshape(-1, 2) for c in contours] # Convert X, Y, X, Y,... to X, Y
            contour = max(contours, key=lambda c: c.shape[0]) # Find longest contour
            poly = PlanarPolygon(contour)
            self._cp_polys.append(poly)
    
    def _recompute(self):
        '''
        Recomputes entire thing
        '''
        assert not self._poly is None and not self._img is None
        self._update_cp_polys(self._img, self._poly)
        self._propose()