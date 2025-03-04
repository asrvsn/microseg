'''
Abstract base class for automated segementors
Includes: 
1. Image pre-processing
2. Dedicated UI section for automated segmentor parameters
3. Post-segmentation filters e.g. based on intensity
'''
import numpy as np
from qtpy.QtWidgets import QCheckBox, QComboBox, QLabel, QPushButton, QRadioButton, QButtonGroup, QLineEdit, QSpinBox
from qtpy.QtGui import QIntValidator
import upolygon
from scipy.ndimage import find_objects
import cv2
import pdb
import skimage
import skimage.restoration as skrest
import traceback

from matgeo import PlanarPolygon, Circle, Ellipse
from microseg.widgets.pg import *
from microseg.utils.image import rescale_intensity
from microseg.utils.mask import draw_poly
import microseg.utils.mask as mutil
from .base import *
from .manual import ROICreatorWidget

class ImageProcessingWidget(VLayoutWidget):
    '''
    Image pre-processing widget
    '''
    FILTERS = [
        ('none', None),
        ('tvb', skrest.denoise_tv_bregman),
        ('bi', skrest.denoise_bilateral),
        ('wvt', skrest.denoise_wavelet),
    ]
    processed = QtCore.Signal()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Widgets
        self._intens_box = QCheckBox('Rescale intensity')
        self.addWidget(self._intens_box)
        down_wdg = HLayoutWidget()
        self._down_box = QCheckBox('Downscale to (px):')
        down_wdg.addWidget(self._down_box)
        down_wdg.addSpacing(10)
        self._down_int = QSpinBox(minimum=100, maximum=10000)
        down_wdg.addWidget(self._down_int)
        self.addWidget(down_wdg)
        self.addWidget(QLabel('Denoise:'))
        self._filt_btn_grp = QButtonGroup(self)
        self._filt_btns = []
        for (name,_) in self.FILTERS:
            btn = QRadioButton(name)
            self.addWidget(btn)
            self._filt_btns.append(btn)
            self._filt_btn_grp.addButton(btn)
    
        # State
        self._img = None
        self._processed_img = None
        self._down_box.setChecked(False)
        self._down_int.setValue(400)
        self._intens_box.setChecked(False)
        self._filt_btns[0].setChecked(True)

        # Listeners
        for btn in [self._down_box, self._intens_box]:
            btn.toggled.connect(self._recalculate)
        self._filt_btn_grp.buttonClicked.connect(self._recalculate)
        self._down_int.valueChanged.connect(self._recalculate)
    
    def _recalculate(self):
        assert not self._img is None
        img = self._img.copy()
        ## Intensity
        if self._intens_box.isChecked():
            img = rescale_intensity(img)
        ## Downsampling
        if self._down_box.isChecked():
            down_n = self._down_int.value()
            if down_n < max(img.shape):
                img = skimage.transform.rescale(img, self.scale, anti_aliasing=True)
        ## Denoising
        for i, btn in enumerate(self._filt_btns):
            if btn.isChecked():
                dn_fn = self.FILTERS[i][1]
                if not dn_fn is None:
                    dnkw = dict(channel_axis=-1) if img.ndim == 3 else {}
                    img = skrest.denoise_invariant(img, dn_fn, denoiser_kwargs=dnkw)
                break
        ## Emit
        self._processed_img = img
        print('Image processed')
        self.processed.emit()

    ''' Public '''

    @property
    def scale(self) -> float:
        return self._down_int.value() / max(self._img.shape) if self._down_box.isChecked() else 1
    
    @property
    def processed_img(self) -> np.ndarray:
        return self._processed_img
    
    def setImage(self, img: np.ndarray):
        self._img = img
        self._recalculate()


class PolySelectionWidget(VLayoutWidget):
    '''
    Widget for selecting polygons from some set given an image
    TODO: some de-deduplication of logic with ROICreatorWidget?
    '''
    processed = QtCore.Signal(object) # List[PlanarPolygon]
    FILTERS = {
        'Area': lambda poly, img: 
            poly.area()
        ,
        'Intensity (max)': lambda poly, img: 
            img[draw_poly(np.zeros(img.shape, dtype=np.uint8), poly)].max()
        ,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Widgets
        self._filters = dict()
        for name in self.FILTERS.keys():
            filt = HistogramFilterWidget(title=name)
            self.addWidget(filt)
            self._filters[name] = filt
            filt.filterChanged.connect(self._recompute)

        # State
        self._img = None
        self._polys = []

    ''' API '''

    def setData(self, img: np.ndarray, polys: List[PlanarPolygon]):
        print('Polygon Selector set data')
        assert img.ndim == 2
        self._img = img
        self._polys = np.array(polys)
        print(self._polys)
        for name, filt in self._filters.items():
            fn = self.FILTERS[name]
            hist_data = np.array([fn(poly, img) for poly in self._polys])
            filt.setData(hist_data)
        self._recompute()

    ''' Private '''

    def _recompute(self):
        if self._polys.size > 0:
            mask = np.logical_and.reduce([
                filt.mask for filt in self._filters.values()
            ])
            self.processed.emit(self._polys[mask])
        else:
            print('No polygons to process')
            self.processed.emit([])


class AutoSegmentorWidget(SegmentorWidget):
    '''
    Abstract base class for automated segmentors
    Note that only automated segmentors can produce multiple proposals.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Widgets
        ## 1. Image preprocessing
        img_gbox = VGroupBox('Image processing')
        self._main.addWidget(img_gbox)
        self._img_proc = ImageProcessingWidget()
        img_gbox.addWidget(self._img_proc)
        self._main.addSpacing(10)
        ## 2. Auto-segmentor
        self._auto_wdg = VGroupBox(self.auto_name())
        self._main.addWidget(self._auto_wdg)
        self._recompute_btn = QPushButton('Recompute')
        self._main.addWidget(self._recompute_btn)
        self._main.addSpacing(10)
        ## 3. Polygon selection
        poly_gbox = VGroupBox('Polygon selection')
        self._main.addWidget(poly_gbox)
        self._poly_sel = PolySelectionWidget()
        poly_gbox.addWidget(self._poly_sel)
        self._main.addSpacing(10)

        # Listeners
        self._recompute_btn.clicked.connect(self._recompute_auto)
        self._poly_sel.processed.connect(self._set_proposals)

    ''' Overrides '''

    @abc.abstractmethod
    def auto_name(self) -> str:
        ''' Name of auto-segmentation method '''
        pass

    @abc.abstractmethod
    def recompute_auto(self) -> List[PlanarPolygon]:
        ''' Re-compute auto-segmentation, setting state as necessary '''
        pass
    
    ''' Privates '''

    def _recompute_auto(self):
        self._poly_sel.setData(self._img, self.recompute_auto()) # _set_proposals() called through bubbled event

def mask_to_polygons(mask: np.ndarray) -> List[PlanarPolygon]:
    polys = []
    slices = find_objects(mask)
    for i, si in enumerate(slices):
        if si is None:
            continue
        sr, sc = si
        i_mask = mask[sr, sc] == (i+1)
        _, contours, __ = upolygon.find_contours(i_mask.astype(np.uint8))
        contours = [np.array(c).reshape(-1, 2) for c in contours] # Convert X, Y, X, Y,... to X, Y
        if len(contours) > 0:
            contour = max(contours, key=lambda c: cv2.contourArea(c)) # Find max-area contour
            if contour.shape[0] < 3:
                continue
            contour = contour + np.array([sc.start, sr.start])
            try:
                poly = PlanarPolygon(contour)
                polys.append(poly)
            except:
                pass
    return polys