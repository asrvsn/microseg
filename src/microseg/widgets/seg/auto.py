'''
Abstract base class for automated segementors
Includes: 
1. Image pre-processing
2. Dedicated UI section for automated segmentor parameters
3. Post-segmentation filters e.g. based on intensity
'''
import numpy as np
from qtpy.QtWidgets import QCheckBox, QComboBox, QLabel, QPushButton, QRadioButton, QButtonGroup, QLineEdit, QSpinBox, QDoubleSpinBox
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
        ## Subimage
        self._subimg_view = QImageWidget()
        self.addWidget(self._subimg_view)
        row_wdg = HLayoutWidget()
        self._subimg_box = QCheckBox('Use subimage')
        row_wdg.addWidget(self._subimg_box)
        self._subimg_size = QDoubleSpinBox(minimum=1.0, maximum=5.0, value=1.5)
        self._subimg_size.setSingleStep(0.1)
        row_wdg.addWidget(self._subimg_size)
        self.addWidget(row_wdg)
        ## Intensity
        self._intens_box = QCheckBox('Rescale intensity')
        self.addWidget(self._intens_box)
        down_wdg = HLayoutWidget()
        ## Downscaling
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
        self._poly_transform = lambda p: p
        self._poly_transform_inv = lambda p: p
        self._down_box.setChecked(False)
        self._down_int.setValue(400)
        self._intens_box.setChecked(False)
        self._filt_btns[0].setChecked(True)
        self._subimg_box.setChecked(False)

        # Listeners
        self._subimg_box.toggled.connect(self._recalculate)
        self._subimg_size.valueChanged.connect(self._recalculate)
        for btn in [self._down_box, self._intens_box]:
            btn.toggled.connect(self._recalculate)
        self._filt_btn_grp.buttonClicked.connect(self._recalculate)
        self._down_int.valueChanged.connect(self._recalculate)
    
    def _recalculate(self):
        assert not self._img is None and not self._poly is None
        img = self._img.copy()
        poly = self._poly.copy()
        offset = np.array([0, 0])
        scale = self._down_int.value() / max(self._img.shape) if self._down_box.isChecked() else 1
        ## Select subimage
        if self._subimg_box.isChecked():
            center = poly.centroid()
            radius = np.linalg.norm(poly.vertices - center, axis=1).max() * self._subimg_size.value()
            # Select image by center +- radius 
            xmin = max(0, math.floor(center[0] - radius))
            xmax = min(img.shape[1], math.ceil(center[0] + radius))
            ymin = max(0, math.floor(center[1] - radius))
            ymax = min(img.shape[0], math.ceil(center[1] + radius))
            # Update offset & img
            offset = np.array([xmin, ymin])
            img = img[ymin:ymax, xmin:xmax].copy()
            # Show image
            self._subimg_view.show()
            self._subimg_view.setFixedSize(220, round(220 * img.shape[0] / img.shape[1]))
            self._subimg_view.setImage(img)
        #     def _render_img(self, poly: PlanarPolygon):
        # subimg = self._img_proc.processed_img.copy()
        # scale = self._img_proc.scale
        # offset = self._offset
        # # Render
        # ar = subimg.shape[0] / subimg.shape[1]
        # self._subimg_view.setFixedSize(220, round(220 * ar))
        # if self._drawing_box.isChecked():
        #     subimg = mutil.draw_outline(subimg, (poly - offset).set_res(scale, scale))
        # self._subimg_view.setImage(subimg)  
        else:
            self._subimg_view.hide()
        ## Intensity
        if self._intens_box.isChecked():
            img = rescale_intensity(img)
        ## Downsampling
        if self._down_box.isChecked():
            down_n = self._down_int.value()
            if down_n < max(img.shape):
                img = skimage.transform.rescale(img, scale, anti_aliasing=True)
        ## Denoising
        for i, btn in enumerate(self._filt_btns):
            if btn.isChecked():
                dn_fn = self.FILTERS[i][1]
                if not dn_fn is None:
                    dnkw = dict(channel_axis=-1) if img.ndim == 3 else {}
                    img = skrest.denoise_invariant(img, dn_fn, denoiser_kwargs=dnkw)
                break
        ## Polygon transforms
        self._poly_transform = lambda p: (p - offset).set_res(scale, scale)
        self._poly_transform_inv = lambda p: p.set_res(1/scale, 1/scale) + offset
        ## Set image
        self._processed_img = img
        print('Image processed')
        self.processed.emit()

    ''' Public '''
    
    @property
    def processed_img(self) -> np.ndarray:
        return self._processed_img

    @property
    def poly_transform(self) -> Callable:
        return self._poly_transform

    @property
    def poly_transform_inv(self) -> Callable:
        return self._poly_transform_inv
    
    def setData(self, img: np.ndarray, poly: PlanarPolygon):
        self._img = img
        self._poly = poly
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
            img[poly.to_mask(img.shape)].max()
        ,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Widgets
        self._closest_box = QCheckBox('Single closest')
        self._closest_box.toggled.connect(self._on_closest_toggle)
        self.addWidget(self._closest_box)
        self.addSpacing(10)
        self._filts_wdg = VLayoutWidget()
        self._filters = dict()
        for name in self.FILTERS.keys():
            filt = HistogramFilterWidget(title=name)
            self._filts_wdg.addWidget(filt)
            self._filters[name] = filt
            filt.filterChanged.connect(self._recompute)
        self.addWidget(self._filts_wdg)

        # State
        self._img = None
        self._poly = None
        self._polys = []
        self._closest_only = False

    ''' API '''

    def setData(self, img: np.ndarray, poly: PlanarPolygon, polys: List[PlanarPolygon]):
        print(f'Polygon Selector got {len(polys)} polygons')
        assert img.ndim == 2
        self._img = img
        self._poly = poly
        self._polys = np.array(polys)
        for name, filt in self._filters.items():
            fn = self.FILTERS[name]
            hist_data = np.array([fn(poly, img) for poly in self._polys])
            print(f'{name} has {len(hist_data)} values')
            filt.setData(hist_data)
        self._recompute()

    ''' Private '''

    def _recompute(self):
        if self._polys.size > 0:
            if self._closest_only:
                center = self._poly.centroid()
                polys = [min(polys, key=lambda p: np.linalg.norm(p.centroid() - center))]
            else:
                N = len(self._polys)
                polys = self._polys[np.logical_and.reduce([
                    filt.mask for filt in self._filters.values()
                ])]
                # print(f'Filtered from {N} to {len(polys)}')
            self.processed.emit(polys)
        else:
            print('No polygons to process')
            self.processed.emit([])

    def _on_closest_toggle(self):
        self._closest_only = self._closest_box.isChecked()
        if self._closest_only:
            self._filts_wdg.hide()
        else:
            self._filts_wdg.show()
        self._recompute()


class AutoSegmentorWidget(SegmentorWidget):
    '''
    Abstract base class for automated segmentors
    Note that only automated segmentors can produce multiple proposals.
    '''
    def __init__(self, *args, **kwargs):
        self._compute_btn = QPushButton('Compute') # Weird
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
        self._main.addWidget(self._compute_btn)
        self._main.addSpacing(10)
        ## 3. Polygon selection
        poly_gbox = VGroupBox('Polygon selection')
        self._main.addWidget(poly_gbox)
        self._poly_sel = PolySelectionWidget()
        poly_gbox.addWidget(self._poly_sel)
        self._main.addSpacing(10)

        # Listeners
        self._img_proc.processed.connect(self._on_img_proc)
        self._compute_btn.clicked.connect(self._on_compute)
        self._poly_sel.processed.connect(self._on_poly_sel)

    ''' Overrides '''

    def recompute(self, img: np.ndarray, poly: PlanarPolygon):
        self._img_proc.setData(img, poly) # self.set_proposals() called through bubbled event

    @abc.abstractmethod
    def auto_name(self) -> str:
        ''' Name of auto-segmentation method '''
        pass

    @abc.abstractmethod
    def recompute_auto(self, img: np.ndarray, poly: PlanarPolygon) -> List[PlanarPolygon]:
        ''' Recompute auto-segmentation, setting state as necessary '''
        pass

    def set_proposals(self, polys: List[PlanarPolygon]):
        self._poly_sel.setData(self._img, self._poly, polys) # super.set_proposals() called through bubbled event

    def reset_state(self):
        super().reset_state()
        self._compute_btn.setEnabled(False)

    ''' Private ''' 

    def _on_img_proc(self):
        self._compute_btn.setEnabled(True)
    
    def _on_compute(self):
        img, F, F_inv = self._img_proc.processed_img, self._img_proc.poly_transform, self._img_proc.poly_transform_inv
        assert not img is None
        polys = self.recompute_auto(img, F(self._poly))
        polys = [F_inv(p) for p in polys] 
        self.set_proposals(polys)

    def _on_poly_sel(self, polys: List[PlanarPolygon]):
        super().set_proposals(polys)
    

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