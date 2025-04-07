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

from matgeo import PlanarPolygon, Circle, Ellipse, PlanarPolygonPacking
from microseg.widgets.pg import *
from microseg.utils.image import rescale_intensity, rgb_to_gray
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
        self._polys_box = QCheckBox('Show polygon')
        self.addWidget(self._polys_box)
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
        self._poly = None
        self._processed_img = None
        self._poly_transform = lambda p: p
        self._poly_transform_inv = lambda p: p
        self._down_box.setChecked(False)
        self._down_int.setValue(400)
        self._intens_box.setChecked(False)
        self._filt_btns[0].setChecked(True)
        self._subimg_box.setChecked(False)
        self._polys_box.setChecked(False)
        self._polys_box.setEnabled(False)

        # Listeners
        self._subimg_box.toggled.connect(self._recalculate)
        self._subimg_size.valueChanged.connect(self._recalculate)
        self._polys_box.toggled.connect(self._redraw_img)
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
        voxsize = np.array([scale, scale])
        self._poly_transform = lambda p: (p - offset).rescale(voxsize)
        self._poly_transform_inv = lambda p: p.rescale(1/voxsize) + offset
        ## Set image
        self._processed_img = img
        self._redraw_img()
        print('Image processed')
        self.processed.emit()

    def _redraw_img(self):
        img = self._processed_img
        ## Show image
        if self._subimg_box.isChecked():
            self._polys_box.setEnabled(True)
            self._subimg_view.show()
            self._subimg_view.setFixedSize(220, round(220 * img.shape[0] / img.shape[1]))
            if self._polys_box.isChecked():
                assert not self._poly is None
                poly = self._poly_transform(self._poly)
                print(f'Drawing polygon with area {poly.area()}')
                img = poly.draw_outline(img, label=img.max())
            self._subimg_view.setImage(img)
        else:
            self._subimg_view.hide()
            self._polys_box.setEnabled(False)

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

    def setPoly(self, poly: PlanarPolygon):
        self._poly = poly
        self._redraw_img()

class MaskProcessingWidget(VLayoutWidget):
    processed = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._erode_box = QSpinBox(minimum=0)
        row = HLayoutWidget()
        row.addWidget(QLabel('Erode:'))
        row.addWidget(self._erode_box)
        self.addWidget(row)
        self._dilate_box = QSpinBox(minimum=0, maximum=self._erode_box.value())
        row = HLayoutWidget()
        row.addWidget(QLabel('Dilate:'))
        row.addWidget(self._dilate_box)
        self.addWidget(row)
        row = HLayoutWidget()
        row.addWidget(QLabel('Method:'))
        self._method_box = QComboBox()
        self._method_box.addItems(['marching_squares', 'standard'])
        row.addWidget(self._method_box)
        self.addWidget(row)
        self._chull_box = QCheckBox('Use convex hull if invalid')
        self.addWidget(self._chull_box)

        # State
        self._reset_state()

        # Listeners
        self._erode_box.valueChanged.connect(self._recalculate)
        self._dilate_box.valueChanged.connect(self._recalculate)
        self._method_box.currentIndexChanged.connect(self._recalculate)
        self._chull_box.toggled.connect(self._recalculate)

    ''' Public '''

    def setData(self, mask: np.ndarray):
        self._reset_state()
        self._mask = mask
        self._recalculate()

    @property
    def polygons(self) -> List[PlanarPolygon]:
        return self._polygons

    ''' Privates '''

    def _reset_state(self):
        self._mask = None
        self._polygons = None
        self._erode_box.setValue(0)
        self._dilate_box.setValue(0)
        self._dilate_box.setMaximum(self._erode_box.value())

    def _recalculate(self):
        if not self._mask is None:
            self._polygons = PlanarPolygonPacking.from_mask(
                self._mask,
                erode=self._erode_box.value(),
                dilate=self._dilate_box.value(),
                use_chull_if_invalid=self._chull_box.isChecked(),
                method=self._method_box.currentText(),
            ).polygons
            self.processed.emit()
        self._dilate_box.setMaximum(self._erode_box.value())

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
        print(f'Polygon Selector got {len(polys)} polygons with average area {np.array([p.area() for p in polys]).mean()}')
        if img.ndim == 3 and img.shape[2] == 3:
            img = rgb_to_gray(img)
        else:
            assert img.ndim == 2
        self._img = img
        self._poly = poly
        self._polys = np.array(polys)
        for name, filt in self._filters.items():
            fn = self.FILTERS[name]
            hist_data = np.array([fn(poly, img) for poly in self._polys])
            # print(f'{name} has {len(hist_data)} values')
            filt.setData(hist_data, percentile=True)
        self._recompute()

    ''' Private '''

    def _recompute(self):
        if self._polys.size > 0:
            if self._closest_only:
                center = self._poly.centroid()
                polys = [min(self._polys, key=lambda p: np.linalg.norm(p.centroid() - center))]
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
        ## 2a. Mask processing (optional)
        if self.produces_mask():
            self._mask_wdg = VGroupBox('Mask processing')
            self._main.addWidget(self._mask_wdg)
            self._mask_proc = MaskProcessingWidget()
            self._mask_wdg.addWidget(self._mask_proc)
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
        if self.produces_mask():
            self._mask_proc.processed.connect(self._on_mask_proc)
        self._poly_sel.processed.connect(self._on_poly_sel)

    ''' Overrides '''

    def recompute(self, img: np.ndarray, poly: PlanarPolygon):
        self._img_proc.setData(img, poly) # self.set_proposals() called through bubbled event

    @abc.abstractmethod
    def auto_name(self) -> str:
        ''' Name of auto-segmentation method '''
        pass

    @abc.abstractmethod
    def recompute_auto(self, img: np.ndarray, poly: PlanarPolygon) -> Union[np.ndarray, List[PlanarPolygon]]:
        ''' Recompute auto-segmentation, setting state as necessary, returns either mask or polygons '''
        pass

    @abc.abstractmethod
    def produces_mask(self) -> bool:
        ''' Whether this segmentor produces a mask or polygons directly '''
        pass

    def set_proposals(self, polys: List[PlanarPolygon]):
        self._poly_sel.setData(self._img, self._poly, polys) # super.set_proposals() called through bubbled event

    def reset_state(self):
        super().reset_state()
        self._compute_btn.setEnabled(False)
        if hasattr(self, '_mask_proc'):
            self._mask_proc.reset_state()

    def on_rois_created(self, rois: List[ROI]):
        print(f'Got {len(rois)} proposals')
        if len(rois) == 1:
            self._img_proc.setPoly(rois[0]) # TODO: hack for visualization
        super().on_rois_created(rois)

    ''' Private ''' 

    def _on_img_proc(self):
        self._compute_btn.setEnabled(True)
    
    def _on_compute(self):
        img, F, F_inv = self._img_proc.processed_img, self._img_proc.poly_transform, self._img_proc.poly_transform_inv
        assert not img is None
        if self.produces_mask():
            mask = self.recompute_auto(img, F(self._poly))
            self._mask_proc.setData(mask) # self.set_proposals() called through bubbled event
        else:
            polys = self.recompute_auto(img, F(self._poly))
            polys = [F_inv(p) for p in polys] 
            self.set_proposals(polys)

    def _on_mask_proc(self):
        F_inv = self._img_proc.poly_transform_inv
        self.set_proposals([F_inv(p) for p in self._mask_proc.polygons])

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