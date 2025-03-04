'''
Threshold-based segmentors
https://scikit-image.org/docs/0.23.x/api/skimage.filters.html#
'''

from microseg.widgets.base import List
import skimage.filters as skfilt
from scipy.ndimage import gaussian_filter, label as ndimage_label
from skimage.measure import regionprops
import numpy as np
from qtpy.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLabel

from .auto import *

class ThresholdSegmentorWidget(AutoSegmentorWidget):
    '''
    Segment objects by thresholding with preprocessing options and reference-based filtering
    '''
    METHODS = {
        'otsu': skfilt.threshold_otsu,
        'li': skfilt.threshold_li,
        'yen': skfilt.threshold_yen,
        'isodata': skfilt.threshold_isodata,
        'triangle': skfilt.threshold_triangle,
        'minimum': skfilt.threshold_minimum,
        'local_gaussian': lambda x, bs: skfilt.threshold_local(x, block_size=bs, method='gaussian'),
        'local_mean': lambda x, bs: skfilt.threshold_local(x, block_size=bs)
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        ## Method selection
        row_wdg = HLayoutWidget()
        row_wdg.addWidget(QLabel('Method:'))
        self._method_drop = QComboBox()
        self._method_drop.addItems(self.METHODS.keys())
        row_wdg.addWidget(self._method_drop)
        self._auto_wdg.addWidget(row_wdg)
        
        ## Preprocessing options
        row_wdg = HLayoutWidget()
        self._smooth_check = QCheckBox('Smooth')
        row_wdg.addWidget(self._smooth_check)
        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setRange(0.1, 5.0)
        self._sigma_spin.setValue(1.0)
        self._sigma_spin.setSingleStep(0.1)
        row_wdg.addWidget(self._sigma_spin)
        self._auto_wdg.addWidget(row_wdg)

        # Listeners
        self._method_drop.currentTextChanged.connect(self.recompute)
        self._smooth_check.stateChanged.connect(self.recompute)
        self._sigma_spin.valueChanged.connect(self.recompute)

        # State
        self._method = 'otsu'

    def name(self) -> str:
        return 'Threshold'

    def auto_name(self) -> str:
        return 'Threshold'
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image before thresholding"""
        processed = img.copy()
        
        if self._smooth_check.isChecked():
            processed = gaussian_filter(processed, sigma=self._sigma_spin.value())
            
        # Normalize to 0-1 range
        processed = (processed - processed.min()) / (processed.max() - processed.min())
        
        return processed
    
    def make_proposals(self, img: np.ndarray, poly: PlanarPolygon) -> List[PlanarPolygon]:
        """Generate segmentation proposals using properties of the user-drawn polygon"""
        self._method = self._method_drop.currentText()
        # Get reference measurements from user polygon
        ref_mask = poly.to_mask(img.shape)
        ref_area = poly.area()
        ref_intensity = np.mean(img[ref_mask])
        ref_diameter = poly.diameter()
        
        # Preprocess image
        processed = self.preprocess_image(img)
        
        # Apply threshold
        if 'local' in self._method:
            # Adjust block size based on reference object size
            block_size = int(ref_diameter * 2)  # 2x diameter ensures local context
            block_size = max(35, block_size if block_size % 2 == 1 else block_size + 1)
            thresh = self.METHODS[self._method](processed, block_size)
        else:
            thresh = self.METHODS[self._method](processed)
        
        mask = processed > thresh
        polys = mask_to_polygons(mask)
        return polys
        
    def recompute_auto(self) -> List[PlanarPolygon]:
        return self.make_proposals(self._img, self._poly)

