'''
Threshold-based segmentors
https://scikit-image.org/docs/0.23.x/api/skimage.filters.html#
'''

from microseg.widgets.base import List
import skimage.filters as skfilt

from .auto import *

class ThresholdSegmentorWidget(AutoSegmentorWidget):
    '''
    Segment objects by thresholding
    '''
    METHODS = {
        'otsu': skfilt.threshold_otsu,
        'li': skfilt.threshold_li,
        'yen': skfilt.threshold_yen,
        'isodata': skfilt.threshold_isodata,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        row_wdg = HLayoutWidget()
        row_wdg.addWidget(QLabel('Method:'))
        self._method_drop = QComboBox()
        self._method_drop.addItems(self.METHODS.keys())
        self._method_drop.currentTextChanged.connect(self._on_method_changed)
        row_wdg.addWidget(self._method_drop)
        self._auto_wdg.addWidget(row_wdg)

        # State
        self._method = 'otsu'

    def name(self) -> str:
        return 'Threshold'

    def auto_name(self) -> str:
        return 'Threshold'
    
    def make_proposals(self, img: np.ndarray, poly: PlanarPolygon) -> List[PlanarPolygon]:
        thresh = self.METHODS[self._method](img)
        mask = img > thresh
        polys = mask_to_polygons(mask)
        return polys
        
    def recompute_auto(self) -> List[PlanarPolygon]:
        return self.make_proposals(self._img, self._poly)

    def _on_method_changed(self):
        self._method = self._method_drop.currentText()
        self._set_proposals(self.make_proposals(self._img, self._poly))

