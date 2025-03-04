'''
Manual segmentor dialog
'''
from numpy import ndarray
from microseg.widgets.base import List
from .base import *

class ManualSegmentorWidget(SegmentorWidget):
    def name(self) -> str:
        return 'Manual'
    
    def recompute(self, img: ndarray, poly: PlanarPolygon) -> List[PlanarPolygon]:
        self.set_proposals([poly])