''' 
2D ROIs
'''

from typing import Tuple, List, Callable
import numpy as np
from qtpy import QtCore
from qtpy.QtWidgets import QGraphicsPolygonItem, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsItem
import pyqtgraph as pg
from abc import ABC, abstractmethod
import pdb

from matgeo import PlanarPolygon, Circle, Ellipsoid, Ellipse

from .base import *
from microseg.utils import pg_colors

'''
ROIs
'''

ROI = Union[PlanarPolygon, Ellipse, Circle]

class LabeledROI:
    def __init__(self, lbl: int, roi: ROI):
        self.lbl = lbl
        self.roi = roi

    def copy(self) -> 'LabeledROI':
        return LabeledROI(self.lbl, self.roi.copy())

    def relabel(self, lbl: int):
        return LabeledROI(lbl, self.roi.copy())

    def flipy(self, yval: float) -> 'LabeledROI':
        return LabeledROI(self.lbl, self.roi.flipy(yval))

    def __add__(self, offset: np.ndarray):
        return LabeledROI(self.lbl, self.roi + offset)
    
    def __sub__(self, offset: np.ndarray):
        return LabeledROI(self.lbl, self.roi - offset)
    
    def __mul__(self, scale: float):
        return LabeledROI(self.lbl, self.roi * scale)
    
    def toItem(self, img_shape: Tuple[int, int], **kwargs) -> 'LabeledROIItem':
        '''
        Converts to item and appropriately transforms to pyqtgraph's orientation defaults
        '''
        self = self.toPyQTOrientation(img_shape)
        if type(self.roi) is PlanarPolygon:
            return LabeledPolygonItem(self, **kwargs)
        elif type(self.roi) is Ellipse:
            return LabeledEllipseItem(self, **kwargs)
        elif type(self.roi) is Circle:
            return LabeledCircleItem(self, **kwargs)
        else:
            raise NotImplementedError
        
    def toPyQTOrientation(self, img_shape: Tuple[int, int]) -> 'LabeledROI':
        '''
        Transform appropriately for rendering in pyqtgraph's weird orientation defaults
        '''
        _, ylen = img_shape
        return self.flipy(ylen)
    
    def fromPyQTOrientation(self, img_shape: Tuple[int, int]) -> 'LabeledROI':
        '''
        Transform appropriately for rendering in pyqtgraph's weird orientation defaults
        '''
        _, ylen = img_shape
        return self.flipy(ylen)
    
    def asPoly(self, n: int=20) -> PlanarPolygon:
        if type(self.roi) is PlanarPolygon:
            return self.roi
        elif type(self.roi) is Ellipse:
            return self.roi.discretize(n)
        elif type(self.roi) is Circle:
            return self.roi.discretize(n)
        else:
            # raise NotImplementedError(f'Invalid ROI type {type(self.roi)}')
            return self.roi
    
    def intersects(self, other: 'LabeledROI') -> bool:
        return self.asPoly().intersects(other.asPoly())
        
'''
ROI Items (for adding to Qt widgets)
'''

class LabeledROIItem:
    def __init__(self, lroi: LabeledROI, selectable: bool=True, show_label: bool=True, dashed: bool=False):
        self._lroi = lroi
        self._selected = False
        self._dashed = dashed
        self._set_pens(show_label=show_label)
        if selectable:
            self._proxy = ClickProxy()
            self.sigClicked = self._proxy.sigClicked
            self.setAcceptHoverEvents(True)

    def _set_pens(self, show_label: bool=True):
        i = self._lroi.lbl % pg_colors.n_pens if show_label else 65
        self._pen = pg_colors.cc_pens_dashed[i] if self._dashed else pg_colors.cc_pens[i]
        self._hpen = pg_colors.cc_pens_hover[i]
        self.setPen(self._pen)

    def hoverEnterEvent(self, event):
        self.setPen(self._hpen)

    def hoverLeaveEvent(self, event):
        if not self._selected:
            self.setPen(self._pen)

    def mousePressEvent(self, event):
        self.sigClicked.emit()
        self.select()

    def select(self):
        self._selected = True
        self.setPen(self._hpen)

    def unselect(self):
        self._selected = False
        self.setPen(self._pen)

    def toROI(self, img_shape: Tuple[int, int]) -> LabeledROI:
        '''
        Converts to ROI and appropriately transforms back from pyqtgraph's orientation defaults
        '''
        return self._lroi.fromPyQTOrientation(img_shape)

    @property
    def lbl(self) -> int:
        return self._lroi.lbl

class LabeledPolygonItem(QGraphicsPolygonItem, LabeledROIItem):
    def __init__(self, lroi: LabeledROI, *args, **kwargs):
        # Both constructors get called automatically, so pass the named one explicitly
        super().__init__(pg.QtGui.QPolygonF([
            QtCore.QPointF(p[0], p[1]) for p in lroi.roi.vertices
        ]), *args, lroi=lroi, **kwargs)

class LabeledEllipseItem(QGraphicsEllipseItem, LabeledROIItem):
    def __init__(self, lroi: LabeledROI, *args, **kwargs):
        # Both constructors get called automatically, so pass the named one explicitly
        super().__init__(*args, lroi=lroi, **kwargs)
        _, rs = lroi.roi.get_axes_stretches()
        hr, vr = rs
        x, y = lroi.roi.v
        self.setRect(x-hr, y-vr, 2*hr, 2*vr)
        theta = lroi.roi.get_rotation()
        self.setTransformOriginPoint(QtCore.QPointF(x, y)) # Transform about ellipse center
        self.setRotation(theta*180/np.pi) # Rotate onto basis P

class LabeledCircleItem(QGraphicsEllipseItem, LabeledROIItem):
    def __init__(self, lroi: LabeledROI, *args, **kwargs):
        # Both constructors get called automatically, so pass the named one explicitly
        super().__init__(*args, lroi=lroi, **kwargs) 
        self.setRadius(lroi.roi.r)

    def setRadius(self, r: float):
        self.setRect(self._lroi.roi.v[0]-r, self._lroi.roi.v[1]-r, 2*r, 2*r)
        self._lroi.roi.r = r