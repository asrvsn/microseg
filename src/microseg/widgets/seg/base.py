'''
Base classes for turning prompt user polygons into segmentations
Segmentor widgets are basically floating menus
'''
import abc
import numpy as np
from typing import Set
from qtpy import QtWidgets
from qtpy.QtWidgets import QPushButton, QWidget, QRadioButton, QButtonGroup, QCheckBox
from qtpy.QtCore import Qt

from matgeo import PlanarPolygon, Circle, Ellipse
from microseg.widgets.base import *
from microseg.widgets.roi import ROI

class TouchpadWidget(QWidget):
    moved = QtCore.Signal(object) # np.array (x, y) offset
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self._mousedown_pos = None

    def mousePressEvent(self, evt):
        if evt.button() == Qt.LeftButton:
            self._mousedown_pos = evt.pos()

    def mouseMoveEvent(self, evt):
        if not self._mousedown_pos is None:
            offset = np.array([
                evt.pos().x() - self._mousedown_pos.x(),
                evt.pos().y() - self._mousedown_pos.y()
            ])
            self.moved.emit(offset)
            self._mousedown_pos = evt.pos()

    def mouseReleaseEvent(self, evt):
        if evt.button() == Qt.LeftButton:
            self._mousedown_pos = None


class ROICreatorWidget(VLayoutWidget):
    '''
    Create ROIs from polygons
    '''
    edited = QtCore.Signal(object) # List[ROI]
    MOVE_SCALE = 0.3
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        self._poly_btn = QRadioButton('Polygon')
        self.addWidget(self._poly_btn)
        self._poly_wdg = VGroupBox()
        self._chull_box = QCheckBox('Convex hull')
        self._poly_wdg.addWidget(self._chull_box)
        self._simplify_sld = FloatSlider(label='Simplify:', step=0.001)
        self._poly_wdg.addWidget(self._simplify_sld)
        self.addWidget(self._poly_wdg)
        self._ellipse_btn = QRadioButton('Ellipse')
        self.addWidget(self._ellipse_btn)
        self._circle_btn = QRadioButton('Circle')
        self.addWidget(self._circle_btn)
        self._roi_grp = QButtonGroup(self)
        for btn in [self._poly_btn, self._ellipse_btn, self._circle_btn]:
            self._roi_grp.addButton(btn)
        self.addSpacing(10)
        self._scale_sld = FloatSlider(label='Scale:', step=0.01)
        self.addWidget(self._scale_sld)
        touch_grp = VGroupBox('Move')
        self._touchpad = TouchpadWidget()
        self._touchpad.setFixedSize(200, 133)
        touch_grp.addWidget(self._touchpad)
        self.addWidget(touch_grp)

        # State
        self._polys = []
        self._offset = np.array([0, 0])
        self._poly_btn.setChecked(True)
        self._chull_box.setChecked(False)
        self._simplify_sld.setData(0., 0.01, 0.)
        self._scale_sld.setData(0.8, 1.2, 1.0)

        # Listeners
        for btn in [self._poly_btn, self._ellipse_btn, self._circle_btn, self._chull_box]:
            btn.toggled.connect(self._recompute)
        self._simplify_sld.valueChanged.connect(lambda _: self._recompute())
        self._touchpad.moved.connect(self._on_touchpad_move)
        self._scale_sld.valueChanged.connect(lambda _: self._recompute())

    def setData(self, polys: List[PlanarPolygon]):
        self._polys = polys
        self._simplify_sld.setValue(0.)
        self._recompute()

    def _recompute(self):
        mk_poly = self._poly_btn.isChecked()
        if mk_poly:
            self._poly_wdg.setEnabled(True)
        else:
            self._poly_wdg.setEnabled(False)
        use_chull = self._chull_box.isChecked()
        mk_ell = self._ellipse_btn.isChecked()
        mk_circ = self._circle_btn.isChecked()
        scale = self._scale_sld.value()
        simplify = self._simplify_sld.value()
        rois = []
        for poly in self._polys:
            poly = poly * scale + self._offset * self.MOVE_SCALE
            if mk_poly:
                poly = poly.simplify(simplify)
                if use_chull: 
                    roi = poly.hullify()
                else:
                    roi = poly
            elif mk_ell:
                roi = Ellipse.from_poly(poly)
            elif mk_circ:
                roi = Circle.from_poly(poly)
            else:
                raise Exception('Invalid ROI type')
            rois.append(roi)
        self.edited.emit(rois)

    def _on_touchpad_move(self, dx: np.ndarray):
        ''' Compute offset by integrating touchpad movement '''
        self._offset += dx
        self._recompute()

class SegmentorWidget(VLayoutWidget, metaclass=QtABCMeta):
    propose = QtCore.Signal(object) # List[ROI]
    add = QtCore.Signal() 
    cancel = QtCore.Signal()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Widgets
        self._main = VLayoutWidget()
        self.addWidget(self._main)
        roi_wdg = VGroupBox('ROI settings')
        self._roi_creator = ROICreatorWidget()
        roi_wdg.addWidget(self._roi_creator)
        self.addWidget(roi_wdg)
        self._bottom = HLayoutWidget()
        self._ok_btn = QPushButton('OK')
        self._bottom.addWidget(self._ok_btn)
        self._cancel_btn = QPushButton('Cancel')
        self._bottom.addWidget(self._cancel_btn)
        self.addWidget(self._bottom)

        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint)
        self.setWindowModality(Qt.NonModal)
        self.setWindowTitle(self.name())

        # State
        self.reset_state()

        # Listeners
        self._roi_creator.edited.connect(self.propose.emit) # Bubble from the editor
        self._ok_btn.clicked.connect(self._ok)
        self._cancel_btn.clicked.connect(self._cancel)
        
    ''' Overrides '''

    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def make_proposals(self, img: np.ndarray, poly: PlanarPolygon) -> List[PlanarPolygon]:
        '''
        From a prompt image and polygon, produce a list of candidate polygons, given the current settings.
        Should fire the propose() signal at least once synchronously using these ROIs.
        '''
        pass

    def reset_state(self):
        '''
        Do any state resets in here before the next call.
        '''
        self._img = None
        self._poly = None

    def closeEvent(self, evt):
        self._cancel()

    def keyPressEvent(self, evt):
        if evt.key() == Qt.Key_Escape:
            self._cancel()
        elif evt.key() == Qt.Key_Return:
            self._ok()
        # TODO: avoid reaching into privates here
        elif evt.key() == Qt.Key_P:
            self._roi_creator._poly_btn.setChecked(True)
        elif evt.key() == Qt.Key_H:
            self._roi_creator._chull_box.setChecked(not self._roi_creator._chull_box.isChecked())
        elif evt.key() == Qt.Key_E:
            self._roi_creator._ellipse_btn.setChecked(True)
        elif evt.key() == Qt.Key_C:
            self._roi_creator._circle_btn.setChecked(True)
        else:
            super().keyPressEvent(evt)

    ''' API '''

    def prompt(self, img: np.ndarray, poly: PlanarPolygon):
        '''
        Lets the user to fire propose() or cancel() using buttons.
        '''
        self._img = img
        self._poly = poly
        self.show()
        self._set_proposals(self.make_proposals(self._img, self._poly))

    def prompt_immediate(self, img: np.ndarray, poly: PlanarPolygon):
        '''
        Fires the add() signal immediately after proposing.
        '''
        self._set_proposals(self.make_proposals(img, poly))
        self.add.emit()
        self.reset_state()

    def delete(self, indices: Set[int]):
        '''
        Delete from the current proposals
        '''
        self._set_proposals([
            p for i, p in enumerate(self._proposed_polys) if i not in indices
        ])

    ''' Private methods '''

    def _ok(self):
        '''
        Fire the add() signal asynchronously.
        '''
        assert not self._poly is None
        self.hide()
        self.add.emit()
        self.reset_state()

    def _cancel(self):
        self.hide()
        self.cancel.emit()
        self.reset_state()

    def _set_proposals(self, polys: List[PlanarPolygon]):
        self._proposed_polys = polys
        self._roi_creator.setData(polys)
