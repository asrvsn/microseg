'''
Pyqtgraph widgets
'''
import os
from typing import Tuple, Optional, List, Callable
import numpy as np
import pyqtgraph as pg
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import QShortcut
from qtpy.QtGui import QGuiApplication
import pickle

import microseg.utils.mask as mutil
from microseg.utils.colors import *
from microseg.utils.data import *
from matgeo import Sphere
from .roi import *

'''
Functions
'''

def link_plots(p1: pg.PlotWidget, p2: pg.PlotWidget):
    p1.setXLink(p2)
    p1.setYLink(p2)
    p2.setXLink(p1)
    p2.setYLink(p1)

'''
Classes
'''

class NoTouchPlotWidget(pg.PlotWidget):
    '''
    Plot widget with touch events disabled
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Don't accept touch events due to bug on MacOS
        # https://bugreports.qt.io/browse/QTBUG-103935
        self.viewport().setAttribute(QtCore.Qt.WA_AcceptTouchEvents, False)

class PlotWidget(NoTouchPlotWidget):
    '''
    Plot widget with some extra listeners
    '''
    sigKeyPress = QtCore.Signal(object)

    def keyPressEvent(self, ev):
        # print('key press', ev.key())
        self.scene().keyPressEvent(ev)
        self.sigKeyPress.emit(ev)

class ImageItem(pg.ImageItem):
    '''
    ImageItem correcting the insane behavior of pyqtgraph with respect to orientation
    '''
    def setImage(self, img: np.ndarray):
        img = np.swapaxes(img, 0, 1)
        img = img[:, ::-1, ...]
        super().setImage(img)

class ImagePlotWidget(NoTouchPlotWidget):
    '''
    Plot widget with default image item, zoom limits set to image
    '''
    def __init__(self, *args, limit_zoom: bool=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAspectLocked(True)
        self.hideAxis('left')
        self.hideAxis('bottom')
        self._img = ImageItem()
        self.addItem(self._img)
        if limit_zoom:
            self.getViewBox().sigRangeChanged.connect(self._limit_zoom)

    def _limit_zoom(self):
        """Limit zoom to not zoom out beyond the image boundaries."""
        vb = self.getViewBox()
        x_range, y_range = vb.viewRange()
        img_shape = self._img.image.shape

        # Calculate the necessary range to keep the view within the image bounds
        new_x_range = [max(0, min(x_range[0], img_shape[1]-1)), min(img_shape[1], max(x_range[1], 0))]
        new_y_range = [max(0, min(y_range[0], img_shape[0]-1)), min(img_shape[0], max(y_range[1], 0))]

        # Set the new range
        vb.setRange(xRange=new_x_range, yRange=new_y_range, padding=0)

    def _shape(self) -> Tuple[int, int]:
        return self._img.image.shape[:2]
    
    def setImage(self, img: np.ndarray):
        self._img.setImage(img)

class MaskItem(ImageItem):
    '''
    Image mask as widget
    '''
    def __init__(self, mask: np.ndarray=None, alpha: float=0.5, **kwargs):
        super().__init__(**kwargs)
        self._im = None
        self._cmap = lambda mask: map_colors(mask, 'mask', i255=True, alpha=alpha)
        self._alpha = alpha
        self._dark_alpha = min(1, alpha*1.7)
        self._hover_label = None
        self._click_label = None
        self._move_sensitive = True
        if not mask is None:
            self.setMask(mask)
    
    def setMask(self, mask: np.ndarray):
        self._mask = mask
        self._labels = np.unique(mask)
        self.draw_mask(recompute=True)

    def draw_mask(self, recompute: bool=False):
        if recompute:
            self._im = self._cmap(self._mask)
        im = self._im.copy()
        for dark_lbl in [self._click_label, self._hover_label]:
            if not dark_lbl is None and dark_lbl != 0:
                im[self._mask == dark_lbl,3] = int(self._dark_alpha * 255)
        self.setImage(im)

    def get_view_pos(self, pos) -> Tuple[float]:
        pos = self._plot._vb.mapSceneToView(pos)
        return round(pos.x()), round(pos.y())

    def get_label_at_pos(self, pos) -> Optional[int]:
        x, y = self.get_view_pos(pos)
        if 0 <= x < self._mask.shape[0] and 0 <= y < self._mask.shape[1]:
            return self._mask[x, y]
        else:
            return None

    def mouse_move(self, pos):
        if not QtCore.Qt.LeftButton & QtWidgets.QApplication.mouseButtons():
            old_label = self._hover_label
            self._hover_label = self.get_label_at_pos(pos)
            # print('move', self._hover_label)
            if old_label != self._hover_label:
                self.draw_mask()

    def mouseClickEvent(self, event):
        self._click_label = self.get_label_at_pos(event.scenePos())
        # print('click', self._click_label)
        self.draw_mask()

    def add_to_plot(self, plot: pg.PlotItem):
        self._plot = plot
        self._plot._vb = plot.getViewBox()
        plot.addItem(self)
        plot.scene().sigMouseMoved.connect(self.mouse_move)

class EditableMaskItem(MaskItem):
    '''
    Mask with editable labels
    '''
    undo_n: int=100
    
    def __init__(self, max_items: int=np.inf, *args, **kwargs):
        self._max_items = max_items
        self._shortcuts = []
        self._plot = None
        self._reset_drawing_state()
        super().__init__(*args, **kwargs)
    
    def setMask(self, mask: np.ndarray):
        self._undo_stack = [mask.copy()]
        self._redo_stack = []
        self._reset_drawing_state()
        super().setMask(mask)
    
    def _peek_mask(self):
        self._mask = self._undo_stack[-1].copy()
        self.draw_mask(recompute=True)
    
    def _update_mask(self, fun: Callable):
        mask = self._undo_stack[-1].copy()
        mask = fun(mask)
        self._undo_stack.append(mask)
        self._undo_stack = self._undo_stack[-self.undo_n:]
        self._redo_stack = []
        self._peek_mask()

    def _delete(self):
        print('delete')
        if not self._click_label is None:
            self._update_mask(lambda mask: mutil.delete_label(mask, self._click_label))
            self._click_label = None

    def _undo(self):
        print('undo')
        if len(self._undo_stack) > 1:
            self._redo_stack.append(self._undo_stack[-1])
            self._undo_stack = self._undo_stack[:-1]
            self._peek_mask()
        else:
            print('Cannot undo further')

    def _redo(self):
        print('redo')
        if len(self._redo_stack) > 0:
            self._undo_stack.append(self._redo_stack[-1])
            self._redo_stack = self._redo_stack[:-1]
            self._peek_mask()
        else:
            print('Cannot redo further')
    
    def _edit(self):
        print('edit')
        if not self._is_drawing:
            if np.unique(self._mask).shape[0] > self._max_items:
                print('Cannot draw more than', self._max_items, 'labels')
                return
            self._is_drawing = True
            self._drawn_label = self._mask.max() + 1
            # print('New drawn label:', self._drawn_label)
            # Disable viewport movement
            self._plot._vb.setMouseEnabled(x=False, y=False)

    def _stop_editing(self):
        if self._is_drawing:
            self._reset_drawing_state()

    def _reset_drawing_state(self):
        self._is_drawing = False
        self._drawn_poly = None
        self._drawn_label = None
        # Enable viewport movement
        if not self._plot is None:
            self._plot._vb.setMouseEnabled(x=True, y=True)

    def _enter(self):
        print('enter')
        self._stop_editing()

    def _escape(self):
        print('escape')
        self._stop_editing()
        self._click_label = None
        self._peek_mask()
    
    def _add_shortcut(self, key: str, fun: Callable):
        sc = QShortcut(QKeySequence(key), self._plot)
        sc.activated.connect(fun)
        self._shortcuts.append(sc)
    
    def add_to_plot(self, plot: pg.PlotItem):
        super().add_to_plot(plot)
        # Add shortcuts
        self._add_shortcut('Delete', lambda: self._delete())
        self._add_shortcut('Backspace', lambda: self._delete())
        self._add_shortcut('Ctrl+Z', lambda: self._undo())
        self._add_shortcut('Ctrl+Y', lambda: self._redo())
        self._add_shortcut('Ctrl+Shift+Z', lambda: self._redo())
        self._add_shortcut('E', lambda: self._edit())
        self._add_shortcut('Enter', lambda: self._enter())
        self._add_shortcut('Escape', lambda: self._escape())

    def mouseClickEvent(self, event):
        if not self._is_drawing:
            super().mouseClickEvent(event)

    def mouse_move(self, pos):
        if self._is_drawing:
            if QtCore.Qt.LeftButton & QtWidgets.QApplication.mouseButtons():
                x, y = self.get_view_pos(pos)
                if self._drawn_poly is None:
                    print('starting draw')
                    self._drawn_poly = [x, y]
                else:
                    self._drawn_poly.append(x)
                    self._drawn_poly.append(y)
                    # Use self._mask as canvas
                    self._mask = mutil.draw_poly(self._mask, self._drawn_poly, self._drawn_label)
                    self.draw_mask(recompute=True)
            else:
                if not self._drawn_poly is None:
                    print('ending draw')
                    self._update_mask(lambda _: self._mask.copy())
                    self._stop_editing()
        else:
            super().mouse_move(pos)

class MaskImageWidget(NoTouchPlotWidget):
    '''
    Combined image + mask items
    '''
    def __init__(self, editable: bool=False, **kwargs):
        super().__init__(**kwargs)
        self.setAspectLocked(True)
        self.hideAxis('left')
        self.hideAxis('bottom')
        self._editable = editable
        self._img_item = ImageItem()
        self.addItem(self._img_item)
        self._mask_item = EditableMaskItem() if editable else MaskItem()
        self._mask_item.add_to_plot(self)

        # State
        self._img = None
        self._mask = None

    def setData(self, img: np.ndarray, mask: np.ndarray):
        assert img.shape[:2] == mask.shape, 'Image and mask must have same shape'
        self._img = img
        self._mask = mask
        self._img_item.setImage(img)
        self._mask_item.setMask(mask)

class HistogramFilterWidget(pg.PlotWidget):
    '''
    Similar to HistogramLUTItem, but without being tied to images.
    '''
    filterChanged = QtCore.Signal() 

    def __init__(self, *args, title='Histogram', **kwargs):
        super().__init__(*args, **kwargs)
        # Set fixed height and margins
        self.setFixedHeight(100)  
        self.setFixedWidth(220)
        
        # Widgets
        ## Plot
        pitem = self.getPlotItem()
        pitem.setTitle(title)
        pitem.hideAxis('left')
        pitem.setMouseEnabled(x=False, y=False)
        ## Histogram
        self._hist = pitem.plot([], [], stepMode=True, fillLevel=0, brush=(0,0,255,150))
        ## Region selection
        self._region = pg.LinearRegionItem(orientation='vertical')
        self._region.setZValue(10)
        pitem.addItem(self._region)

        # State
        self._hist_data = None
        self._mask = None

    def setData(self, data: np.ndarray, n_per_bin: int=10, percentile: bool=False):
        # Temporarily disconnect the region change signal if it exists
        try:
            self._region.sigRegionChanged.disconnect(self._on_region_changed)
        except TypeError:
            pass  # Signal wasn't connected yet
        
        assert n_per_bin > 0
        if len(data) > 0:
            if percentile:
                # Use percentile range
                data = data - data.min()
                data = data / data.max() 
                data *= 100
            self._hist_data = data
            n_bins = max(1, len(data) // n_per_bin)
            hist, bin_edges = np.histogram(data, bins=n_bins)
            self._hist.setData(bin_edges, hist)
            if percentile:
                self._region.setRegion([0, 100]) 
            else:
                self._region.setRegion([data.min(), data.max()]) 
        else:
            self._hist.setData([], [])
            self._region.setRegion([0, 0])  # Reset region
        self._mask = np.full(data.shape, True)
        self._region.sigRegionChanged.connect(self._on_region_changed)

    def _on_region_changed(self):
        if self._hist_data is None or len(self._hist_data) == 0:
            self._mask = np.array([])
        else:
            minX, maxX = self._region.getRegion()
            self._mask = (self._hist_data >= minX) & (self._hist_data <= maxX)
        self.filterChanged.emit()

    @property
    def mask(self):
        return self._mask