'''
Image + ROI overlays
- Support for manually drawing polygons
- Support for semi-automatically drawing polygons via standard segmentation algorithms (cellpose, watershed, etc.)
'''
from typing import Set, Type
from qtpy.QtWidgets import QRadioButton, QLabel, QCheckBox, QComboBox, QLabel

from .pg import *
from .roi import *
from .seg import *
from microseg.utils.image import rescale_intensity

class ROIsImageWidget(ImagePlotWidget, metaclass=QtABCMeta):
    '''
    Editable widget for displaying and drawing ROIs on an image
    '''
    startDrawing = QtCore.Signal() 
    finishDrawing = QtCore.Signal(object) # PlanarPolygon
    delete = QtCore.Signal(object) # Set[int], indices into ROIs

    def __init__(self, drawable: bool=False, **kwargs):
        super().__init__(**kwargs)

        # State
        self._is_drawable = drawable
        self._show_rois = True
        self._rois: List[LabeledROI] = []
        self._items: List[LabeledROIItem] = []
        self._selected = set()
        self._shortcuts = []
        self._vb = None
        self._reset_drawing_state()

        # Listeners
        self.scene().sigMouseMoved.connect(self._mouse_move)

    ''' Overrides '''

    def keyPressEvent(self, evt):
        k = evt.key()
        if k == Qt.Key_Escape:
            self._escape()
        elif k == Qt.Key_Delete or k == Qt.Key_Backspace:
            self._delete()
        elif k == Qt.Key_E:
            self.edit()
        elif k == Qt.Key_S:
            self.selectDraw()
        elif evt.modifiers() == Qt.ControlModifier and k == Qt.Key_A:
            self._select(set(range(len(self._rois))))
        else:
            super().keyPressEvent(evt)

    ''' API methods '''

    def setData(self, img: np.ndarray, rois: List[LabeledROI]):
        self.setImage(img)
        self.setROIs(rois)

    def setROIs(self, rois: List[LabeledROI]):
        # Remove previous rois
        self._selected = set()
        self._rois = rois
        for item in self._items:
            self.removeItem(item)
        # Add new rois
        self._items = []
        if self._show_rois:
            for i, roi in enumerate(rois):
                item = roi.toItem(self._shape())
                self.addItem(item)
                self._listen_item(i, item)
                self._items.append(item)
        self._reset_drawing_state()

    def setDrawable(self, bit: bool):
        self._is_drawable = bit

    def setShowROIs(self, bit: bool):
        self._show_rois = bit
        self.setROIs(self._rois)

    def edit(self):
        if self._is_drawable and not self._is_drawing and not self._is_selecting:
            print('edit')
            self.startDrawing.emit()
            self._start_drawing()
    
    def selectDraw(self):
        if not self._is_drawing:
            print('selectDraw')
            self._is_selecting = True
            self._start_drawing()

    ''' Private methods '''

    def _listen_item(self, i: int, item: LabeledROIItem):
        item.sigClicked.connect(lambda: self._select({i}))

    def _select(self, indices: Set[int]):
        selected = (
            self._selected | indices if QGuiApplication.keyboardModifiers() & Qt.ShiftModifier
            else indices
        )
        self._unselect_all()
        for i in selected:
            self._items[i].select()
        print(f'Selected: {selected}')
        self._selected = selected

    def _unselect_all(self):
        for i in self._selected:
            self._items[i].unselect()
        self._selected = set()

    def _delete(self):
        if len(self._selected) > 0:
            print(f'propose delete {len(self._selected)} things')
            self.delete.emit(self._selected)

    def _escape(self):
        print('escape')
        if not self._drawn_item is None:
            self.removeItem(self._drawn_item)
        self._reset_drawing_state()
        self._select({})

    def _init_drawing_state(self):
        self._drawn_poses = []

    def _start_drawing(self):
        self._select({})
        self._is_drawing = True
        self._vb = self.getViewBox()
        self._vb.setMouseEnabled(x=False, y=False)

    def _reset_drawing_state(self):
        self._init_drawing_state()
        self._is_drawing = False
        self._is_selecting = False
        self._drawn_item = None
        if not self._vb is None:
            self._vb.setMouseEnabled(x=True, y=True)
        self._vb = None

    def _modify_drawing_state(self, pos: np.ndarray):
        self._drawn_poses.append(pos.copy())
        vertices = np.array(self._drawn_poses)
        if vertices.shape[0] > 2:
            if not self._drawn_item is None:
                self.removeItem(self._drawn_item)
            lroi = LabeledROI(65, PlanarPolygon(vertices, use_chull_if_invalid=True)).fromPyQTOrientation(self._shape())
            self._drawn_item = lroi.toItem(
                self._shape(), # Original vertices already in PyQT orientation, do the identity transform
                dashed=self._is_selecting,
            ) 
            self.addItem(self._drawn_item)

    def _mouse_move(self, pos):
        if self._is_drawing:
            if QtCore.Qt.LeftButton & QtWidgets.QApplication.mouseButtons():
                pos = self._vb.mapSceneToView(pos)
                pos = np.array([pos.x(), pos.y()])
                self._modify_drawing_state(pos)
            else:
                if not self._drawn_item is None:
                    print('ending draw')
                    lroi = self._drawn_item.toROI(self._shape())
                    self.removeItem(self._drawn_item)
                    if self._is_selecting:
                        post_reset_drawing = lambda: self._select({
                            i for i, lr in enumerate(self._rois) if 
                            lr.intersects(lroi) 
                        })
                    else:
                        post_reset_drawing = lambda: self.finishDrawing.emit(lroi.roi)
                    self._reset_drawing_state()
                    post_reset_drawing()

class ROIsCreator(VLayoutWidget):
    '''
    Parent container for ROIsImageWidget for creating ROIs with several options
    - mode selector
    - show options 
    - channel selector for image
    Expects image in XYC format
    '''
    add = QtCore.Signal(object) # List[ROI]
    delete = QtCore.Signal(object) # Set[int], labels of deleted ROIs
    image_changed = QtCore.Signal(object) # np.ndarray XY
    AVAIL_MODES: List[SegmentorWidget] = [
        CellposeSegmentorWidget,
        ThresholdSegmentorWidget,
        ManualSegmentorWidget,
    ]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        self._widget = ROIsImageWidget(editable=True)
        self.addWidget(self._widget)
        self._seg_row = HLayoutWidget()
        self.addWidget(self._seg_row)
        self._img_row = HLayoutWidget()
        self.addWidget(self._img_row)

        ## Segmentation row
        self._seg_row.addSpacing(10)
        self._seg_row.addWidget(QLabel('Segmentation: '))
        self._seg_row.addSpacing(10)
        self._sel_btn = QPushButton('Select')
        self._seg_row.addWidget(self._sel_btn)
        self._seg_row.addSpacing(10)
        self._draw_btn = QPushButton('Draw')
        self._seg_row.addWidget(self._draw_btn)
        self._seg_row.addSpacing(10)
        self._seg_row.addWidget(QLabel('Mode:'))
        self._seg_row.addSpacing(10)
        self._segmentors = [self._add_mode(c) for c in self.AVAIL_MODES]
        self._mode_drop = QComboBox()
        self._mode_drop.addItems([s.name() for s in self._segmentors])
        self._seg_row.addWidget(self._mode_drop)
        self._seg_row.addSpacing(10)
        self._options_box = QCheckBox('Show options')
        self._seg_row.addWidget(self._options_box)
        self._seg_row.addSpacing(10)
        self._rois_box = QCheckBox('Show ROIs')
        self._seg_row.addWidget(self._rois_box)
        self._seg_row.addSpacing(10)
        self._count_lbl = QLabel()
        self._seg_row.addWidget(self._count_lbl)

        ## Image row
        self._img_row.addSpacing(10)
        self._img_row.addWidget(QLabel('Image: '))
        self._img_row.addSpacing(10)
        self._intens_btn = QCheckBox('Rescale intensity')
        self._img_row.addWidget(self._intens_btn)
        self._img_row.addStretch()
        self._chan_slider = IntegerSlider(label='Chan:', mode='scroll')
        self._img_row.addWidget(self._chan_slider)
        self._chan_slider.hide()
        self._img_row.addSpacing(10)
        self._rgb_box = QCheckBox('Interpret RGB')
        self._img_row.addWidget(self._rgb_box)
        self._rgb_box.hide()
        self._img_row.addSpacing(10)

        # State
        self._mode = 0
        self._img = None
        self._rois = []
        self._set_proposing(False)
        self._rois_box.setChecked(True)
        self._rgb_box.setChecked(False)
        self._options_box.setChecked(True)
        self._chan_slider.setData(0, 255, 0)
        self._intens_btn.setChecked(False)
        self._update_settings(redraw=False)

        # Listeners
        self._sel_btn.clicked.connect(lambda: self._widget.selectDraw())
        self._draw_btn.clicked.connect(lambda: self._widget.edit())
        self._mode_drop.currentIndexChanged.connect(self._set_mode)
        self._options_box.stateChanged.connect(lambda _: self._update_settings(redraw=False))
        self._rgb_box.stateChanged.connect(lambda _: self._update_settings(redraw=True))
        self._rois_box.stateChanged.connect(lambda _: self._set_show_rois(self._rois_box.isChecked()))
        self._chan_slider.valueChanged.connect(lambda _: self._update_settings(redraw=True))
        self._widget.startDrawing.connect(lambda: self._draw_btn.setEnabled(False))
        self._widget.finishDrawing.connect(self._add_from_child)
        self._widget.delete.connect(self._delete_from_child)
        self._intens_btn.stateChanged.connect(lambda _: self._redraw_img())

    ''' Overrides '''

    def keyPressEvent(self, evt):
        if evt.key() == Qt.Key_O:
            self._options_box.setChecked(not self._options_box.isChecked())
        elif evt.key() == Qt.Key_M:
            self._mode_drop.setCurrentIndex((self._mode_drop.currentIndex() + 1) % len(self._segmentors))
        elif evt.key() == Qt.Key_R:
            self._rois_box.setChecked(not self._rois_box.isChecked())
        elif evt.key() == Qt.Key_I:
            self._intens_btn.setChecked(not self._intens_btn.isChecked())
        elif evt.key() == Qt.Key_C and self._chan_slider.isVisible():
            self._chan_slider.setValue((self._chan_slider.value() + 1) % self._img.shape[2])
        else:
            super().keyPressEvent(evt)

    ''' API methods '''

    def setData(self, img: np.ndarray, rois: List[LabeledROI]):
        self.setImage(img)
        self.setROIs(rois)

    def setImage(self, img: np.ndarray):
        assert img.ndim in [2, 3], 'Expected XY or XYC image'
        self._img = img
        if img.ndim == 2 or img.shape[2] == 1:
            self._rgb_box.hide()
            self._chan_slider.hide()
        else:
            if img.shape[2] == 3:
                self._rgb_box.show()
                self._rgb_box.setChecked(self._interpret_rgb)
            if self._interpret_rgb:
                self._chan_slider.hide()
            else:
                self._chan_slider.show()
                cmax = img.shape[2] - 1
                self._chan = min(self._chan, cmax)
                self._chan_slider.setData(0, cmax, self._chan)
        self._redraw_img()

    def setROIs(self, rois: List[LabeledROI]):
        assert not self._is_proposing
        self._rois = rois
        self._widget.setROIs(rois)
        self._count_lbl.setText(f'Current: {len(rois)}')

    ''' Private methods '''

    def _add_mode(self, Cls: Type[SegmentorWidget]) -> SegmentorWidget:
        seg = Cls(self)
        seg.propose.connect(self._propose)
        seg.add.connect(self._add)
        seg.cancel.connect(self._cancel)
        return seg

    def _set_mode(self, i: int):
        self._mode = i

    def _update_settings(self, redraw: bool=True):
        self._show_options = self._options_box.isChecked()
        self._interpret_rgb = self._rgb_box.isChecked()
        self._chan = self._chan_slider.value()
        if redraw:
            self.setImage(self._img)

    def _get_img(self) -> np.ndarray:
        ''' Get displayed image using current settings '''
        if self._img.ndim == 2:
            img = self._img
        elif self._img.shape[2] == 1:
            img = self._img[:, :, 0]
        elif self._interpret_rgb:
            img = self._img
        else:
            img = self._img[:, :, self._chan]
        if self._intens_btn.isChecked():
            img = rescale_intensity(img)
        return img
    
    def _redraw_img(self):
        img = self._get_img()
        self._widget.setImage(img)
        self.image_changed.emit(img)

    def _add_from_child(self, poly: PlanarPolygon):
        '''
        Process add() signal from child
        '''
        assert not self._is_proposing
        img = self._get_img()
        seg = self._segmentors[self._mode]
        if self._show_options:
            seg.prompt(img, poly)
        else:
            seg.prompt_immediate(img, poly)

    def _delete_from_child(self, indices: Set[int]):
        '''
        Process delete() signal from child. 
        '''
        if self._is_proposing: 
            # Pass deletion indices to segmentor
            self._segmentors[self._mode].delete(indices)
        else: 
            # Convert indices to labels and bubble to parent
            self.delete.emit({self._rois[i].lbl for i in indices}) 

    def _propose(self, rois: List[ROI]):
        '''
        Show additional proposed ROIs at this level without bubbling them to the parent yet.
        Intended to be called any number of times in sequence.
        '''
        if not self._is_proposing:
            self._set_proposing(True)
        lrois = [LabeledROI(65, r) for r in rois] # All with that color
        self._widget.setROIs(lrois) 
        self._count_lbl.setText(f'Proposed: {len(rois)}')

    def _add(self, rois: List[ROI]):
        '''
        Process add() signal from segmentor, which asynchronously bubbles add() signal from child.
        '''
        if self._is_proposing:
            self._set_proposing(False)
        if len(rois) == 0: 
            # Short-circuit at this level
            self.setROIs(self._rois)
        else:
            # Bubble to parent
            self.add.emit(rois)

    def _cancel(self):
        '''
        Process cancel() signal from segmentor, does not fire add() signal to parent.
        '''
        if self._is_proposing:
            self._set_proposing(False)
        self.setROIs(self._rois)

    def _set_proposing(self, bit: bool):
        self._is_proposing = bit
        self._draw_btn.setEnabled(not bit)
        self._mode_drop.setEnabled(not bit)
        self._widget.setDrawable(not bit)
        if bit:
            self._set_show_rois(True)

    def _set_show_rois(self, bit: bool):
        self._widget.setShowROIs(bit)
        if bit:
            self._count_lbl.show()
        else:
            self._count_lbl.hide()