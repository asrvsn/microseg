'''
Image + ROI overlays
- Support for manually drawing polygons
- Support for semi-automatically drawing polygons via standard segmentation algorithms (cellpose, watershed, etc.)
'''
from typing import Set
from qtpy.QtWidgets import QRadioButton, QLabel, QCheckBox, QComboBox

from .pg import *
from .roi import *
from .seg.base import *

class ROIsImageWidget(ImagePlotWidget, metaclass=QtABCMeta):
    '''
    Editable widget for displaying and drawing ROIs on an image
    '''
    add = QtCore.Signal(object) # PlanarPolygon
    delete = QtCore.Signal(object) # Set[int]

    def __init__(self, editable: bool=False, **kwargs):
        super().__init__(**kwargs)
        self._editable = editable

        # State
        self._rois: List[LabeledROI] = []
        self._items: List[LabeledROIItem] = []
        self._selected = []
        self._shortcuts = []
        self._vb = None
        self._reset_drawing_state()

        # Listeners
        if editable:
            self._add_sc('Delete', lambda: self._delete())
            self._add_sc('Backspace', lambda: self._delete())
            self._add_sc('E', lambda: self._edit())
        self._add_sc('Escape', lambda: self._escape())
        self.scene().sigMouseMoved.connect(self._mouse_move)

    ''' API methods '''

    def setData(self, img: np.ndarray, rois: List[LabeledROI]):
        self.setImage(img)
        self.setROIs(rois)

    def setImage(self, img: np.ndarray):
        self._img.setImage(img)

    def setROIs(self, rois: List[LabeledROI]):
        # Remove previous rois
        self._selected = []
        self._rois = rois
        for item in self._items:
            self.removeItem(item)
        # Add new rois
        self._items = []
        for i, roi in enumerate(rois):
            item = roi.toItem(self._shape())
            self.addItem(item)
            self._listen_item(i, item)
            self._items.append(item)
        self._reset_drawing_state()

    ''' Private methods '''

    def _add_sc(self, key: str, fun: Callable):
        sc = QShortcut(QKeySequence(key), self)
        sc.activated.connect(fun)
        self._shortcuts.append(sc)

    def _listen_item(self, i: int, item: LabeledROIItem):
        item.sigClicked.connect(lambda: self._select(i))

    def _select(self, i: Optional[int]):
        if i is None:
            self._unselect_all()
        elif not i in self._selected:
            if not QGuiApplication.keyboardModifiers() & Qt.ShiftModifier:
                self._unselect_all()
            self._selected.append(i)
            self._items[i].select()
        print(f'Selected: {self._selected}')

    def _unselect_all(self):
        for i in self._selected:
            self._items[i].unselect()
        self._selected = []

    def _delete(self):
        if self._editable and len(self._selected) > 0:
            print(f'propose delete {len(self._selected)} things')
            self.delete.emit(set(self._items[i].lbl for i in self._selected))

    def _edit(self):
        print('edit')
        if self._editable and not self._is_drawing:
            self._select(None)
            self._is_drawing = True
            self._vb = self.getViewBox()
            self._vb.setMouseEnabled(x=False, y=False)

    def _escape(self):
        print('escape')
        if not self._drawn_item is None:
            self.removeItem(self._drawn_item)
        self._reset_drawing_state()
        self._select(None)

    def _init_drawing_state(self):
        self._drawn_poses = []

    def _reset_drawing_state(self):
        self._init_drawing_state()
        self._is_drawing = False
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
            self._drawn_item = lroi.toItem(self._shape()) # Original vertices already in PyQT orientation, do the identity transform
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
                    poly = self._drawn_item.toROI(self._shape()).roi
                    self.removeItem(self._drawn_item)
                    self._reset_drawing_state()
                    self.add.emit(poly)

class ROIsCreator(PaneledWidget):
    '''
    Thin wrapper around ROIsImageWidget for creating ROIs with several options
    - convert drawn poly to ellipse or circle
    - take convex hull of drawn poly
    - channel selector for image
    Expects image in XYC format
    '''
    add = QtCore.Signal(object) # List[ROI]
    delete = QtCore.Signal(object) # Set[int]
    AVAIL_MODES: List[SegmentorWidget] = [
        ManualSegmentorWidget,
        CellposeSegmentorWidget,
    ]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        self._widget = ROIsImageWidget(editable=True)
        self._main_layout.addWidget(self._widget)
        self._bottom_layout.addWidget(QLabel('Mode:'))
        self._bottom_layout.addSpacing(10)
        self._seg_widgets = []
        self._mode_btns = []
        for i in range(len(self.AVAIL_MODES)):
            self._add_mode(i)
        self._bottom_layout.addStretch()
        self._bottom_layout.addWidget(QLabel('Chan:'))
        self._chan_slider = IntegerSlider(mode='scroll')
        self._bottom_layout.addWidget(self._chan_slider)
        self._chan_slider.hide()
        self._rgb_box = QCheckBox('Interpret RGB')
        self._bottom_layout.addSpacing(10)
        self._bottom_layout.addWidget(self._rgb_box)
        self._rgb_box.hide()
        self._options_box = QCheckBox('Show options')
        self._bottom_layout.addSpacing(10)
        self._bottom_layout.addWidget(self._options_box)
        self._count_lbl = QLabel()
        self._bottom_layout.addSpacing(10)
        self._bottom_layout.addWidget(self._count_lbl)
        self._candidates_box = QCheckBox('Candidates only')
        self._bottom_layout.addSpacing(10)
        self._bottom_layout.addWidget(self._candidates_box)
        self._candidates_lbl = QLabel()
        self._bottom_layout.addSpacing(10)
        self._bottom_layout.addWidget(self._candidates_lbl)

        # State
        self._mode = 0
        self._img = None
        self._rois = []
        self._candidate_rois = []
        self._is_prompting = False
        self._mode_btns[self._mode].setChecked(True)
        self._rgb_box.setChecked(True)
        self._options_box.setChecked(False)
        self._chan_slider.setData(0, 255, 0)
        self._candidates_box.setChecked(False)
        self._update_settings(redraw=False)

        # Listeners
        self._options_box.stateChanged.connect(lambda _: self._update_settings(redraw=False))
        self._rgb_box.stateChanged.connect(lambda _: self._update_settings(redraw=True))
        self._chan_slider.valueChanged.connect(lambda _: self._update_settings(redraw=True))
        self._widget.add.connect(self._add_from_child)
        self._widget.delete.connect(self._delete_from_child)

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
            img = img if img.ndim == 2 else img[:, :, 0]
            self._widget.setImage(img)
        else:
            if img.shape[2] == 3:
                self._rgb_box.show()
                self._rgb_box.setChecked(self._interpret_rgb)
            if self._interpret_rgb:
                self._chan_slider.hide()
                self._widget.setImage(img)
            else:
                self._chan_slider.show()
                cmax = img.shape[2] - 1
                self._chan = min(self._chan, cmax)
                self._chan_slider.setData(0, cmax, self._chan)
                self._widget.setImage(img[:, :, self._chan])

    def setROIs(self, rois: List[LabeledROI], reset: bool=True):
        n, m = len(rois), len(self._candidate_rois)
        self._rois = rois
        if reset:
            self._candidate_rois = []
            self._only_candidates = False
            self._candidates_box.setChecked(False)
            self._candidates_box.hide()
            self._candidates_lbl.hide()
        elif self._only_candidates:
            rois = self._candidate_rois
            self._count_lbl.hide()
            self._candidates_box.show()
            self._candidates_lbl.show()
        else:
            rois = rois + self._candidate_rois
            if self._is_prompting:
                self._candidates_box.show()
                self._candidates_lbl.show()
            else:
                self._candidates_box.hide()
                self._candidates_lbl.hide()
            self._count_lbl.show()

        self._count_lbl.setText(f'Current: {n}')
        self._candidates_lbl.setText(f'Candidates: {m}')
        self._widget.setROIs(rois)

    ''' Private methods '''

    def _add_mode(self, i: int):
        seg = self.AVAIL_MODES[i](self)
        btn = QRadioButton(seg.name())
        self._mode_btns.append(btn)
        self._bottom_layout.addWidget(btn)
        btn.clicked.connect(lambda: self._set_mode(i))
        # Listeners
        seg.propose.connect(self._propose)
        seg.add.connect(self._add)
        seg.cancel.connect(self._cancel)

    def _set_mode(self, i: int):
        self._mode = i

    def _update_settings(self, redraw: bool=True):
        self._show_options = self._options_box.isChecked()
        self._interpret_rgb = self._rgb_box.isChecked()
        self._chan = self._chan_slider.value()
        self._only_candidates = self._candidates_box.isChecked()
        if redraw:
            self.setImage(self._img)

    def _add_from_child(self, poly: PlanarPolygon):
        '''
        Process add() signal from child
        '''
        assert not self._is_prompting
        assert len(self._candidate_rois) == 0
        self._is_prompting = True
        self.AVAIL_MODES[self._mode].prompt(
            poly, self._show_options
        )

    def _delete_from_child(self, lbls: Set[int]):
        '''
        Process delete() signal from child
        '''
        pass

    def _propose(self, rois: List[ROI]):
        '''
        Show candidate ROIs at this level without committing them to the parent yet.
        '''
        pass

    def _add(self, rois: List[ROI]):
        '''
        add() from segmentor -> add() bubble up. This asynchronously processes the containee's request.
        '''
        pass

    def _cancel(self):
        pass