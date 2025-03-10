'''
General PyQT widgets
'''
from typing import List, Tuple, Union, Any
import abc
import os
import pickle
import math
import numpy as np
import pyqtgraph.opengl as gl # Has to be imported before qtpy
from qtpy import QtCore
from qtpy import QtGui, QtWidgets
from qtpy.QtCore import Qt, QTimer, QObject
from qtpy.QtWidgets import QApplication, QFileSystemModel, QHeaderView, QLabel, QSizePolicy, QTableWidget, QTreeView, QVBoxLayout, QWidget, QGraphicsOpacityEffect, QSlider, QScrollBar, QAction, QMessageBox
from qtpy.QtGui import QKeySequence, QShortcut, QGuiApplication, QCursor
from qtpy.QtGui import QImage, QPixmap
from superqt import QRangeSlider

from .layout import *

''' Metaclasses '''

class QtABCMeta(type(QtCore.QObject), abc.ABCMeta):
    pass

class ClickProxy(QObject):
    sigClicked = QtCore.Signal()

''' Parent classes '''

class PaneledWidget(QWidget):
    '''
    Widget with bottom panel
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        self._layout = VLayout()
        self.setLayout(self._layout)
        self._main_layout = HLayout()
        self._main_widget = QWidget()
        self._main_widget.setLayout(self._main_layout)
        self._layout.addWidget(self._main_widget)
        self._bottom_layout = HLayout()
        self._bottom_widget = QWidget()
        self._bottom_widget.setLayout(self._bottom_layout)
        self._layout.addWidget(self._bottom_widget)

class SaveableWidget(PaneledWidget, metaclass=QtABCMeta):
    '''
    Basic panel widget containing a top and bottom layout, with settings on the bottom and a save button with Ctrl+S shortcut
    '''
    saved = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        self._settings_layout = HLayout()
        self._settings_widget = QWidget()
        self._settings_widget.setLayout(self._settings_layout)
        self._bottom_layout.addWidget(self._settings_widget)
        self._save_btn = PushButton('Save')
        self._bottom_layout.addWidget(self._save_btn)
        self.setDisabled(True) # Initially disabled

        # Listeners
        self._save_btn.clicked.connect(self._on_save)

    def _on_save(self):
        self.saved.emit()

    @abc.abstractmethod
    def getData(self):
        '''
        Get data on save
        '''
        pass

    def keyPressEvent(self, ev):
        # Check for Ctrl+Save event
        if ev.key() == QtCore.Qt.Key_S and ev.modifiers() & QtCore.Qt.ControlModifier:
            self._on_save()

''' Widgets '''

class PushButton(QtWidgets.QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setContentsMargins(0, 0, 0, 0)
        # self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resizeToScreen()
        # pg.setConfigOptions(antialias=True, useOpenGL=False)
        
    def resizeToScreen(self, offset=0):
        screens = QApplication.instance().screens()
        cursor_screen = QGuiApplication.screenAt(QCursor.pos())
        target_screen = screens[(screens.index(cursor_screen) + offset) % len(screens)]
        if target_screen is None:
            # Fallback to primary screen if no screen is detected
            target_screen = QApplication.primaryScreen()
        # Move and resize the window to fit the active screen
        self.move(target_screen.geometry().topLeft())
        self.resize(target_screen.geometry().width(), target_screen.geometry().height())

class SaveableApp(MainWindow, metaclass=QtABCMeta):
    '''
    Saveable application with action menu item, warn on close, undo/redo
    '''
    undo_n: int=100
    redo_n: int=100

    def __init__(self, title: str, save_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._title = title
        self._save_path = save_path
        self._mark_edited(False)

        # Load existing data if exists
        if os.path.isfile(self._save_path):
            print(f'Loading existing data from {self._save_path}')
            self.copyIntoState(self.readData(self._save_path))

        # Add Save menu item
        mbar = self.menuBar()
        fmenu = mbar.addMenu('File')
        # oaction = QAction('Open', self)
        # oaction.setShortcut(QtGui.QKeySequence('Ctrl+O'))
        # oaction.triggered.connect(self._open)
        # fmenu.addAction(oaction)
        saction = QAction('Save', self)
        saction.setShortcut(QtGui.QKeySequence('Ctrl+S'))
        saction.triggered.connect(self._save)
        fmenu.addAction(saction)

        # Listeners
        undo_sc = QShortcut(QKeySequence('Ctrl+Z'), self)
        undo_sc.activated.connect(self._undo)
        redo_sc = QShortcut(QKeySequence('Ctrl+Y'), self)
        redo_sc.activated.connect(self._redo)
        redo_sc_ = QShortcut(QKeySequence('Ctrl+Shift+Z'), self)
        redo_sc_.activated.connect(self._redo)

        # State
        self._undo_stack: List[Any] = []
        self._redo_stack: List[Any] = []
        self.pushEdit(mark=False) # Push initial state onto undo stack

    ''' API methods '''

    def pushEdit(self, mark: bool=True):
        self._undo_stack.append(self.copyFromState())
        self._undo_stack = self._undo_stack[-self.undo_n:]
        self._redo_stack = []
        self._mark_edited(mark)

    ''' Abstract methods '''

    @abc.abstractmethod
    def copyIntoState(self, state: Any):
        '''
        COPY and load data into state
        '''
        pass

    @abc.abstractmethod
    def copyFromState(self) -> Any:
        '''
        COPY and load data from state
        '''
        pass

    @abc.abstractmethod
    def readData(path: str) -> Any:
        pass

    @abc.abstractmethod
    def writeData(path: str, data: Any):
        pass

    ''' Private listener methods '''

    def _mark_edited(self, bit: bool):
        '''
        Specify when an edit is made
        '''
        self._is_edited = bit
        title = f'{self._title} (edited)' if bit else self._title
        self.setWindowTitle(title)

    def _save(self):
        print(f'Saving data to {self._save_path}')
        self.writeData(self._save_path, self.copyFromState())
        self._mark_edited(False)

    def _undo(self):
        print('undo')
        if len(self._undo_stack) > 1:
            self._redo_stack.append(self._undo_stack[-1])
            self._undo_stack = self._undo_stack[:-1]
            self.copyIntoState(self._undo_stack[-1])
            self._mark_edited(True)
        else:
            print('Cannot undo further')

    def _redo(self):
        print('redo')
        if len(self._redo_stack) > 0:
            self._undo_stack.append(self._redo_stack[-1])
            self._redo_stack = self._redo_stack[:-1]
            self.copyIntoState(self._undo_stack[-1])
            self._mark_edited(True)
        else:
            print('Cannot redo further')

    # def closeEvent(self, event):
    #     if self._is_edited:
    #         reply = QMessageBox.question(self, 'Unsaved changes', 'You have unsaved changes. Do you want to proceed without saving?', QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel)
    #         if reply == QMessageBox.Ok:
    #             event.accept()
    #         else:
    #             event.ignore()
    #     else:
    #         event.accept()

class TreeViewLeafSelectable(QTreeView):
    ''' Version of QTreeView in which only leaf nodes are selectable '''
    def selectionCommand(self, index, event=None):
        if index.isValid() and not index.model().hasChildren(index):
            return QTreeView.selectionCommand(self, index, event)
        
class ExtensionViewer(QWidget):
    file_selected = QtCore.Signal(str)
    
    def __init__(self, folder_path, name_filters: List[str]=['*.*'], show_filtered: bool=False, parent=None):
        super(ExtensionViewer, self).__init__(parent)

        self.folder_path = folder_path

        layout = QVBoxLayout(self)

        model = QFileSystemModel()
        model.setRootPath(self.folder_path)
        model.setNameFilters(name_filters)
        model.setNameFilterDisables(show_filtered)

        tree_view = TreeViewLeafSelectable()
        tree_view.setModel(model)
        tree_view.setRootIndex(model.index(self.folder_path))
        tree_view.setSortingEnabled(True)
        
        # Show only name column
        for i in range(1, tree_view.model().columnCount()):
            tree_view.hideColumn(i)

        layout.addWidget(tree_view)
        self.setLayout(layout)

        tree_view.selectionModel().selectionChanged.connect(self.on_selection_changed)

    def on_selection_changed(self, selected):
        for ix in selected.indexes():
            if ix.column() == 0:
                path = self.sender().model().filePath(ix)
                self.file_selected.emit(path)

class StretchTableWidget(QTableWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMinimumSize(1, 1)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.setShowGrid(False)
        self.setAlternatingRowColors(True)
        # # Set vertical size to minimum
        # self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        # Remove column and row labels
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setVisible(False)

class FlashOverlay(QWidget):
    def __init__(self, parent):
        super(FlashOverlay, self).__init__(parent)
        self._parent = parent
        self.setWindowFlags(Qt.WindowType.WindowTransparentForInput | Qt.WindowType.WindowStaysOnTopHint)
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.fade_out)

        # Add centered text
        self.text = QLabel(self, text='Saved')
        self.text.setAlignment(Qt.AlignCenter)

    def flash(self):
        self.setGeometry(self._parent.geometry())
        self.setWindowOpacity(1.0)
        self.timer.start(50)

    def fade_out(self):
        opacity = self.windowOpacity()
        if opacity < 0.0:
            self.setWindowOpacity(opacity - 0.1)
        else:
            self.setWindowOpacity(0)
            self.timer.stop()

class IntegerSlider(HLayoutWidget):
    '''
    Integer slider with single handle
    '''
    def __init__(self, *args, label: str=None, mode: str='slide', step_size=1, **kwargs):
        super().__init__(*args, **kwargs)
        if not (label is None):
            self._layout.addWidget(QLabel(label))
        if mode == 'slide':
            self._slider = QSlider()
            self._slider.setTickPosition(QSlider.TicksBelow)
            self._slider.setTickInterval(1)
        elif mode == 'scroll':
            self._slider = QScrollBar()
        self._layout.addWidget(self._slider)
        self._label = QLabel()
        self._layout.addWidget(self._label)
        self._slider.setOrientation(QtCore.Qt.Horizontal)
        self._slider.setSingleStep(step_size)
        self._slider.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self._slider.valueChanged.connect(lambda x: self._label.setText(f'{x}/{self._max}'))
        self.valueChanged = self._slider.valueChanged
        self.value = self._slider.value

    def setData(self, min: int, max: int, x: int):
        assert min <= x <= max, 'x must be within min and max'
        self._min = min
        self._max = max
        self._slider.setMinimum(min)
        self._slider.setMaximum(max)
        self.setValue(x)

    def setValue(self, x: int):
        assert self._min <= x <= self._max, 'x must be within min and max'
        self._slider.setValue(x)
        self._label.setText(f'{x}/{self._max}')

class FloatSlider(HLayoutWidget):
    '''
    Float slider with single handle
    '''
    valueChanged = QtCore.Signal(float)

    def __init__(self, *args, label: str=None, step: float=1., **kwargs):
        super().__init__(*args, **kwargs)
        assert step > 0
        self._step = step
        self._prec = -int(math.log10(step - math.floor(step)))
        if not (label is None):
            self._layout.addWidget(QLabel(label))
        self._slider = QScrollBar()
        self._layout.addWidget(self._slider)
        self._label = QLabel()
        self._layout.addWidget(self._label)
        self._slider.setOrientation(QtCore.Qt.Horizontal)
        self._slider.setSingleStep(1)
        self._slider.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self._slider.valueChanged.connect(self._value_changed)

    def setData(self, min: float, max: float, x: float):
        assert min <= x <= max, 'x must be within min and max'
        self._slider.setMinimum(math.floor(min / self._step))
        self._slider.setMaximum(math.ceil(max / self._step))
        self.setValue(x)

    def setValue(self, x: float):
        self._slider.setValue(round(x / self._step))
        self._label.setText(f'{x:.{self._prec}f}')

    def value(self) -> float:
        return float(self._step * self._slider.value())

    def _value_changed(self, x: int):
        y = float(x * self._step)
        self._label.setText(f'{y:.{self._prec}f}')
        self.valueChanged.emit(y)

class IntegerRangeSlider(HLayoutWidget):
    '''
    Integer slider with two handles
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slider = QRangeSlider()
        self._layout.addWidget(self._slider)
        self._label = QLabel()
        self._layout.addWidget(self._label)
        self._slider.setOrientation(QtCore.Qt.Horizontal)
        self._slider.setTickPosition(QSlider.TicksBelow)
        self._slider.setTickInterval(1)
        self._slider.setSingleStep(1)
        self._slider.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self._slider.valueChanged.connect(lambda x,y: self._label.setText(f'{x}-{y}/{self._max}'))
        self.valueChanged = self._slider.valueChanged

    def setData(self, min: int, max: int, range: Tuple[int, int]):
        x, y = range
        assert min <= x <= y <= max, 'range must be within min and max'
        self._max = max
        self._slider.setMinimum(min)
        self._slider.setMaximum(max)
        self._slider.setValue(range)
        self._label.setText(f'{x}-{y}/{max}')

class QImageWidget(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setScaledContents(True)
        self.setAlignment(Qt.AlignCenter)

    def setImage(self, img: np.ndarray):
        if img.dtype == np.uint8:
            pass
        elif img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        elif img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        else:
            raise ValueError(f'Invalid image dtype: {img.dtype}')
        img_bytes = img.data.tobytes()
        h, w = img.shape[:2]
        if img.ndim == 2:
            qImage = QImage(img_bytes, w, h, w, QImage.Format_Grayscale8)
        elif img.ndim == 3 and img.shape[2] == 3:
            qImage = QImage(img_bytes, w, h, 3 * w, QImage.Format_RGB888)
        else:
            raise ValueError('Invalid image shape')
        self.setPixmap(QPixmap.fromImage(qImage))