'''
Image + ROI overlays
'''

from .pg import *
from .rois import *

class ROIsImageWidget(ImagePlotWidget, metaclass=QtABCMeta):
    '''
    Editable widget for displaying and drawing ROIs on an image
    '''
    
class OldWidget:
    edited = QtCore.Signal()
    undo_n: int=100

    def __init__(self, editable: bool=False, **kwargs):
        super().__init__(**kwargs)
        self._editable = editable

        # State
        self._things: List[LabeledROI] = []
        self._items: List[SelectableROIItem] = []
        self._selected = []
        self._shortcuts = []
        self._vb = None
        self._reset_drawing_state()
        self._next_label: int = None
        self._undo_stack = None # Uninitialized
        self._redo_stack = None

        if editable:
            self.add_sc('Delete', lambda: self._delete())
            self.add_sc('Backspace', lambda: self._delete())
            self.add_sc('E', lambda: self._edit())
            self.add_sc('Ctrl+Z', lambda: self._undo())
            self.add_sc('Ctrl+Y', lambda: self._redo())
            self.add_sc('Ctrl+Shift+Z', lambda: self._redo())
        self.add_sc('Escape', lambda: self._escape())
        self.scene().sigMouseMoved.connect(self._mouse_move)

    def add_sc(self, key: str, fun: Callable):
        sc = QShortcut(QKeySequence(key), self)
        sc.activated.connect(fun)
        self._shortcuts.append(sc)

    def setImage(self, img: np.ndarray):
        self._img.setImage(img)

    @abc.abstractmethod
    def createItemFromThing(self, thing: LabeledROI) -> SelectableROIItem:
        pass

    @abc.abstractmethod
    def getThingFromItem(self, item: SelectableROIItem) -> LabeledROI:
        pass

    def setThings(self, things: List[LabeledROI], reset_stacks: bool=True):
        # Remove previous things
        self._selected = []
        self._things = things
        for item in self._items:
            self.removeItem(item)
        # Add new things
        self._items = []
        for i, thing in enumerate(things):
            item = self.createItemFromThing(thing)
            self.addItem(item)
            self._listenItem(i, item)
            self._items.append(item)
        if reset_stacks:
            self._undo_stack = [self.getThings()]
            self._redo_stack = []
        self._reset_drawing_state()

    def getThings(self) -> List[LabeledROI]:
        return [t.copy() for t in self._things]

    def _listenItem(self, i: int, item: SelectableROIItem):
        item.sigClicked.connect(lambda: self._select(i))

    def _select(self, i: Optional[int]):
        if i is None:
            self._unselect_all()
        if not i in self._selected:
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
            print(f'delete {len(self._selected)} things')
            for i in self._selected:
                self.removeItem(self._items[i])
            selset = set(self._selected)
            self._things = [t for i, t in enumerate(self._things) if not i in selset]
            self._items = [t for i, t in enumerate(self._items) if not i in selset]
            self._selected = []
            self._push_stack()

    def _edit(self):
        print('edit')
        if self._editable and not self._is_drawing:
            self._select(None)
            self._is_drawing = True
            assert not self._next_label is None, 'No next label'
            print('Next label:', self._next_label)
            self._drawn_lbl = self._next_label
            self._vb = self.getViewBox()
            self._vb.setMouseEnabled(x=False, y=False)

    def _escape(self):
        print('escape')
        if not self._drawn_item is None:
            self.removeItem(self._drawn_item)
        self._reset_drawing_state()
        self._select(None)

    @abc.abstractmethod
    def initDrawingState(self):
        ''' Reset the drawing state specific to this item '''
        pass

    def _reset_drawing_state(self):
        self.initDrawingState()
        self._is_drawing = False
        self._drawn_lbl = None
        self._drawn_item = None
        if not self._vb is None:
            self._vb.setMouseEnabled(x=True, y=True)
        self._vb = None

    @abc.abstractmethod
    def modifyDrawingState(self, pos):
        ''' Modify the drawing state from given pos specific to this item '''
        pass

    @abc.abstractmethod
    def finishDrawingState(self):
        ''' Finish the drawing state specific to this item '''
        pass

    def _mouse_move(self, pos):
        if self._is_drawing:
            if QtCore.Qt.LeftButton & QtWidgets.QApplication.mouseButtons():
                pos = self._vb.mapSceneToView(pos)
                pos = np.array([pos.x(), pos.y()])
                self.modifyDrawingState(pos)
            else:
                if not self._drawn_item is None:
                    print('ending draw')
                    self.finishDrawingState()
                    thing = self.getThingFromItem(self._drawn_item)
                    self._things.append(thing)
                    self._items.append(self._drawn_item)
                    N = len(self._things)-1
                    self._listenItem(N, self._drawn_item)
                    self._reset_drawing_state()
                    self._push_stack()
                    # self._select(N)

    def _push_stack(self):
        self._undo_stack.append(self.getThings())
        self._undo_stack = self._undo_stack[-self.undo_n:]
        self._redo_stack = []
        print(f'Current stacks: undo {len(self._undo_stack)}, redo {len(self._redo_stack)}')
        self.edited.emit()

    def _undo(self):
        print('undo')
        if len(self._undo_stack) > 1:
            self._redo_stack.append(self._undo_stack[-1])
            self._undo_stack = self._undo_stack[:-1]
            things = [t.copy() for t in self._undo_stack[-1]]
            self.setThings(things, reset_stacks=False)
            self.edited.emit()
        else:
            print('Cannot undo further')

    def _redo(self):
        print('redo')
        if len(self._redo_stack) > 0:
            self._undo_stack.append(self._redo_stack[-1])
            self._redo_stack = self._redo_stack[:-1]
            things = [t.copy() for t in self._undo_stack[-1]]
            self.setThings(things, reset_stacks=False)
            self.edited.emit()
        else:
            print('Cannot redo further')