'''
Widget to register points in 3d
'''
import os
import numpy as np
import pyqtgraph.opengl as gl
from scipy.spatial.distance import pdist
from qtpy.QtWidgets import QPushButton, QCheckBox
from qtpy.QtGui import QKeyEvent
from qtpy.QtCore import Qt

from matgeo import Triangulation

from microseg.widgets.base import *
from microseg.widgets.pg_gl import *
from microseg.widgets.seg_2d import *
from microseg.utils.colors import map_colors

class SurfaceConstructorApp2(SaveableApp):
    undo_n: int = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        self._vw = GLSelectableSurfaceViewWidget()
        self._main_layout.addWidget(self._vw)
        self._dist_lbl = QLabel(f'Distance: -')
        self._settings_layout.addWidget(self._dist_lbl)
        self._norm_btn = QPushButton('Toggle normals')
        self._settings_layout.addWidget(self._norm_btn)

        # Listeners
        self._c_sc = QShortcut(QKeySequence('C'), self._vw)
        self._c_sc.activated.connect(self._collapse_labels)
        self._u_sc = QShortcut(QKeySequence('Ctrl+Z'), self._vw)
        self._u_sc.activated.connect(self._undo)
        self._r_sc = QShortcut(QKeySequence('Ctrl+Shift+Z'), self._vw)
        self._r_sc.activated.connect(self._redo)
        self._vw.selectionChanged.connect(self._selection_changed)
        self._norm_btn.clicked.connect(self._vw.toggleNormals)

        # State
        self._tri = None
        self._undo_stack = []
        self._redo_stack = []

    def setData(self, pts: np.ndarray):
        self.setDisabled(False)
        self._redraw(pts)
        self._undo_stack.append(self._tri.copy())
        self._redo_stack = []

    def getData(self) -> np.ndarray:
        return self._tri.pts.copy()

    def _redraw(self, pts: np.ndarray):
        tri = Triangulation.surface_3d(pts, method='advancing_front')
        if self._tri is None:
        # If the first time, pick a consistent orientation for sanity
            tri.orient_outward(tri.pts.mean(axis=0))
        # If an old one exists, try to match its orientation with respect to the camera position
        else:
            tri.match_orientation(self._tri)
        self._tri = tri
        self._vw.setTriangulation(self._tri)

    def _collapse_labels(self):
        if not (self._tri is None):
            sel = list(self._vw._selected)
            if len(sel) > 1:
                print(f'Collapsing {sel}')
                pts_unsel = np.delete(self._tri.pts, sel, axis=0)
                pt = self._tri.pts[sel].mean(axis=0)
                pts = np.vstack([pts_unsel, pt])
                self.setData(pts)

    def _selection_changed(self, sel: np.ndarray):
        if len(sel) == 2:
            r = np.linalg.norm(self._tri.pts[sel[0]] - self._tri.pts[sel[1]])
            self._dist_lbl.setText(f'Distance: {r:.2f}')
        else:
            self._dist_lbl.setText(f'Distance: -')

    def _undo(self):
        if len(self._undo_stack) > 1:
            print('undo')
            self._redo_stack.append(self._undo_stack[-1])
            self._undo_stack = self._undo_stack[:-1]
            self._redraw(self._undo_stack[-1].pts)
        else:
            print('Cannot undo further')

    def _redo(self):
        if len(self._redo_stack) > 0:
            print('redo')
            self._undo_stack.append(self._redo_stack[-1])
            self._redo_stack = self._redo_stack[:-1]
            self._redraw(self._undo_stack[-1].pts)
        else:
            print('Cannot redo further')

class Register3DWindow(MainWindow):
    def __init__(self, path: str, *args, ignore_existing=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._path = path
        assert os.path.isfile(path), f'{path} is not a file'
        fname, ext = os.path.splitext(path)
        assert ext in ['.txt', '.csv']
        self._path_new = Register3DWindow.make_path(fname)

        if os.path.isfile(self._path_new) and (not ignore_existing):
            print(f'Found previously registered points, using: {self._path_new}')
            pts = np.loadtxt(self._path_new)
        else:
            pts = np.loadtxt(path)

        assert pts.shape[1] == 3, 'Wrong number of columns'
        print(f'Loaded data of shape: {pts.shape}')

        # Widgets
        self.setWindowTitle('Register 3D')
        self._reg = SurfaceConstructorApp()
        self.setCentralWidget(self._reg)
        self._reg.setData(pts)
        self.resizeToActiveScreen()

        # Listeners
        self._reg.saved.connect(self._save)

    def _save(self):
        pts = self._reg.getData()
        assert not pts is None
        print(f'Saving {len(pts)} points to: {self._path_new}')
        np.savetxt(self._path_new, pts)

    @staticmethod
    def make_path(name: str) -> str:
        return f'{name}.registered.txt'
    
class SurfaceConstructorApp(SaveableApp):
    tri_methods = [
        'advancing_front',
        'chull',
    ]
    subdiv_methods = [
        'modified_butterfly',
        'catmull_clark',
        'loop',
    ]

    def __init__(self, pts_path: str, *args, **kwargs):
        # State
        assert os.path.isfile(pts_path), f'{pts_path} is not a file'
        self._pts = np.loadtxt(pts_path)
        self._tri = None
        self._sel = []

        # Widgets
        self._main = VLayoutWidget()
        self._vw = GLSelectableSurfaceViewWidget()
        self._vw.selectionChanged.connect(lambda sel: self._on_sel_change(sel))
        self._main.addWidget(self._vw)
        self._settings = HLayoutWidget()
        self._main.addWidget(self._settings)

        center = self._pts.mean(axis=0)
        # self._vw.opts['center'] = pg.Vector(*center)
        viewsize = np.linalg.norm(self._pts - center, axis=1).max()
        self._vw.setCameraPosition(distance=1.3 * viewsize)

        self._settings.addWidget(QLabel('Method:'))
        self._method_cb = QComboBox()
        self._method_cb.addItems(self.tri_methods)
        self._settings.addWidget(self._method_cb)
        self._method_cb.currentIndexChanged.connect(self._recompute_tri)
        self._merge_btn = QPushButton('Merge nodes')
        self._settings.addWidget(self._merge_btn)
        self._merge_btn.setEnabled(False)
        self._merge_btn.clicked.connect(self._merge_nodes)
        self._delete_nodes_btn = QPushButton('Delete nodes')
        self._settings.addWidget(self._delete_nodes_btn)
        self._delete_nodes_btn.setEnabled(False)
        self._delete_nodes_btn.clicked.connect(self._delete_nodes)
        self._delete_edge_btn = QPushButton('Delete edge')
        self._settings.addWidget(self._delete_edge_btn)
        self._delete_edge_btn.setEnabled(False)
        self._delete_edge_btn.clicked.connect(self._delete_edge)
        self._flip_btn = QPushButton('Flip normals')
        self._settings.addWidget(self._flip_btn)
        self._flip_btn.clicked.connect(self._flip_normals)
        # self._settings.addWidget(QLabel('Rescale:'))
        # self._rescale_sb = QDoubleSpinBox(minimum=0.0, value=1.0)
        # self._settings.addWidget(self._rescale_sb)
        # self._rescale_sb.valueChanged.connect(self._rescale)
        # self._settings.addWidget(QLabel('Subdivide:'))
        # self._subdiv_cb = QComboBox()
        # self._subdiv_cb.addItems(self.subdiv_methods)
        # self._settings.addWidget(self._subdiv_cb)
        # self._subdiv_cb.currentIndexChanged.connect(self._subdivide)
        # self._subdiv_n = QSpinBox(minimum=0, maximum=10, value=0)
        # self._settings.addWidget(self._subdiv_n)
        # self._subdiv_n.valueChanged.connect(self._subdivide)

        self._settings.addStretch()

        self._pts_box = QCheckBox('Points')
        self._settings.addWidget(self._pts_box)
        self._pts_box.clicked.connect(self._vw.togglePoints)
        self._norm_box = QCheckBox('Normals')
        self._settings.addWidget(self._norm_box)
        self._norm_box.clicked.connect(self._vw.toggleNormals)
        self._dark_box = QCheckBox('Dark')
        self._settings.addWidget(self._dark_box)
        self._dark_box.clicked.connect(self._toggle_dark)
        self._dist_lbl = QLabel(f'Distance: -')
        self._settings.addWidget(self._dist_lbl)

        self._recompute_tri(push=False)

        super().__init__(
            f'Constructing surface from {os.path.basename(pts_path)}', 
            f'{os.path.splitext(pts_path)[0]}.surface',
            *args, **kwargs
        )
        self.setCentralWidget(self._main)

        # Install event filter to capture key events
        self._vw.installEventFilter(self)

    ''' Overrides '''

    def eventFilter(self, obj, event):
        if obj is self._vw and event.type() == event.KeyPress:
            self.keyPressEvent(event)
            return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event: QKeyEvent):
        print(event.key())
        if event.key() == Qt.Key_M:
            self._merge_nodes()
        elif event.key() == Qt.Key_D:
            self._delete_nodes()
        elif event.key() == Qt.Key_E:
            self._delete_edge()
        else:
            super().keyPressEvent(event)

    def copyIntoState(self, tri: Triangulation):
        self._pts = tri.pts
        self._tri = tri
        self._vw.setTriangulation(tri)

    def copyFromState(self) -> Triangulation:
        return self._tri.copy()
    
    def readData(self, path: str) -> Triangulation:
        return pickle.load(open(path, 'rb'))
    
    def writeData(self, path: str, tri: Triangulation):
        pickle.dump(tri, open(path, 'wb'))

    ''' Privates '''

    def _on_sel_change(self, sel: list):
        self._sel = sel

        if len(sel) < 2:
            self._merge_btn.setEnabled(False)
        else:
            self._merge_btn.setEnabled(True)

        if len(sel) == 2:
            self._delete_edge_btn.setEnabled(True)
        else:
            self._delete_edge_btn.setEnabled(False)

        if len(sel) >= 1:
            self._delete_nodes_btn.setEnabled(True)
        else:
            self._delete_nodes_btn.setEnabled(False)

    def _merge_nodes(self):
        if len(self._sel) >= 2:
            pts = np.delete(self._pts, self._sel, axis=0)
            pt = self._pts[self._sel].mean(axis=0)
            self._pts = np.vstack([pts, pt])
            self._recompute_tri(push=True)  
        else:
            print('Need at least 2 points to merge')

    def _delete_nodes(self):
        if len(self._sel) == 0:
            print('No nodes to delete')
        else:
            self._pts = np.delete(self._pts, self._sel, axis=0)
            self._recompute_tri(push=True)

    def _delete_edge(self):
        if len(self._sel) == 2:
            tri = self._tri.remove_edge(self._sel[0], self._sel[1])
            self.copyIntoState(tri)
            self.pushEdit()
        else:
            print('Need exactly 2 points to delete edge')

    def _flip_normals(self):
        if self._tri is not None:
            self._tri.flip_orientation()
            self.copyIntoState(self._tri)

    def _rescale(self):
        scale = self._rescale_sb.value()
        tri = self._tri.rescale(scale)
        self.copyIntoState(tri)
        self.pushEdit()

    def _subdivide(self):
        method = self._subdiv_cb.currentText()
        n = self._subdiv_n.value()
        tri = self._tri.subdivide(n, mode=method)
        self.copyIntoState(tri)
        self.pushEdit()

    def _recompute_tri(self, push=True):
        method = self._method_cb.currentText()
        tri = Triangulation.surface_3d(self._pts, method=method)
        # tri.orient_outward(tri.pts.mean(axis=0))
        if not self._tri is None:
            tri.match_orientation(self._tri)
        self.copyIntoState(tri)
        if push:
            self.pushEdit()

    def _toggle_dark(self):
        if self._dark_box.isChecked():
            pg.setConfigOption('background', 'k')
            pg.setConfigOption('foreground', 'w')
        else:
            pg.setConfigOption('background', 'w')
            pg.setConfigOption('foreground', 'k')

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to source points file (numpy format)')
    parser.add_argument('--ignore-existing', action='store_true', help='Ignore previously registered points')
    args = parser.parse_args()

    # pg.setConfigOption('background', 'w')
    # pg.setConfigOption('foreground', 'k')

    app = QtWidgets.QApplication(sys.argv)
    window = SurfaceConstructorApp(args.file, ignore_existing=args.ignore_existing)
    window.show()
    sys.exit(app.exec_())