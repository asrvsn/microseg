'''
Construct ellipsoidal surfaces from data
'''
import os
import numpy as np
import pyqtgraph.opengl as gl
from scipy.spatial.distance import pdist
from qtpy.QtWidgets import QPushButton, QCheckBox
from qtpy.QtGui import QKeyEvent
from qtpy.QtCore import Qt
import pickle
from matgeo import Ellipsoid
import pyqtgraph as pg
from microseg.widgets.base import *
from microseg.widgets.pg_gl import *
from microseg.utils.colors import map_colors
from microseg.utils.args import GuiArgumentParser

class EllipsoidConstructorApp(SaveableApp):
    def __init__(self, pts_path: str,*args, **kwargs):
        # State
        assert os.path.isfile(pts_path), f'{pts_path} is not a file'
        self._pts = np.loadtxt(pts_path)
        self._ell = Ellipsoid.fit_outer_iterative(self._pts, tol=1e-2)
        # self._center = self._pts.mean(axis=0)
        self._center = self._ell.v.copy()

        # Widgets
        self._main = VLayoutWidget()
        self._vw = gl.GLViewWidget()
        self._vw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._main.addWidget(self._vw)
        self._settings = HLayoutWidget()
        self._main.addWidget(self._settings)
        self._ell_item = GLEllipsoidItem()
        self._vw.addItem(self._ell_item)
        self._pts_item = gl.GLScatterPlotItem(pxMode=True)
        self._vw.addItem(self._pts_item)
        self._center_item = gl.GLScatterPlotItem(pxMode=True)
        self._vw.addItem(self._center_item)

        # XYZ sliders
        diam = pdist(self._pts).max()
        centmin, centmax = self._center - diam, self._center + diam
        self._sliders = [None] * 3
        for i in range(3):
            self._sliders[i] = FloatSlider(label=f'{["X", "Y", "Z"][i]}', step=0.05)
            self._sliders[i].setData(centmin[i], centmax[i], self._center[i])
            self._settings.addWidget(self._sliders[i])
            self._sliders[i].valueChanged.connect(self._update_center)
        self._scale_sld = FloatSlider(label='Scale', step=0.02)
        self._scale_sld.setData(1, 2, 1)
        self._settings.addWidget(self._scale_sld)

        self._settings.addStretch()
        self._fit_btn = QPushButton('Fit')
        self._settings.addWidget(self._fit_btn)
        self._fit_btn.clicked.connect(self._fit)

        center = self._pts.mean(axis=0)
        viewsize = np.linalg.norm(self._pts - center, axis=1).max()
        self._vw.setCameraPosition(distance=1.3 * viewsize)

        self._pts_box = QCheckBox('Points')
        self._pts_box.setChecked(True)
        self._settings.addWidget(self._pts_box)
        self._pts_box.clicked.connect(self._redraw)
        self._ell_box = QCheckBox('Ellipsoid')
        self._settings.addWidget(self._ell_box)
        self._ell_box.setChecked(True)
        self._ell_box.clicked.connect(self._redraw)

        
        super().__init__(
            f'Constructing ellipsoid from {os.path.basename(pts_path)}', 
            f'{os.path.splitext(pts_path)[0]}.ellipsoid',
            *args, **kwargs
        )
        self.setCentralWidget(self._main)
        self._redraw()

    def copyIntoState(self, state: Any):
        pass

    def copyFromState(self) -> Any:
        pass
    
    def readData(self, path: str) -> Any:
        # TODO: hack using this to load the ellipsoid
        self._ell = pickle.load(open(path, 'rb'))
        self._center = self._ell.v

    def writeData(self, path: str, data: Any):
        if self._ell is None:
            print('No ellipsoid to save')
        else:
            pickle.dump(self._ell, open(path, 'wb'))
            print(f'Saved ellipsoid to {path}')

    def _redraw(self):
        if self._pts_box.isChecked():
            self._pts_item.setData(pos=self._pts)
        else:
            self._pts_item.setData(pos=[])
        if self._ell_box.isChecked():
            self._ell_item.setData(ell=self._ell)
        else:
            self._ell_item.setData(ell=None)
        self._redraw_center()

    def _redraw_center(self):
        self._center_item.setData(pos=np.array([self._center]), size=10, color=(0,1,0,1))
        self._vw.opts['center'] = pg.Vector(*self._center)

    def _update_center(self):
        self._center = np.array([slider.value() for slider in self._sliders])
        self._redraw_center()

    def _fit(self):
        # self._ell = Ellipsoid.fit(self._pts, self._center) * self._scale_sld.value()
        # Mirror image points about plane at self.center[2]
        pts_mirror = self._pts.copy()
        pts_mirror[:, 2] = 2 * self._center[2] - pts_mirror[:, 2]
        pts = np.concatenate([self._pts, pts_mirror])
        self._ell = Ellipsoid.fit_outer_iterative(pts, tol=1e-2)
        self._redraw()

if __name__ == '__main__':
    import sys
    import argparse

    parser = GuiArgumentParser(prog="Ellipsoid Constructor")
    parser.add_argument('file', type=argparse.FileType('r'), help='Path to source points file (numpy format)')
    parser.add_argument('--ignore-existing', action='store_true', help='Ignore previously registered points')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    window = EllipsoidConstructorApp(args.file.name, ignore_existing=args.ignore_existing)
    window.show()
    sys.exit(app.exec_())