'''
Project polygons onto ellipsoid in Z coordinate
'''

import os
import numpy as np
import pyqtgraph.opengl as gl
from scipy.spatial.distance import pdist
from qtpy.QtWidgets import QPushButton, QCheckBox
from qtpy.QtGui import QKeyEvent
from qtpy.QtCore import Qt
import pickle
from matgeo import Ellipsoid, Plane
import pyqtgraph as pg
from microseg.widgets.base import *
from microseg.widgets.pg_gl import *
from microseg.utils.colors import map_colors
from microseg.utils.pg import ppolygons_3d


class EllipsoidProjectorApp(SaveableApp):
    def __init__(self, polys_path: str, ell_path: str, *args, **kwargs):
        # State
        assert os.path.isfile(polys_path), f'{polys_path} is not a file'
        assert os.path.isfile(ell_path), f'{ell_path} is not a file'
        self._polys = pickle.load(open(polys_path, 'rb'))
        self._ell = pickle.load(open(ell_path, 'rb'))
        assert all(p.ndim == 2 for p in self._polys), 'All polygons must be 2D'
        assert self._ell.ndim == 3, 'Ellipsoid must be 3D'
        # Translate ellipsoid up so it's above the XY plane
        self._ell.v[2] = self._ell.get_radii().max() * 1.2

        # Widgets
        self._main = VLayoutWidget()
        self._vw = gl.GLViewWidget()
        self._vw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._main.addWidget(self._vw)
        self._ell_item = GLEllipsoidItem(ell=self._ell)
        self._vw.addItem(self._ell_item)
        self._vw.opts['center'] = pg.Vector(*self._ell.v)

        # Compute projection
        polys_XY = [p.embed_XY() for p in self._polys]
        XY_centers = Plane.XY().reverse_embed(np.array([p.centroid() for p in self._polys]))
        self._polys_proj = [self._ell.project_poly_z(p, mode='top') for p in self._polys]
        proj_centers = np.array([p.plane.reverse_embed(p.centroid()) for p in self._polys_proj])
        ppolygons_3d(self._vw, polys_XY, XY_centers)
        ppolygons_3d(self._vw, self._polys_proj, proj_centers)

        super().__init__(
            f'Projecting polygons onto ellipsoid',
            f'{polys_path}.projected',
            *args, **kwargs
        )
        self.setCentralWidget(self._main)

    def copyIntoState(self, state: Any):
        pass

    def copyFromState(self) -> Any:
        pass
    
    def readData(self, path: str) -> Any:
        pass

    def writeData(self, path: str, data: Any):
        pass


if __name__ == '__main__':
    import sys
    import argparse
    from microseg.utils.args import GuiArgumentParser

    parser = GuiArgumentParser(prog="Ellipsoid Projector")
    parser.add_argument('polys_path', type=argparse.FileType('r'), help='Path to polygons file (pickle format)')
    parser.add_argument('ell_path', type=argparse.FileType('r'), help='Path to ellipsoid file (pickle format)')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    window = EllipsoidProjectorApp(args.polys_path.name, args.ell_path.name)
    window.show()
    sys.exit(app.exec_())