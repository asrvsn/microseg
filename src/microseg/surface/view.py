'''
Surface viewer utility
'''
from qtpy.QtWidgets import QPushButton
from microseg.widgets.base import *
from microseg.widgets.pg_gl import *

class SurfaceViewerApp(MainWindow):
    def __init__(self, path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._path = path
        self._tri = pickle.load(open(path, 'rb'))

        # Widgets
        self._main = VLayoutWidget()
        self._vw = GLSelectableSurfaceViewWidget()
        self._main.addWidget(self._vw)
        self._settings = HLayoutWidget()
        self._main.addWidget(self._settings)
        self._vw.setTriangulation(self._tri)
        self.setWindowTitle(f'Surface viewer: {os.path.basename(path)}')

        self._flip_btn = QPushButton('Flip normals')
        self._main.addWidget(self._flip_btn)
        self._flip_btn.clicked.connect(self._flip_normals)

        self.setCentralWidget(self._main)

    def _flip_normals(self):
        self._tri.flip_orientation()
        self._vw.setTriangulation(self._tri)

if __name__ == '__main__':
    import argparse
    import sys
    from microseg.utils.args import GuiArgumentParser

    parser = GuiArgumentParser(prog="Surface Viewer")
    parser.add_argument('path', type=argparse.FileType('r'), help='Path to triangulation (pickle file)')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    window = SurfaceViewerApp(args.path.name)
    window.show()
    sys.exit(app.exec_())