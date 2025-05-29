'''
Extract vertices of objects from 3D image
'''
import pickle
from microseg.widgets.roi_apps import *

class VerticesExtractorApp(VolumeSegmentorApp):

    ''' Overrides '''
    
    def readData(self, path: str) -> Any:
        pass

    def writeData(self, path: str, data: Any):
        ''' Write vertices '''
        vertices = self._viewer._vertices
        if vertices is None:
            print('No vertices to write')
        else:
            np.savetxt(path, vertices)
            print(f'Wrote vertices to {path}')


if __name__ == '__main__':
    import sys
    import argparse
    from microseg.utils.args import GuiArgumentParser

    parser = GuiArgumentParser(prog="Voxel Vertex Extractor")
    parser.add_argument('file', type=argparse.FileType('r'), help='Path to source img [tiff|jpg|png|czi|...]')
    parser.add_argument('-d', type=str, default='vertices', help='Descriptor')
    parser.add_argument('-b', type=str, default='boundary', help='Boundary descriptor')
    parser.add_argument('-t', '--transpose-xy', action='store_true', help='Transpose x and y coordinates')
    args = parser.parse_args()

    win = QtWidgets.QApplication(sys.argv)
    app = VerticesExtractorApp(args.file.name, desc=args.d, boundary_desc=args.b, transpose_xy=args.transpose_xy)
    app.show()
    sys.exit(win.exec())