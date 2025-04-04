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

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to source img [tiff|jpg|png|czi|...]')
    parser.add_argument('-d', type=str, default='vertices', help='Descriptor')
    args = parser.parse_args()

    win = QtWidgets.QApplication(sys.argv)
    app = VerticesExtractorApp(args.file, desc=args.d)
    app.show()
    sys.exit(win.exec())