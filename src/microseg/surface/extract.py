'''
Extract surface from 3D volume
'''
import pickle
from microseg.widgets.roi_apps import *

class SurfaceExtractorApp(VolumeSegmentorApp):

    ''' Overrides '''
    
    def readData(self, path: str) -> Any:
        pass

    def writeData(self, path: str, data: Any):
        ''' Write triangulation '''
        tri = self._viewer._tri
        if tri is None:
            print('No surface to write')
        else:
            pickle.dump(tri, open(path, 'wb'))
            print(f'Wrote surface to {path}')


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to source img [tiff|jpg|png|czi|...]')
    parser.add_argument('-d', type=str, default='surface', help='Descriptor')
    args = parser.parse_args()

    win = QtWidgets.QApplication(sys.argv)
    app = SurfaceExtractorApp(args.file, desc=args.d)
    app.show()
    sys.exit(win.exec())