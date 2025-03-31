'''
Extract centroids of objects from 3D image
'''
import pickle
from microseg.widgets.roi_apps import *

class CentroidsExtractorApp(VolumeSegmentorApp):

    ''' Overrides '''
    
    def readData(self, path: str) -> Any:
        pass

    def writeData(self, path: str, data: Any):
        ''' Write centroids '''
        centroids = self._viewer._centroids
        if centroids is None:
            print('No centroids to write')
        else:
            np.savetxt(path, centroids)
            print(f'Wrote surface to {path}')


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to source img [tiff|jpg|png|czi|...]')
    parser.add_argument('-d', type=str, default='centroids', help='Descriptor')
    args = parser.parse_args()

    win = QtWidgets.QApplication(sys.argv)
    app = CentroidsExtractorApp(args.file, desc=args.d)
    app.show()
    sys.exit(win.exec())