'''
Boundary drawing (for drawing a single polygon on image or z-stack) app
'''
from .widgets.roi_apps import *

class BoundaryDrawerApp(ImageSegmentorApp):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('Boundary Drawer')
        self.setCentralWidget(self._main)

    def readData(self, path: str) -> List[List[LabeledROI]]:
        poly = pickle.load(open(path, 'rb'))
        assert isinstance(poly, PlanarPolygon)
        return [[LabeledROI(0, poly)] for _ in range(self._zmax)]

    def writeData(self, path: str, rois: List[List[LabeledROI]]):
        assert all(len(ls) == len(rois[0]) for ls in rois), 'All z-slices must have the same number of ROIs'
        if len(rois[0]) == 0:
            print(f'Nothing to save')
        elif len(rois[0]) == 1:
            pickle.dump(rois[0][0].roi, open(path, 'wb'))
        else:
            raise ValueError(f'Expected 1 ROI per z-slice, got {len(rois[0])}')
        
    def _add(self, rois: List[ROI]):
        assert len(rois) == 1, 'Only one ROI can be added at a time'
        l = self.next_label
        lroi = LabeledROI(l, rois[0])
        for z in range(self._zmax):
            self._rois[z].append(lroi)
        self.refreshROIs(push=True)

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to source img [tiff|jpg|png|czi|...]')
    parser.add_argument('-d', type=str, default='boundary', help='Descriptor')
    args = parser.parse_args()

    win = QtWidgets.QApplication(sys.argv)
    app = BoundaryDrawerApp(args.file, desc=args.d)
    app.show()
    sys.exit(win.exec())