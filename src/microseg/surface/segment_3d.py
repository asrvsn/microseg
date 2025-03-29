'''
3D z-stack segmentor
'''
if __name__ == '__main__':
    import sys
    import argparse

    from microseg.widgets.roi_apps import *

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to source img [tiff|jpg|png|czi|...]')
    parser.add_argument('-d', type=str, default='rois', help='Descriptor')
    args = parser.parse_args()

    win = QtWidgets.QApplication(sys.argv)
    app = ZStackSegmentorApp(args.file, desc=args.d)
    app.show()
    sys.exit(win.exec())