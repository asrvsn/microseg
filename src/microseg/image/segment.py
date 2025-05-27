'''
Image segmentor
'''
if __name__ == '__main__':
    import sys
    import argparse

    from microseg.widgets.roi_apps import *
    from microseg.utils.args import GuiArgumentParser

    parser = GuiArgumentParser()
    parser.add_argument('file', type=argparse.FileType('r'), help='Path to source img [tiff|jpg|png|czi|...]')
    parser.add_argument('-d', type=str, default='polygons', help='Descriptor')
    args = parser.parse_args()

    win = QtWidgets.QApplication(sys.argv)
    app = ImageSegmentorApp(args.file.name, desc=args.d)
    app.show()
    sys.exit(win.exec())