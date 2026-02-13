'''
Image viewer
'''
if __name__ == '__main__':
    import sys
    import argparse

    from microseg.widgets.roi_apps import *
    from microseg.utils.args import GuiArgumentParser

    parser = GuiArgumentParser(prog="Image Viewer")
    parser.add_argument('file', type=argparse.FileType('r'), help='Path to source img [tiff|jpg|png|czi|...]')
    args = parser.parse_args()

    win = QtWidgets.QApplication(sys.argv)
    app = ImageViewerApp(args.file.name)
    app.show()
    sys.exit(win.exec())