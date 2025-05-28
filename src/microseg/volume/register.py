'''
Perform pairwise registration of two z-stacks using ITK / elastix.
'''
from qtpy.QtWidgets import QLabel, QComboBox, QPushButton
from microseg.widgets.roi_apps import *
from microseg.utils.data import load_stack, get_voxel_size, get_view_transform

class RegistrationApp(MainWindow):
    def __init__(self, file1: str, file2: str, *args, affine1: Optional[str]=None, affine2: Optional[str]=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._main = PaneledWidget()
        self.setCentralWidget(self._main)
        
        # Create two side-by-side image plot widgets
        self._imw1 = ImagePlotWidget()
        self._imw2 = ImagePlotWidget()
        self._main._main_layout.addWidget(self._imw1)
        self._main._main_layout.addWidget(self._imw2)
        
        # Registration options - fixed width groupbox on the right
        self._reg_options = VGroupBox('Registration settings')
        self._reg_options.setFixedWidth(250)  # Set fixed width
        self._main._main_layout.addWidget(self._reg_options)
        
        # Add some placeholder registration controls
        self._reg_options.addWidget(QLabel('Registration method:'))
        self._reg_method = QComboBox()
        self._reg_method.addItems(['Rigid', 'Affine', 'Deformable'])
        self._reg_options.addWidget(self._reg_method)
        self._reg_options.addSpacing(10)
        
        self._reg_options.addWidget(QLabel('Metric:'))
        self._reg_metric = QComboBox()
        self._reg_metric.addItems(['Mutual Information', 'Mean Squares', 'Correlation'])
        self._reg_options.addWidget(self._reg_metric)
        self._reg_options.addSpacing(10)
        
        self._run_reg_btn = QPushButton('Run Registration')
        self._reg_options.addWidget(self._run_reg_btn)
        self._reg_options.addStretch()  # Push everything to the top
        
        # Load and display the images
        self._im1 = load_stack(file1, fmt='ZXY')
        self._im2 = load_stack(file2, fmt='ZXY')
        self._vox1 = get_voxel_size(file1)
        self._vox2 = get_voxel_size(file2)
        self._aff1 = get_view_transform(affine1) if affine1 else None
        self._aff2 = get_view_transform(affine2) if affine2 else None
        self._imr1 = None # registered image 1
        self._imr2 = None # registered image 2

        # View options
        self._z_slider = IntegerSlider(mode='scroll')
        self._main._bottom_layout.addWidget(self._z_slider)
        self._z_slider.setEnabled(False)
        self._z_slider.valueChanged.connect(self._on_z_slider_changed)

        # Registration options
        self._reg_options = VGroupBox('Registration settings')
        self._reg_options.setFixedWidth(250)
        self._main._main_layout.addWidget(self._reg_options)
        self._reg_options.addStretch()
        self._reg_btn = QPushButton('Run Registration')
        self._reg_options.addWidget(self._reg_btn)
        self._reg_btn.clicked.connect(self._on_reg_btn_clicked)

    def _on_reg_btn_clicked(self):

        

if __name__ == '__main__':
    import sys
    import argparse
    from microseg.utils.args import GuiArgumentParser

    parser = GuiArgumentParser()
    parser.add_argument('file1', type=argparse.FileType('r'), help='Path to source img 1 [tiff|jpg|png|czi|...]')
    parser.add_argument('file2', type=argparse.FileType('r'), help='Path to source img 2 [tiff|jpg|png|czi|...]')
    parser.add_argument('--affine1', type=argparse.FileType('r'), help='Path to affine matrix 1 [txt|...]')
    parser.add_argument('--affine2', type=argparse.FileType('r'), help='Path to affine matrix 2 [txt|...]')
    args = parser.parse_args()

    win = QtWidgets.QApplication(sys.argv)
    app = RegistrationApp(args.file1, args.file2, affine1=args.affine1, affine2=args.affine2)
    app.show()
    sys.exit(win.exec())