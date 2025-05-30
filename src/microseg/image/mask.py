'''
Mask pixels from an image, either negative or positive
'''

from skimage.restoration import inpaint_biharmonic
import tifffile
from tqdm import tqdm
from microseg.widgets.roi_apps import *
from matgeo.utils.mask import draw_poly

class MaskApp(FlatImageSegmentorApp):

    def __init__(self, *args, desc: str='mask', **kwargs):
        super().__init__(*args, desc=desc, **kwargs)
        self._creator._mode_drop.setCurrentText('Manual')
    
    ''' Public overrides '''
    
    def refreshROIs(self, push: bool=True):
        super().refreshROIs(push)
        self._update_mask()
    
    ''' Private overrides '''

    def _pre_super_init(self):
        super()._pre_super_init()
        self._mask = np.full(self._img.shape[1:], False)

        mask_settings = HLayoutWidget()
        self._main._layout.addWidget(mask_settings)
        self._apply_mask = QCheckBox('Apply mask')
        self._apply_mask.setChecked(True)
        mask_settings._layout.addWidget(self._apply_mask)
        mask_settings.addSpacing(10)
        mask_settings.addWidget(QLabel('Mode:'))
        mask_settings.addSpacing(5)
        self._sign_box = QComboBox()
        self._sign_box.addItems(['positive', 'negative'])
        mask_settings._layout.addWidget(self._sign_box)
        mask_settings.addSpacing(10)
        mask_settings.addWidget(QLabel('Fill:'))
        self._fill_box = QComboBox()
        self._fill_box.addItems(['black', 'white', 'inpaint_biharmonic'])
        mask_settings._layout.addWidget(self._fill_box)
        mask_settings.addStretch()
        self._export_btn = QPushButton('Export')
        mask_settings._layout.addWidget(self._export_btn)
        mask_settings.addSpacing(10)

        self._apply_mask.stateChanged.connect(self._update_mask)
        self._sign_box.currentTextChanged.connect(self._update_mask)
        self._fill_box.currentTextChanged.connect(self._update_mask)
        self._export_btn.clicked.connect(self._export_mask)

    def _query_frame(self, z):
        frame = self._img[z].copy()
        if self._apply_mask.isChecked():
            fill_str = self._fill_box.currentText()
            if fill_str == 'black':
                frame[self._mask] = 0
            elif fill_str == 'white':
                dtype = self._img.dtype
                frame[self._mask] = 255 if dtype == np.uint8 else 1
            elif fill_str == 'inpaint_biharmonic':
                frame = inpaint_biharmonic(frame, self._mask)
        return frame

    def _update_current_frame(self):
        self._creator.setImage(self._query_frame(self._z))

    def _update_mask(self):
        shape = self._img.shape[1:3]
        mask = np.zeros(shape, dtype=np.uint8)
        for roi in self._rois:
            mask = draw_poly(mask, roi.roi.vertices.flatten().tolist(), 1)
        mask = mask.astype(bool)
        if self._sign_box.currentText() == 'negative':
            mask = ~mask
        if self._img.ndim > 3:
            # Multichannel image. Mask all channels
            assert self._img.ndim == 4, 'Expected 4D image'
            mask = np.stack([mask] * self._img.shape[3], axis=3)
        self._mask = mask
        self._update_current_frame()

    def _export_mask(self):
        exp_img = []
        for z in tqdm(range(0, self._zmax), desc='Exporting masked image'):
            exp_img.append(self._query_frame(z))
        exp_img = np.stack(exp_img)
        if exp_img.dtype != np.uint8:
            exp_img = (exp_img * 255).astype(np.uint8)
        orig_name = os.path.splitext(self._img_path)[0] # Use QT dialog for save path
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save masked image', f'{orig_name}.masked.tif', 'TIFF (*.tif)')
        if save_path:  # Only save if user didn't cancel
            tifffile.imwrite(save_path, exp_img)
            print(f'Saved masked image to {save_path}')


if __name__ == '__main__':
    import sys
    import argparse

    from microseg.utils.args import GuiArgumentParser

    parser = GuiArgumentParser(prog="Mask App")
    parser.add_argument('file', type=argparse.FileType('r'), help='Path to source img [tiff|jpg|png|czi|...]')
    parser.add_argument('-d', type=str, default='mask', help='Descriptor')
    args = parser.parse_args()

    win = QtWidgets.QApplication(sys.argv)
    app = MaskApp(args.file.name, desc=args.d)
    app.show()
    sys.exit(win.exec())

