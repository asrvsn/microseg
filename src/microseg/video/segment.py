'''
Video segmentor for detecting moving objects in brightfield microscopy videos
Particularly designed for detecting beads in flow with various background artifacts
'''
import numpy as np
import scipy.ndimage
from typing import Tuple
from qtpy.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLabel, QPushButton
from qtpy import QtWidgets
from tqdm import tqdm
import tifffile
import os

from microseg.widgets.roi_apps import ImageSegmentorApp
from microseg.widgets.base import HLayoutWidget

class MotionDetector(HLayoutWidget):

    def __init__(self, video: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._video = video

    def query_frame(self, z: int) -> np.ndarray:
        return self._video[z]
    
    def query_slider_range(self) -> Tuple[int, int]:
        return 0, self._video.shape[0] - 1
    
    def _normalize_img(self, img: np.ndarray) -> np.ndarray:
        """Normalize img to [0,1] float32 range regardless of input type"""
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        else:
            return img.astype(np.float32)
        
    def _unnormalize_img(self, img: np.ndarray) -> np.ndarray:
        if self._video.dtype == img.dtype:
            return img
        elif self._video.dtype == np.uint8 and img.dtype == np.float32:
            return (img * 255).astype(np.uint8)
        elif self._video.dtype == np.float32 and img.dtype == np.uint8:
            return (img / 255).astype(np.float32)
        else:
            raise ValueError(f'Unsupported dtype: {img.dtype}')

class MedianMotionDetector(MotionDetector):

    def __init__(self, video: np.ndarray, *args, **kwargs):
        super().__init__(video, *args, **kwargs)
        vid = self._normalize_img(self._video)
        self._median_background = np.median(vid, axis=0)
        self._mean_intensity = vid.mean()

    def query_frame(self, z: int) -> np.ndarray:
        frame = self._normalize_img(self._video[z]) - self._median_background
        frame *= self._mean_intensity / frame.mean()
        return frame

class VideoSegmentorApp(ImageSegmentorApp):
    '''
    Video segmentation app for detecting moving objects like beads in brightfield microscopy
    Extends ImageSegmentorApp with motion detection capabilities optimized for video sequences
    '''
    MOTION_DETECTORS = {
        'none': MotionDetector,
        'median': MedianMotionDetector,
    }
    def __init__(self, video_path: str, desc: str='beads', *args, **kwargs):
        # State
        self._motion_threshold = 0.12  # In [0,1] range
        self._apply_threshold = False
        
        super().__init__(video_path, desc, *args, **kwargs)

        assert self._zmax > 1, f'Expected video with multiple frames, got {self._zmax} frames'
        print(f'Initialized VideoSegmentorApp for {self._zmax} frame video')

    def _pre_super_init(self):
        """Create motion detection control widgets"""
        motion_controls = HLayoutWidget()
        
        # Motion detection label and mode selection
        print('Initializing motion detectors...')
        self._motion_detectors = {k: v(self._img) for k, v in self.MOTION_DETECTORS.items()}
        print('done.')
        motion_controls.addSpacing(10)
        motion_controls.addWidget(QLabel('Motion detection:'))
        self._motion_detector_combo = QComboBox()
        self._motion_detector_combo.addItems(list(self._motion_detectors.keys()))
        motion_controls.addWidget(self._motion_detector_combo)
        motion_controls.addSpacing(10)
        for v in self._motion_detectors.values():
            motion_controls.addWidget(v)
            motion_controls.addSpacing(10)
        self._update_motion_detector(set_frame=False)
        
        # Threshold checkbox and spinbox
        motion_controls.addStretch()
        motion_controls.addWidget(QLabel('Apply threshold:'))
        self._threshold_checkbox = QCheckBox()
        self._threshold_checkbox.setChecked(False)
        motion_controls.addWidget(self._threshold_checkbox)
        motion_controls.addSpacing(5)
        
        self._threshold_spinbox = QDoubleSpinBox(minimum=0.0, maximum=1.0, value=0.12, singleStep=0.01)
        motion_controls.addWidget(self._threshold_spinbox)
        motion_controls.addSpacing(10)
        self._export_btn = QPushButton('Export')
        motion_controls.addWidget(self._export_btn)
        
        # Connect signals
        self._motion_detector_combo.currentTextChanged.connect(self._update_motion_detector)
        self._threshold_checkbox.stateChanged.connect(self._update_motion_params)
        self._threshold_spinbox.valueChanged.connect(self._update_motion_params)
        self._export_btn.clicked.connect(self._export)
        
        self._main.addWidget(motion_controls)

    ''' Motion Detection Methods '''

    def _update_motion_detector(self, set_frame: bool=True):
        key = self._motion_detector_combo.currentText()
        for k in self._motion_detectors.keys():
            if k != key:
                self._motion_detectors[k].hide()
            else:
                self._motion_detectors[k].show()
        self._motion_detector = self._motion_detectors[key]
        self._zmin, self._zmax = self._motion_detector.query_slider_range()
        z = self._z if self._z is not None else self._zmin
        z = max(self._zmin, min(z, self._zmax))
        self._z_slider.setData(self._zmin, self._zmax, z)
        if set_frame:
            self._update_current_frame()

    def _update_motion_params(self):
        self._motion_threshold = self._threshold_spinbox.value()
        self._apply_threshold = self._threshold_checkbox.isChecked()
        self._update_current_frame()

    def _export(self):
        exp_img = []
        for z in tqdm(range(self._zmin, self._zmax), desc='Exporting video'):
            exp_img.append(self._motion_detector.query_frame(z))
        exp_img = np.stack(exp_img)
        # exp_img = (exp_img * 255).astype(np.uint8)
        orig_name = os.path.splitext(self._img_path)[0] # Use QT dialog for save path
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save motion video', f'{orig_name}.motion.tif', 'TIFF (*.tif)')
        if save_path:  # Only save if user didn't cancel
            tifffile.imwrite(save_path, exp_img)
            print(f'Saved motion video to {save_path}')

    def _query_frame(self, z: int) -> np.ndarray:
        frame = self._motion_detector.query_frame(z)
        if self._apply_threshold:
            frame = np.where(frame > self._motion_threshold, frame, 0)
        return frame

    ''' Overrides'''

    def _update_current_frame(self):
        frame = self._query_frame(self._z)
        # Apply thresholding and morphological operations if enabled
        if self._apply_threshold:
            frame = np.where(frame > self._motion_threshold, frame, 0)
            
            # # Create binary mask and apply morphological operations
            # # Handle both grayscale (2D) and color (3D) frames
            # if frame.ndim == 3:
            #     mask = np.any(frame > 0, axis=2)
            # else:
            #     mask = frame > 0
            # mask = scipy.ndimage.binary_opening(mask, structure=np.ones((3, 3)))
            # mask = scipy.ndimage.binary_closing(mask, structure=np.ones((5, 5)))
            # if frame.ndim == 3:
            #     frame[~mask] = 0
            # else:
            #     frame = np.where(mask, frame, 0)
        # assert self._img.dtype == frame.dtype
        self._creator.setData(frame, self._rois[self._z])

if __name__ == '__main__':
    import sys
    import argparse
    from qtpy import QtWidgets
    from microseg.utils.args import GuiArgumentParser

    parser = GuiArgumentParser(prog="Video Segmentor")
    parser.add_argument('file', type=argparse.FileType('r'), help='Path to video file [tiff|avi|mp4|...]')
    parser.add_argument('-d', type=str, default='beads', help='Descriptor for objects to segment')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    segmentor = VideoSegmentorApp(args.file.name, desc=args.d)
    segmentor.show()
    sys.exit(app.exec()) 