'''
Video segmentor for detecting moving objects in brightfield microscopy videos
Particularly designed for detecting beads in flow with various background artifacts
'''
import numpy as np
import scipy.ndimage
from qtpy.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox, QLabel
from qtpy.QtCore import Qt

from microseg.widgets.roi_apps import ImageSegmentorApp
from microseg.widgets.base import HLayoutWidget

class VideoSegmentorApp(ImageSegmentorApp):
    '''
    Video segmentation app for detecting moving objects like beads in brightfield microscopy
    Extends ImageSegmentorApp with motion detection capabilities optimized for video sequences
    '''
    
    def __init__(self, video_path: str, desc: str='beads', *args, **kwargs):
        # Initialize motion detection state first (before calling super)
        self._motion_mode = False
        self._difference_mode = 'frame_diff'
        self._background_learning_rate = 0.05
        self._temporal_window = 7
        self._motion_threshold = 0.12  # Now in [0,1] range
        self._background_model = None
        self._frame_buffer = []
        
        # Call parent constructor
        super().__init__(video_path, desc, *args, **kwargs)
        
        # Assert this is a video
        assert self._zmax > 1, f'Expected video with multiple frames, got {self._zmax} frames'
        
        # Remove parent's motion controls if they exist and add our video-specific ones
        if hasattr(self, '_motion_controls'):
            self._main._layout.removeWidget(self._motion_controls)
            self._motion_controls.deleteLater()
        
        # Add our video-specific motion controls after creator but before z-slider  
        motion_controls = self._create_motion_controls()
        # Find z-slider position and insert before it
        z_slider_index = self._main._layout.indexOf(self._z_slider)
        if z_slider_index >= 0:
            self._main._layout.insertWidget(z_slider_index, motion_controls)
        else:
            self._main._layout.addWidget(motion_controls)
        
        # Enable motion detection by default for videos
        self._motion_checkbox.setChecked(True)
        self._motion_mode = True
        
        # Update slider constraints based on initial motion mode
        self._update_slider_constraints()
        
        print(f'Initialized VideoSegmentorApp for {self._zmax} frame video')

    def _create_motion_controls(self):
        """Create motion detection control widgets"""
        motion_controls = HLayoutWidget()
        
        # Motion detection checkbox
        self._motion_checkbox = QCheckBox('Motion Detection')
        motion_controls.addWidget(self._motion_checkbox)
        motion_controls.addSpacing(10)
        
        # Difference mode selection
        self._diff_mode_combo = QComboBox()
        self._diff_mode_combo.addItems(['frame_diff', 'background_subtract', 'temporal_median'])
        motion_controls.addWidget(QLabel('Mode:'))
        motion_controls.addWidget(self._diff_mode_combo)
        motion_controls.addSpacing(10)
        
        # Learning rate for background subtraction
        self._learning_rate_spinbox = QDoubleSpinBox(minimum=0.001, maximum=0.5, value=0.05, singleStep=0.01)
        motion_controls.addWidget(QLabel('Learning Rate:'))
        motion_controls.addWidget(self._learning_rate_spinbox)
        motion_controls.addSpacing(10)
        
        # Temporal window for median filtering
        self._window_spinbox = QSpinBox(minimum=3, maximum=31, value=7)
        self._window_spinbox.setSingleStep(2)  # Ensure odd numbers
        motion_controls.addWidget(QLabel('Window:'))
        motion_controls.addWidget(self._window_spinbox)
        motion_controls.addSpacing(10)
        
        # Motion threshold
        self._threshold_spinbox = QDoubleSpinBox(minimum=0.0, maximum=1.0, value=0.12, singleStep=0.01)
        motion_controls.addWidget(QLabel('Threshold:'))
        motion_controls.addWidget(self._threshold_spinbox)
        
        # Connect signals
        self._motion_checkbox.stateChanged.connect(self._toggle_motion_mode)
        self._diff_mode_combo.currentTextChanged.connect(self._update_motion_params)
        self._learning_rate_spinbox.valueChanged.connect(self._update_motion_params)
        self._window_spinbox.valueChanged.connect(self._update_motion_params)
        self._threshold_spinbox.valueChanged.connect(self._update_motion_params)
        
        return motion_controls

    ''' Motion Detection Methods '''

    def _toggle_motion_mode(self, state):
        self._motion_mode = state == Qt.Checked
        
        # Reset background model when enabling motion detection
        if self._motion_mode and self._difference_mode == 'background_subtract':
            self._background_model = None
        
        # Update slider constraints when toggling motion mode
        self._update_slider_constraints()
            
        # Update display when toggling motion mode
        self._update_current_frame()

    def _update_motion_params(self):
        self._difference_mode = self._diff_mode_combo.currentText()
        self._background_learning_rate = self._learning_rate_spinbox.value()
        self._temporal_window = self._window_spinbox.value()
        self._motion_threshold = self._threshold_spinbox.value()
        
        # Reset background model if switching to background subtraction
        if self._difference_mode == 'background_subtract':
            self._background_model = None
        
        # Update slider constraints for new motion params
        self._update_slider_constraints()
            
        # Update display with new parameters
        self._update_current_frame()

    def _update_current_frame(self):
        """Override parent method to use motion-processed images consistently"""
        if not self._motion_mode:
            # Normal mode - show original image
            current_img = self._img[self._z]
        else:
            # Motion detection mode - compute difference image
            current_img = self._compute_motion_difference(self._z)
        
        self._creator.setData(current_img, self._rois[self._z])

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame to [0,1] float32 range regardless of input type"""
        if frame.dtype == np.uint8:
            return frame.astype(np.float32) / 255.0
        else:
            return frame.astype(np.float32)

    def _compute_motion_difference(self, z: int) -> np.ndarray:
        """Compute motion difference image based on current settings"""
        current_frame = self._normalize_frame(self._img[z])
        
        if self._difference_mode == 'frame_diff':
            return self._compute_frame_difference(z, current_frame)
        elif self._difference_mode == 'background_subtract':
            return self._compute_background_subtraction(current_frame)
        elif self._difference_mode == 'temporal_median':
            return self._compute_temporal_median_difference(z, current_frame)
        else:
            return current_frame

    def _compute_frame_difference(self, z: int, current_frame: np.ndarray) -> np.ndarray:
        """Simple frame differencing with previous frame - optimized for bead detection"""
        # Slider constraints ensure z > 0 when this method is called in frame_diff mode
        prev_frame = self._normalize_frame(self._img[z-1])
        
        # Compute absolute difference and apply threshold
        diff = np.abs(current_frame - prev_frame)
        diff = np.where(diff > self._motion_threshold, diff, 0)
        
        # Create binary mask from any channel above threshold
        mask = np.any(diff > 0, axis=2)
        
        # Morphological operations for bead detection
        mask = scipy.ndimage.binary_opening(mask, structure=np.ones((3, 3)))
        mask = scipy.ndimage.binary_closing(mask, structure=np.ones((5, 5)))
        mask = scipy.ndimage.binary_opening(mask, structure=np.ones((2, 2)))
        
        # Apply mask and return
        diff[~mask] = 0
        return diff

    def _compute_background_subtraction(self, current_frame: np.ndarray) -> np.ndarray:
        """Background subtraction using running average - optimized for bead detection"""
        if self._background_model is None:
            self._initialize_background_model()
        
        # Update background model and compute difference
        self._background_model = (1 - self._background_learning_rate) * self._background_model + \
                                self._background_learning_rate * current_frame
        diff = np.abs(current_frame - self._background_model)
        diff = np.where(diff > self._motion_threshold, diff, 0)
        
        # Create binary mask and apply morphological operations
        mask = np.any(diff > 0, axis=2)
        mask = scipy.ndimage.binary_opening(mask, structure=np.ones((3, 3)))
        mask = scipy.ndimage.binary_closing(mask, structure=np.ones((7, 7)))
        mask = scipy.ndimage.binary_erosion(mask, structure=np.ones((2, 2)))
        mask = scipy.ndimage.binary_dilation(mask, structure=np.ones((3, 3)))
        
        # Apply mask and return
        diff[~mask] = 0
        return diff

    def _compute_temporal_median_difference(self, z: int, current_frame: np.ndarray) -> np.ndarray:
        """Temporal median background subtraction - good for static background with moving beads"""
        # Slider constraints ensure valid frame range
        half_window = self._temporal_window // 2
        frames = np.stack([self._normalize_frame(self._img[i]) for i in range(z - half_window, z + half_window + 1)])
        
        # Compute temporal median background and difference
        background = np.median(frames, axis=0)
        diff = np.abs(current_frame - background)
        diff = np.where(diff > self._motion_threshold, diff, 0)
        
        # Create binary mask and apply morphological operations
        mask = np.any(diff > 0, axis=2)
        mask = scipy.ndimage.binary_opening(mask, structure=np.ones((3, 3)))
        mask = scipy.ndimage.binary_closing(mask, structure=np.ones((5, 5)))
        
        # Apply mask and return
        diff[~mask] = 0
        return diff

    def _initialize_background_model(self):
        """Initialize background model for background subtraction"""
        if self._img.ndim == 4:  # ZXYC format
            # Use first frame as initial background, preserving all channels
            self._background_model = self._normalize_frame(self._img[0])

    ''' Override parent methods to ensure consistency with motion detection '''

    def _update_slider_constraints(self):
        """Update the slider constraints based on the current motion mode and parameters"""
        if not self._motion_mode:
            # Normal mode - full range
            min_z, max_z = 0, self._zmax - 1
        else:
            # Motion detection mode - constrain based on algorithm requirements
            if self._difference_mode == 'frame_diff':
                # Frame diff needs either previous or next frame
                # For consistency, use previous frame (so first frame is excluded)
                min_z, max_z = 1, self._zmax - 1
            elif self._difference_mode == 'background_subtract':
                # Background subtraction works on all frames
                min_z, max_z = 0, self._zmax - 1
            elif self._difference_mode == 'temporal_median':
                # Temporal median needs surrounding frames
                half_window = self._temporal_window // 2
                min_z = half_window
                max_z = self._zmax - 1 - half_window
                # Ensure at least one valid frame
                if min_z > max_z:
                    min_z, max_z = 0, self._zmax - 1
            else:
                min_z, max_z = 0, self._zmax - 1
        
        # Update slider range
        current_z = self._z if self._z is not None else 0
        self._z_slider.setData(min_z, max_z, max(min_z, min(current_z, max_z)))
        
        # If current z is outside valid range, move to nearest valid frame
        if current_z < min_z:
            self._set_z(min_z, set_slider=True)
        elif current_z > max_z:
            self._set_z(max_z, set_slider=True)

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