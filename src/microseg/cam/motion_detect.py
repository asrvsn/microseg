#!/usr/bin/env python3
"""
Motion detection application that monitors camera feed and records videos when motion is detected.

This script uses OpenCV to capture video from a camera, detect motion using
background subtraction, and record video segments to disk when motion is detected.
It features several optimizations for performance and file size, including
resolution scaling for both motion detection and recording, adjustable framerates,
and asynchronous video writing.
"""

import cv2
import numpy as np
import os
import argparse
import datetime
import time
import logging
from pathlib import Path
import threading
from queue import Queue, Empty, Full
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default values for parameters
DEFAULT_MOTION_CHECK_INTERVAL: int = 3
DEFAULT_MOTION_DETECTION_SCALE: float = 0.5
DEFAULT_RECORDING_SCALE: float = 0.5
DEFAULT_RECORDING_FPS: int = 24
DEFAULT_MOTION_TIMEOUT_MINUTES: float = 1.0
DEFAULT_MAX_RECORDING_HOURS: float = 1.0
DEFAULT_CAMERA_INDEX: int = 0

MIN_CONTOUR_AREA: int = 1000 # Minimum area for a contour to be considered motion
VIDEO_QUEUE_SIZE: int = 60 # Max number of frames to buffer for video writing


class MotionDetector:
    """Class to handle motion detection, camera feed, and video recording."""

    def __init__(self, 
                 camera_index: int = DEFAULT_CAMERA_INDEX, 
                 output_dir: Optional[str] = None, 
                 max_recording_hours: float = DEFAULT_MAX_RECORDING_HOURS, 
                 recording_scale: float = DEFAULT_RECORDING_SCALE, 
                 recording_fps: int = DEFAULT_RECORDING_FPS, 
                 motion_timeout_minutes: float = DEFAULT_MOTION_TIMEOUT_MINUTES,
                 motion_check_interval: int = DEFAULT_MOTION_CHECK_INTERVAL,
                 motion_detection_scale: float = DEFAULT_MOTION_DETECTION_SCALE):
        """
        Initialize motion detector.

        Args:
            camera_index: Camera device index.
            output_dir: Directory to save recorded videos. Must be provided.
            max_recording_hours: Maximum recording duration in hours.
            recording_scale: Scale factor for recording resolution (0.1 to 1.0).
            recording_fps: Recording framerate (1 to 60 FPS).
            motion_timeout_minutes: Minutes to continue recording after last motion.
            motion_check_interval: Check for motion every N frames.
            motion_detection_scale: Scale factor for motion detection resolution (0.1 to 1.0).
        """
        if output_dir is None:
            logging.error("Output directory is required.")
            raise ValueError("output_dir is required")
        
        self.camera_index: int = camera_index
        self.output_dir: Path = Path(output_dir)
        self.max_recording_duration: float = max_recording_hours * 3600  # Convert to seconds
        self.motion_timeout: float = motion_timeout_minutes * 60      # Convert minutes to seconds
        
        # Performance and file size optimizations
        self.motion_check_interval: int = motion_check_interval
        self.frame_counter: int = 0
        self.motion_detection_scale: float = motion_detection_scale
        self.recording_scale: float = recording_scale
        self.recording_fps: int = recording_fps
        
        # Pre-allocate reusable objects
        self.motion_kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Motion detection parameters
        self.bg_subtractor: cv2.BackgroundSubtractorMOG2 = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False,
            varThreshold=25,
            history=200
        )
        self.min_contour_area: int = MIN_CONTOUR_AREA
        
        # Recording state
        self.is_recording: bool = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.last_motion_time: Optional[float] = None
        self.recording_start_time: Optional[float] = None
        self.current_video_path: Optional[Path] = None
        
        # Threading for async video writing
        self.frame_queue: Queue = Queue(maxsize=VIDEO_QUEUE_SIZE)
        self.writing_thread: Optional[threading.Thread] = None
        self.stop_writing_event: threading.Event = threading.Event()
        
        # Initialize camera
        self.cam: cv2.VideoCapture = cv2.VideoCapture(camera_index)
        if not self.cam.isOpened():
            logging.error(f"Could not open camera {camera_index}")
            raise RuntimeError(f"Could not open camera {camera_index}")
        
        self._configure_camera()
        
        # Get camera properties
        raw_camera_fps: float = self.cam.get(cv2.CAP_PROP_FPS)
        self.camera_fps: int = int(raw_camera_fps) if raw_camera_fps > 0 else 30
        self.frame_width: int = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height: int = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.frame_width == 0 or self.frame_height == 0:
            logging.error(f"Failed to get valid frame dimensions from camera {camera_index}. Check camera connection.")
            self.cam.release()
            raise RuntimeError(f"Invalid frame dimensions from camera {camera_index}.")

        # Motion detection dimensions
        self.motion_width: int = int(self.frame_width * self.motion_detection_scale)
        self.motion_height: int = int(self.frame_height * self.motion_detection_scale)
        
        # Recording dimensions
        self.recording_width: int = int(self.frame_width * self.recording_scale)
        self.recording_height: int = int(self.frame_height * self.recording_scale)
        self.recording_width -= (self.recording_width % 2)
        self.recording_height -= (self.recording_height % 2)
        
        logging.info(f"Camera [{camera_index}]: {self.frame_width}x{self.frame_height} @ {self.camera_fps} FPS")
        logging.info(f"Motion detection: {self.motion_width}x{self.motion_height} (scale: {self.motion_detection_scale:.2f})")
        logging.info(f"Recording: {self.recording_width}x{self.recording_height} @ {self.recording_fps} FPS (scale: {self.recording_scale:.2f})")
        logging.info(f"Motion timeout: {motion_timeout_minutes:.1f} minute(s)")
        logging.info(f"Max recording duration: {max_recording_hours:.1f} hour(s)")
        logging.info(f"Motion check interval: {self.motion_check_interval} frames")

        self._log_estimated_file_size_reduction()

        # Add to __init__:
        self.video_writer_lock: threading.Lock = threading.Lock()

        self.warmup_frames: int = 30  # Number of frames to learn background
        self.warmup_counter: int = 0
        self.is_warmed_up: bool = False

        if not (0.1 <= recording_scale <= 1.0):
            raise ValueError(f"recording_scale must be between 0.1 and 1.0, got {recording_scale}")
        if not (1 <= recording_fps <= 60):
            raise ValueError(f"recording_fps must be between 1 and 60, got {recording_fps}")
        if not (0.1 <= motion_detection_scale <= 1.0):
            raise ValueError(f"motion_detection_scale must be between 0.1 and 1.0, got {motion_detection_scale}")
        if not (0.1 <= motion_timeout_minutes <= 10):
            raise ValueError(f"motion_timeout_minutes must be between 0.1 and 10, got {motion_timeout_minutes}")
        if not (0.1 <= max_recording_hours <= 24):
            raise ValueError(f"max_recording_hours must be between 0.1 and 24, got {max_recording_hours}")
        if not (1 <= motion_check_interval <= 100):
            raise ValueError(f"motion_check_interval must be between 1 and 100, got {motion_check_interval}")

    def _log_estimated_file_size_reduction(self) -> None:
        """Helper to log estimated file size reduction based on current settings."""
        resolution_reduction_factor = self.recording_scale ** 2
        fps_reduction_factor = self.recording_fps / self.camera_fps if self.camera_fps > 0 else 1.0
        
        total_reduction_percentage = (1.0 - (resolution_reduction_factor * fps_reduction_factor)) * 100
        
        logging.info(f"Estimated file size: ~{100-total_reduction_percentage:.1f}% of original (due to resolution and FPS changes)")

    def create_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Output directory: {self.output_dir.absolute()}")
        except OSError as e:
            logging.error(f"Could not create output directory '{self.output_dir}': {e}")
            raise
        
    def detect_motion(self, frame: np.ndarray) -> bool:
        """
        Detect motion in the given frame.

        Args:
            frame: Input frame from camera (BGR format).
            
        Returns:
            True if motion detected, False otherwise.
        """
        small_frame: np.ndarray = cv2.resize(frame, (self.motion_width, self.motion_height))
        fg_mask: np.ndarray = self.bg_subtractor.apply(small_frame) 
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.motion_kernel)
        
        motion_pixels: int = cv2.countNonZero(fg_mask)
        # Adjust motion area based on the square of the detection scale factor
        # to compare it with min_contour_area defined for the original resolution.
        # This is an approximation for actual contour area.
        effective_motion_area: float = motion_pixels / (self.motion_detection_scale ** 2) if self.motion_detection_scale > 0 else motion_pixels
        
        return effective_motion_area > self.min_contour_area
        
    def _video_writer_thread(self) -> None:
        """Background thread for writing video frames from the queue."""
        logging.info("Video writer thread started.")
        while not self.stop_writing_event.is_set() or not self.frame_queue.empty():
            try:
                frame_to_write: Optional[np.ndarray] = self.frame_queue.get(timeout=0.1) # Short timeout to check stop_event
                with self.video_writer_lock:
                    if self.video_writer and self.video_writer.isOpened():
                        self.video_writer.write(frame_to_write)
                self.frame_queue.task_done()
            except Empty:
                if self.stop_writing_event.is_set() and self.frame_queue.empty():
                    break # Exit if stop is set and queue is drained
                continue # Continue to periodically check queue and stop_event
            except Exception as e:
                logging.error(f"Error writing frame: {e}", exc_info=True)
        logging.info("Video writer thread finished.")
                
    def start_recording(self) -> None:
        """Start recording video with optimized compression and async writing."""
        if self.is_recording:
            return
            
        timestamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_video_path = self.output_dir / f"motion_{timestamp}.mp4"
        
        # Try H.264 first, then XVID as fallback
        codecs_to_try: list[str] = ['H264', 'XVID', 'MP4V'] # MP4V as a last resort
        
        for codec_name in codecs_to_try:
            fourcc: int = cv2.VideoWriter_fourcc(*codec_name)
            self.video_writer = cv2.VideoWriter(
                str(self.current_video_path),
                fourcc,
                float(self.recording_fps),
                (self.recording_width, self.recording_height),
                True  # isColor
            )
            if self.video_writer.isOpened():
                logging.info(f"Using {codec_name} codec for recording.")
                break
            else:
                logging.warning(f"Failed to open video writer with {codec_name} codec.")
                self.video_writer = None # Ensure it's None if not opened
        
        if not self.video_writer:
            logging.error(f"Could not open video writer with any available codecs for {self.current_video_path}")
            # Potentially raise an error or handle this more gracefully
            return

        self.stop_writing_event.clear()
        self.writing_thread = threading.Thread(target=self._video_writer_thread, daemon=True)
        self.writing_thread.start()
        
        self.is_recording = True
        self.recording_start_time = time.time()
        logging.info(f"Started recording: {self.current_video_path.name} ({self.recording_width}x{self.recording_height} @ {self.recording_fps} FPS)")
        
    def stop_recording(self) -> None:
        """Stop recording video and ensure all frames are written."""
        if not self.is_recording:
            return
            
        self.is_recording = False
        logging.info("Stopping recording...")

        self.stop_writing_event.set() # Signal writer thread to stop after emptying queue
        
        if self.writing_thread and self.writing_thread.is_alive():
            logging.debug("Waiting for video writer thread to finish...")
            self.writing_thread.join(timeout=5.0) # Wait for thread to finish (max 5s)
            if self.writing_thread.is_alive():
                logging.warning("Video writer thread did not finish in time.")
        
        with self.video_writer_lock:
            if self.video_writer:
                try:
                    self.video_writer.release()
                    logging.info(f"Video writer released for {self.current_video_path.name if self.current_video_path else 'N/A'}")
                except Exception as e:
                    logging.error(f"Error releasing video writer: {e}", exc_info=True)
                finally:
                    self.video_writer = None

        # Clear any remaining frames in the queue if thread timed out
        # This prevents old frames from being written to a new file if recording restarts quickly
        if not self.frame_queue.empty():
            logging.warning(f"Clearing {self.frame_queue.qsize()} frames from queue after stopping recording.")
            remaining_frames = 0
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                    remaining_frames += 1
                except Empty:
                    break

        if self.recording_start_time and self.current_video_path:
            duration: float = time.time() - self.recording_start_time
            logging.info(f"Stopped recording: {self.current_video_path.name} (Duration: {duration:.1f}s)")
        else:
            logging.info("Stopped recording (no start time/path info available).")
        
        self.recording_start_time = None
        self.current_video_path = None # Reset after logging
        
    def should_stop_recording(self) -> bool:
        """Check if recording should be stopped based on duration or motion timeout."""
        if not self.is_recording or self.recording_start_time is None:
            return False # Not recording or start time not set
            
        current_time: float = time.time()
        
        # Check maximum recording duration
        if current_time - self.recording_start_time >= self.max_recording_duration:
            logging.info("Maximum recording duration reached.")
            return True
            
        # Check motion timeout
        # If last_motion_time is None, it means no motion since recording started, 
        # so rely on max_recording_duration or initial motion detection to stop.
        if self.last_motion_time and (current_time - self.last_motion_time >= self.motion_timeout):
            timeout_minutes: float = self.motion_timeout / 60
            logging.info(f"No motion detected for {timeout_minutes:.1f} minute(s).")
            return True
            
        return False
        
    def _add_overlay(self, frame: np.ndarray, motion_detected: bool) -> None:
        """Add status overlay to the frame (e.g., REC, MON, motion indicator)."""
        current_time: float = time.time()
        
        # Recording status text
        status_text: str
        color: Tuple[int, int, int]
        if self.is_recording:
            status_text = "REC"
            color = (0, 0, 255) # Red
        else:
            status_text = "MONITORING"
            color = (0, 255, 0) # Green
        
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
        # Motion detected indicator (e.g., a yellow circle)
        if motion_detected:
            cv2.circle(frame, (self.frame_width - 30, 30), 10, (0, 255, 255), -1) # Yellow circle
            
        # Recording duration text
        if self.is_recording and self.recording_start_time:
            duration: int = int(current_time - self.recording_start_time)
            duration_text: str = f"{duration}s"
            cv2.putText(frame, duration_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
    def _add_warmup_overlay(self, frame: np.ndarray) -> None:
        """Add warm-up overlay to the frame."""
        cv2.putText(frame, "LEARNING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame for motion detection and recording."""
        current_time: float = time.time()
        motion_detected_this_check: bool = False
        
        # Warm-up period: let background subtractor learn the scene
        if not self.is_warmed_up:
            self.warmup_counter += 1
            # Feed frames to background subtractor during warmup but don't check for motion
            small_frame = cv2.resize(frame, (self.motion_width, self.motion_height))
            self.bg_subtractor.apply(small_frame)  # Train background model
            
            if self.warmup_counter >= self.warmup_frames:
                self.is_warmed_up = True
                logging.info(f"Background model trained with {self.warmup_frames} frames. Motion detection active.")
            
            # During warmup, show "LEARNING" status
            self._add_warmup_overlay(frame)
            return frame
        
        # Normal motion detection logic starts here...
        self.frame_counter += 1
        if self.frame_counter >= self.motion_check_interval:
            self.frame_counter = 0
            motion_detected_this_check = self.detect_motion(frame)
            
            if motion_detected_this_check:
                self.last_motion_time = current_time # Update last motion time
                if not self.is_recording:
                    self.start_recording()
        
        if self.is_recording:
            # Create a copy of the frame for the writing thread to avoid modifications
            # if the main thread reuses the frame buffer for the next camera read.
            # Also, scale it here to offload work from the writer thread if not 1.0 scale.
            try:
                frame_to_queue: np.ndarray
                if self.recording_scale != 1.0:
                    frame_to_queue = cv2.resize(frame, (self.recording_width, self.recording_height), interpolation=cv2.INTER_AREA)
                else:
                    frame_to_queue = frame.copy() # Still copy if full resolution to decouple from main loop's frame
                
                self.frame_queue.put_nowait(frame_to_queue)
            except Full:
                logging.warning("Frame queue is full. Skipping frame for recording.")
            except Exception as e:
                logging.error(f"Error queueing frame: {e}", exc_info=True)
                
            if self.should_stop_recording():
                self.stop_recording()
        
        # Add overlay to the original frame for display
        # Pass motion_detected_this_check for consistent display with current detection status
        self._add_overlay(frame, motion_detected_this_check)
        
        return frame # Return the original frame (with overlay) for display
        
    def run(self) -> None:
        """Main loop for motion detection application."""
        self.create_output_directory() # Raises error if it fails
        
        logging.info("Motion detection started. Press 'q' to quit.")
        # Detailed settings already logged in __init__
        
        try:
            while True:
                ret, frame = self.cam.read()
                if not ret or frame is None:
                    logging.warning("Failed to read frame from camera or end of stream. Attempting to re-open camera...")
                    self.cam.release()
                    self.cam = cv2.VideoCapture(self.camera_index)
                    if not self.cam.isOpened():
                        logging.error("Failed to re-open camera. Exiting.")
                        break
                    time.sleep(0.5) # Brief pause before retrying
                    self._configure_camera()
                    continue
                    
                display_frame: np.ndarray = self.process_frame(frame)
                cv2.imshow('Motion Detection', display_frame)
                
                key: int = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logging.info("'q' pressed, exiting.")
                    break
                elif key == ord('s'): # Secret key to manually stop recording if stuck
                    if self.is_recording:
                        logging.info("'s' pressed, manually stopping recording.")
                        self.stop_recording()
                    
        except KeyboardInterrupt:
            logging.info("Interrupted by user (Ctrl+C).")
        except Exception as e:
            logging.error(f"An unhandled error occurred in the main loop: {e}", exc_info=True)
        finally:
            self.cleanup()
            
    def cleanup(self) -> None:
        """Clean up resources (camera, video writer, windows)."""
        logging.info("Cleaning up resources...")
        if self.is_recording:
            self.stop_recording() # Ensure recording is stopped and finalized
            
        if self.cam.isOpened():
            self.cam.release()
            logging.info("Camera released.")
            
        cv2.destroyAllWindows()
        logging.info("OpenCV windows destroyed.")
        logging.info("Cleanup completed.")

    def _configure_camera(self) -> None:
        """Configure camera settings for optimal performance."""
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def main() -> int:
    """Parse CLI arguments and run the motion detector."""
    parser = argparse.ArgumentParser(
        description="Motion detection application.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    parser.add_argument("--cam", type=int, default=DEFAULT_CAMERA_INDEX, 
                        help="Camera device index.")
    parser.add_argument("--output-dir", type=str, required=True, 
                        help="Output directory for recorded videos (required). Example: /tmp/recordings")
    parser.add_argument("--max-hours", type=float, default=DEFAULT_MAX_RECORDING_HOURS,
                        help="Maximum recording duration in hours for a single video file.")
    parser.add_argument("--motion-interval", type=int, default=DEFAULT_MOTION_CHECK_INTERVAL,
                        help="Check for motion every N frames.")
    parser.add_argument("--detection-scale", type=float, default=DEFAULT_MOTION_DETECTION_SCALE,
                        help="Scale factor for motion detection frame (0.1-1.0). Smaller = faster but less detail.")
    parser.add_argument("--recording-scale", type=float, default=DEFAULT_RECORDING_SCALE, 
                       help="Scale factor for recorded video resolution (0.1-1.0). Smaller = smaller files.")
    parser.add_argument("--recording-fps", type=int, default=DEFAULT_RECORDING_FPS,
                       help="Recording framerate (1-60 FPS). Lower = smaller files.")
    parser.add_argument("--motion-timeout-minutes", type=float, default=DEFAULT_MOTION_TIMEOUT_MINUTES,
                       help="Minutes to continue recording after last motion detected.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level.")
    
    args = parser.parse_args()

    # Update logging level based on CLI argument
    try:
        logging.getLogger().setLevel(args.log_level.upper())
    except ValueError:
        logging.error(f"Invalid log level: {args.log_level}. Using INFO.")
        logging.getLogger().setLevel(logging.INFO)

    try:
        detector = MotionDetector(
            camera_index=args.cam,
            output_dir=args.output_dir,
            max_recording_hours=args.max_hours,
            recording_scale=args.recording_scale,
            recording_fps=args.recording_fps,
            motion_timeout_minutes=args.motion_timeout_minutes,
            motion_check_interval=args.motion_interval,
            motion_detection_scale=args.detection_scale
        )
        detector.run()
        return 0
    except ValueError as ve:
        logging.error(f"Configuration error: {ve}") # e.g. output_dir not specified if default was removed from MotionDetector
        return 1
    except RuntimeError as rte:
        logging.error(f"Runtime error during initialization or camera operation: {rte}")
        return 1
    except Exception as e:
        logging.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit_code = main()
    logging.info(f"Application finished with exit code {exit_code}.")
    exit(exit_code)
