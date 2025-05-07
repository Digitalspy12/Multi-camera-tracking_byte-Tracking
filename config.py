# Configuration for the Multi-Camera Pedestrian Tracking System
import os
import glob

# --- Model Paths ---
OLO_MODEL_PATH = 'yolov8m.pt' # Using official YOLOv8 nano model (compatible with ultralytics lib)
REID_MODEL_NAME = 'resnet50' # Example: 'osnet_x0_25', 'osnet_x1_0', 'resnet50'

# --- Tracking Parameters ---
TRACKER_CONFIG = './bytetrack.yaml' # Use local ByteTrack config
DETECTION_CONFIDENCE_THRESHOLD = 0.4
TRACKING_CLASSES = [0] # COCO class ID for 'person'

# --- Re-Identification Parameters ---
REID_DEVICE = 'cuda' # Use 'cuda' if GPU is available and configured
REID_COSINE_THRESHOLD = 0.7 # Similarity threshold for matching (Increased from 0.6)

# --- Input Video Paths ---
# --- Input Video Paths ---
# Automatically find video files in the 'videos' subdirectory
VIDEO_DIR = 'videos'
# Supported video extensions
VIDEO_EXTENSIONS = ['*.mp4', '*.avi', '*.mov', '*.mkv']

VIDEO_PATHS = []
if os.path.isdir(VIDEO_DIR):
    for ext in VIDEO_EXTENSIONS:
        VIDEO_PATHS.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))
    VIDEO_PATHS.sort() # Ensure consistent order
else:
    print(f"Warning: Video directory '{VIDEO_DIR}' not found. Please create it and add video files.")

# Print found video paths for verification (optional)
# print(f"Found video paths: {VIDEO_PATHS}")

# --- Output Logging ---
LOG_FILE = 'tracking_log.csv'
LOG_LEVEL = 'INFO' # Example: 'DEBUG', 'INFO', 'WARNING'

# --- Visualization ---
ENABLE_VISUALIZATION = True # Set to False to disable OpenCV display windows
