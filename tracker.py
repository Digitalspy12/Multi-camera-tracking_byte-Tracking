from ultralytics import YOLO
import config
import logging
import numpy as np
import cv2 # For testing block

logging.basicConfig(level=config.LOG_LEVEL)

class IntegratedTracker:
    """
    Handles object detection and tracking using YOLOv8 and a specified tracker config.
    """
    def __init__(self):
        """
        Initializes the YOLO model.
        """
        self.model_path = config.YOLO_MODEL_PATH
        self.tracker_config = config.TRACKER_CONFIG
        self.confidence_threshold = config.DETECTION_CONFIDENCE_THRESHOLD
        self.tracking_classes = config.TRACKING_CLASSES # Filter for specific classes (e.g., [0] for person)

        try:
            self.model = YOLO(self.model_path)
            logging.info(f"Successfully loaded YOLO model: {self.model_path}")
        except Exception as e:
            logging.error(f"Error loading YOLO model from {self.model_path}: {e}")
            raise

    def process_frame(self, frame: np.ndarray):
        """
        Processes a single frame for detection and tracking.

        Args:
            frame (np.ndarray): The input frame (BGR format).

        Returns:
            list: A list of dictionaries, where each dictionary represents an active track
                  and contains 'track_id' (int) and 'bbox' (list [x1, y1, x2, y2]).
                  Returns an empty list if no tracks are found or an error occurs.
        """
        tracks_data = []
        try:
            # Run tracking
            # persist=True tells the tracker that the current image is the next frame in a sequence
            results = self.model.track(
                source=frame,
                persist=True,
                tracker=self.tracker_config,
                conf=self.confidence_threshold,
                classes=self.tracking_classes,
                verbose=False # Set to True for detailed tracking output
            )

            # Check if results and boxes exist
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy() # Bounding boxes (x1, y1, x2, y2)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int) # Track IDs

                for box, track_id in zip(boxes, track_ids):
                    tracks_data.append({
                        'track_id': track_id,
                        'bbox': box.tolist() # Convert numpy array to list [x1, y1, x2, y2]
                    })
                # logging.debug(f"Frame processed. Found {len(tracks_data)} tracks.")
            else:
                # logging.debug("Frame processed. No tracks detected or tracking IDs missing.")
                pass # No tracks found or IDs missing

        except Exception as e:
            logging.error(f"Error during tracking process: {e}")
            # Depending on desired robustness, could return empty list or re-raise

        return tracks_data

# Example Usage (can be run standalone for testing)
if __name__ == '__main__':
    # Create a dummy black frame for testing
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Or load a single frame from a test video (replace with a valid path)
    # test_video_path = config.VIDEO_PATHS[0] if config.VIDEO_PATHS else None
    # if test_video_path:
    #     cap = cv2.VideoCapture(test_video_path)
    #     if cap.isOpened():
    #         ret, dummy_frame = cap.read()
    #         if not ret:
    #             print(f"Failed to read frame from {test_video_path}")
    #             dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    #         cap.release()
    #     else:
    #         print(f"Failed to open test video {test_video_path}")

    print("Initializing IntegratedTracker...")
    try:
        tracker = IntegratedTracker()
        print("Tracker initialized.")

        print("Processing dummy frame...")
        tracks = tracker.process_frame(dummy_frame)
        print(f"Processing complete. Found {len(tracks)} tracks in dummy frame:")
        for track in tracks:
            print(f"  Track ID: {track['track_id']}, BBox: {track['bbox']}")

    except Exception as e:
        print(f"An error occurred during testing: {e}")

    print("Tracker test finished.")