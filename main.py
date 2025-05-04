import cv2
import time
import cv2
import time
import logging
import os
import config
import numpy as np # Added for ReID feature handling if needed
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict # Added for potential future trail implementation

# Import core modules
from ultralytics import YOLO # Use YOLO directly
from video_loader import VideoLoader
# from tracker import IntegratedTracker # Removed custom tracker
from reid import ReIDExtractor
from global_id import GlobalIDManager
# from logger import CsvLogger # Removed CsvLogger class import
from logger import log_batch_events, close_logger, LOG_FILE_PATH # Import new logger functions/path
from visualization import visualize_tracks # Import the new visualization function

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Variables/Objects (Initialized in main) ---
# yolo_model = None # Removed global YOLO model instance
reid_extractor = None
global_id_manager = None
# csv_logger = None # Removed CsvLogger instance
output_video_writers = []

# --- Processing Function for a Single Camera Frame ---
def process_camera_frame(args):
    """
    Processes a single frame from a specific camera.
    Includes detection, tracking, Re-ID, and visualization prep.
    """
    cam_id, frame, frame_id = args
    if frame is None:
        return cam_id, None, [] # Return None frame and empty events if input is None

    logging.debug(f"Frame {frame_id}: Processing Camera {cam_id} in thread...")
    processed_frame = frame.copy() # Work on a copy for visualization
    log_events = []

    try:
        # Initialize YOLO model within the thread-specific function
        # This ensures each thread has its own model instance, avoiding potential conflicts
        yolo_model = YOLO(config.YOLO_MODEL_PATH)

        # 3. Perform detection and tracking using YOLOv9/v8 integrated tracker
        results = yolo_model.track(
            source=processed_frame,
            tracker=config.TRACKER_CONFIG,
            conf=config.DETECTION_CONFIDENCE_THRESHOLD,
            classes=config.TRACKING_CLASSES,
            device=config.REID_DEVICE, # Assuming tracker runs on same device as ReID
            persist=True, # Persist tracks between frames
            verbose=False # Reduce console output from YOLO
        )

        # Check if results contain tracks
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            tracks = results[0].boxes.data.cpu().numpy() # Get track data as numpy array [x1, y1, x2, y2, track_id, conf, class_id]
            logging.debug(f"Frame {frame_id}, Cam {cam_id}: Found {len(tracks)} tracks.")

            # 4. Process each track for Re-ID and Global ID assignment
            for track in tracks:
                # Extract bounding box and local track ID
                x1, y1, x2, y2, local_track_id, conf, _ = track
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                local_track_id = int(local_track_id)

                # 5. Extract Re-ID feature
                feature = reid_extractor.extract(processed_frame, bbox) # Use processed_frame (copy)

                global_id = -1 # Default to invalid ID
                if feature is not None:
                    # 6. Match feature for Global ID
                    global_id = global_id_manager.match(feature)
                    logging.debug(f"  Cam {cam_id}, Track {local_track_id}: Feature extracted. Assigned Global ID: {global_id}")
                else:
                    logging.warning(f"  Cam {cam_id}, Track {local_track_id}: Feature extraction failed.")

                # 7. Prepare log event
                log_event = {
                    'frame_id': frame_id,
                    'camera_id': cam_id,
                    'track_id': local_track_id,
                    'global_id': global_id,
                    'bbox': bbox
                }
                log_events.append(log_event)

                # Visualization is now handled after the loop
        else:
            logging.debug(f"Frame {frame_id}, Cam {cam_id}: No tracks found by YOLO.")

        # 8. (Optional) Visualization (Call the dedicated function)
        if config.ENABLE_VISUALIZATION and log_events: # Check if there are events to visualize
             visualize_tracks(processed_frame, log_events) # Modify frame in-place

    except Exception as e:
        logging.error(f"Error processing frame for Camera {cam_id}: {e}", exc_info=True)
        # Return original frame on error? Or None? Returning processed_frame might have partial drawing.
        # If visualize_tracks failed, processed_frame might be partially drawn.
        return cam_id, processed_frame, [] # Return potentially modified frame and empty events on error

    # Return the frame (potentially modified by visualization) and the log events
    return cam_id, processed_frame, log_events


def main():
    # Make objects global so they can be accessed by process_camera_frame
    # No need for csv_logger or yolo_model in globals anymore
    global reid_extractor, global_id_manager, output_video_writers # Removed yolo_model

    """
    Main function to run the multi-camera pedestrian tracking pipeline.
    """
    logging.info("Starting multi-camera tracking pipeline...")
    start_time = time.time()

    """
    Main function to run the multi-camera pedestrian tracking pipeline.
    """
    logging.info("Starting multi-camera tracking pipeline...")
    start_time = time.time()

    # --- Initialization ---
    try:
        logging.info("Initializing modules...")
        # Initialize YOLO model - MOVED to process_camera_frame
        # yolo_model = YOLO(config.YOLO_MODEL_PATH) # Removed from main initialization
        # logging.info(f"YOLO model loaded from {config.YOLO_MODEL_PATH}") # Removed log message

        video_loader = VideoLoader(config.VIDEO_PATHS)
        # tracker = IntegratedTracker() # Removed
        reid_extractor = ReIDExtractor() # Assumes ReIDExtractor handles its own model loading
        global_id_manager = GlobalIDManager()
        # csv_logger = CsvLogger(config.LOG_FILE) # Removed CsvLogger initialization
        # The new logger initializes itself upon import in logger.py
        logging.info("Core modules initialized successfully.")
    except Exception as e:
        logging.error(f"Initialization failed: {e}", exc_info=True)
        return

    num_cameras = video_loader.get_num_cameras()
    if num_cameras == 0:
        logging.warning("No video sources available. Exiting.")
        return

    frame_id = 0
    processing_times = []
    # output_video_writers = [] # Moved to global

    # --- Setup Output Video Writers (if visualization enabled) ---
    if config.ENABLE_VISUALIZATION:
        output_dir = config.OUTPUT_VIDEO_DIR if hasattr(config, 'OUTPUT_VIDEO_DIR') else "output_videos" # Use config if available
        os.makedirs(output_dir, exist_ok=True)
        for i, cap in enumerate(video_loader.caps):
            if cap.isOpened():
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
                base_name = os.path.basename(video_loader.video_paths[i])
                output_filename = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_output.mp4")
                writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
                output_video_writers.append(writer)
                logging.info(f"Initialized video writer for Camera {i} to {output_filename}")
            else:
                output_video_writers.append(None) # Placeholder if capture failed
                logging.warning(f"Could not initialize video writer for Camera {i} (capture not open).")
    else:
         output_video_writers = [None] * num_cameras # Ensure list has correct size even if not writing

    # --- Main Processing Loop with ThreadPoolExecutor ---
    num_workers = config.NUM_WORKERS if hasattr(config, 'NUM_WORKERS') else min(4, num_cameras if num_cameras > 0 else 1) # Default workers
    logging.info(f"Using {num_workers} worker threads for parallel processing.")

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            while video_loader.active:
                loop_start_time = time.time()
                logging.info(f"--- Processing Frame {frame_id} ---")

                # 1. Read frames from all cameras
                frames, active = video_loader.read_frames()
                if not active:
                    logging.info("Video stream ended or failed. Exiting loop.")
                    break

                # Prepare arguments for parallel processing
                tasks_args = [(cam_id, frame, frame_id) for cam_id, frame in enumerate(frames)]

                # 2. Submit frame processing tasks to the thread pool
                futures = [executor.submit(process_camera_frame, args) for args in tasks_args]

                all_frame_log_events = []
                processed_frames = [None] * num_cameras # To store results in order

                # 3. Collect results as they complete
                for future in futures:
                    try:
                        cam_id_res, processed_frame_res, log_events_res = future.result()
                        if processed_frame_res is not None:
                             processed_frames[cam_id_res] = processed_frame_res # Store processed frame
                        if log_events_res:
                            all_frame_log_events.extend(log_events_res)
                    except Exception as exc:
                        logging.error(f'Generated an exception: {exc}', exc_info=True)

                # 4. Log all events collected for this frame_id using the new function
                if all_frame_log_events:
                    log_batch_events(all_frame_log_events) # Use imported function

                # 5. Write processed frames to output videos (if enabled)
                if config.ENABLE_VISUALIZATION:
                    for cam_id, p_frame in enumerate(processed_frames):
                        if p_frame is not None and output_video_writers[cam_id] is not None:
                            try:
                                output_video_writers[cam_id].write(p_frame)
                            except Exception as write_e:
                                logging.error(f"Error writing frame for Camera {cam_id}: {write_e}")

                frame_id += 1
                loop_end_time = time.time()
                processing_times.append(loop_end_time - loop_start_time)

                # Optional: Add a small sleep if CPU usage is too high,
                # though usually I/O or GPU will be the bottleneck.
                # time.sleep(0.001)

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Stopping pipeline.")
    except Exception as e:
        logging.error(f"An error occurred during the main loop: {e}", exc_info=True)
    finally:
        # --- Cleanup ---
        logging.info("Cleaning up resources...")
        video_loader.release()
        # csv_logger.close() # Removed - new logger uses atexit for cleanup
        # Release video writers
        for writer in output_video_writers:
            if writer is not None:
                writer.release()
        logging.info("Released output video writers.")
        # cv2.destroyAllWindows() # Not needed if not using imshow

        logging.info("Resources cleaned up.")

        # --- Performance Summary ---
        end_time = time.time()
        total_time = end_time - start_time
        avg_fps = frame_id / total_time if total_time > 0 else 0
        avg_frame_time = sum(processing_times) / len(processing_times) if processing_times else 0
        logging.info(f"--- Pipeline Finished ---")
        logging.info(f"Total frames processed: {frame_id}")
        logging.info(f"Total execution time: {total_time:.2f} seconds")
        if processing_times:
             logging.info(f"Average processing time per frame batch: {avg_frame_time:.4f} seconds")
             # Note: FPS calculation might be less accurate with parallel processing timing
             # It reflects batch completion time, not individual frame throughput.
             logging.info(f"Average Batch FPS (overall): {avg_fps:.2f}")
        logging.info(f"Log file saved to: {LOG_FILE_PATH}") # Use imported log file path


if __name__ == "__main__":
    main()
