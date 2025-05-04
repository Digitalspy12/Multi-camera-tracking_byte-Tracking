import cv2
import config
import logging

logging.basicConfig(level=config.LOG_LEVEL)

class VideoLoader:
    """
    Handles loading and reading synchronized frames from multiple video files.
    """
    def __init__(self, video_paths):
        """
        Initializes VideoCapture objects for each video path.

        Args:
            video_paths (list): A list of strings containing paths to video files.
        """
        self.video_paths = video_paths
        self.caps = []
        self.num_cameras = len(video_paths)
        self.active = True # Flag to indicate if all streams are active

        for i, path in enumerate(self.video_paths):
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                logging.error(f"Error opening video file: {path}")
                # Handle error appropriately - maybe raise exception or set active=False
                self.active = False
                # Release any already opened captures
                self.release()
                raise IOError(f"Could not open video file: {path}")
            self.caps.append(cap)
            logging.info(f"Successfully opened video {i}: {path}")

        if not self.caps:
            logging.warning("No video files were successfully opened.")
            self.active = False

    def read_frames(self):
        """
        Reads the next frame from each video stream.

        Returns:
            tuple: A tuple containing:
                - list: A list of frames (NumPy arrays), one for each camera.
                        Contains None for cameras that have ended or failed.
                - bool: True if all video streams are still active and returning frames,
                        False otherwise.
        """
        if not self.active:
            return [None] * self.num_cameras, False

        frames = []
        all_read_successful = True
        for i, cap in enumerate(self.caps):
            ret, frame = cap.read()
            if not ret:
                logging.info(f"Video stream {i} ({self.video_paths[i]}) ended or failed to read.")
                frames.append(None)
                all_read_successful = False # Mark as inactive if any stream ends
            else:
                frames.append(frame)

        # Update overall active status - if any stream failed this round, we stop
        if not all_read_successful:
            self.active = False

        return frames, self.active

    def release(self):
        """
        Releases all VideoCapture objects.
        """
        logging.info("Releasing video capture objects.")
        for cap in self.caps:
            if cap.isOpened():
                cap.release()
        self.active = False

    def get_num_cameras(self):
        """Returns the number of camera streams."""
        return self.num_cameras

# Example Usage (can be run standalone for testing)
if __name__ == '__main__':
    try:
        # Use paths from config for testing
        loader = VideoLoader(config.VIDEO_PATHS)
        num_cams = loader.get_num_cameras()
        print(f"Number of cameras: {num_cams}")

        frame_count = 0
        while loader.active:
            frames, active = loader.read_frames()
            if not active:
                print("Finished processing videos.")
                break

            print(f"Read frame {frame_count}:")
            for i, frame in enumerate(frames):
                if frame is not None:
                    print(f"  Camera {i}: shape {frame.shape}")
                    # Example: Display frames (optional)
                    # cv2.imshow(f"Camera {i}", frame)
                else:
                    print(f"  Camera {i}: Stream ended.")

            frame_count += 1

            # Example: Break after a few frames for testing
            if frame_count >= 5:
                 print("Stopping after 5 frames for testing.")
                 break

            # Example: Press 'q' to quit visualization window if shown
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    except IOError as e:
        print(f"Initialization failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'loader' in locals():
            loader.release()
        # cv2.destroyAllWindows() # If using imshow
        print("Video loader released.")