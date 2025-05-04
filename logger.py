import json
import logging
import config
from datetime import datetime
import atexit # To ensure file is closed on exit

# --- Configuration ---
LOG_FILE_PATH = config.LOG_FILE.replace('.csv', '.jsonl') if hasattr(config, 'LOG_FILE') else 'tracking_log.jsonl'

# --- Global File Handle ---
# Using a global handle simplifies access but requires careful management
log_file_handle = None

def initialize_logger():
    """Opens the JSON Lines log file in append mode."""
    global log_file_handle
    try:
        # Open file in append mode, create if it doesn't exist
        log_file_handle = open(LOG_FILE_PATH, 'a')
        logging.info(f"Initialized JSON Lines logger. Logging to: {LOG_FILE_PATH}")
        # Register the cleanup function to be called on normal program exit
        atexit.register(close_logger)
    except IOError as e:
        logging.error(f"Error opening JSON Lines log file {LOG_FILE_PATH}: {e}")
        log_file_handle = None

def log_event(event_data: dict):
    """
    Writes a single event dictionary as a JSON line to the log file.
    Adds a timestamp automatically.

    Args:
        event_data (dict): A dictionary containing the data to log.
                           Expected keys: 'frame_id', 'camera_id', 'track_id',
                                          'global_id', 'bbox'.
    """
    if log_file_handle is None:
        logging.warning("JSON Lines logger not initialized. Cannot log event.")
        return

    try:
        # Add timestamp
        event_data['timestamp'] = datetime.now().isoformat()
        # Write the dictionary as a JSON string followed by a newline
        log_file_handle.write(json.dumps(event_data) + '\n')
        # Flush immediately to ensure data is written (can impact performance)
        # Consider flushing less frequently (e.g., after batches) if performance is critical
        log_file_handle.flush()
    except Exception as e:
        logging.error(f"Error writing JSON log entry: {e} - Data: {event_data}")

def log_batch_events(events: list):
    """
    Writes a batch of event dictionaries as JSON lines.

    Args:
        events (list): A list of dictionaries to log.
    """
    if log_file_handle is None:
        logging.warning("JSON Lines logger not initialized. Cannot log batch.")
        return

    try:
        timestamp = datetime.now().isoformat()
        lines = []
        for event in events:
            # Basic check for expected keys (optional but good practice)
            if all(k in event for k in ['frame_id', 'camera_id', 'track_id', 'global_id', 'bbox']):
                event['timestamp'] = timestamp # Use the same timestamp for the batch
                lines.append(json.dumps(event))
            else:
                 logging.warning(f"Skipping malformed log event in batch: {event}")

        if lines:
            log_file_handle.write('\n'.join(lines) + '\n')
            log_file_handle.flush() # Flush after writing the batch
    except Exception as e:
        logging.error(f"Error writing JSON log batch: {e}")


def close_logger():
    """Closes the log file."""
    global log_file_handle
    if log_file_handle:
        try:
            log_file_handle.flush()
            log_file_handle.close()
            logging.info(f"Closed JSON Lines log file: {LOG_FILE_PATH}")
            log_file_handle = None
        except Exception as e:
            logging.error(f"Error closing JSON Lines log file {LOG_FILE_PATH}: {e}")

# Initialize the logger when the module is imported
initialize_logger()

# Example Usage (can be run standalone for testing)
if __name__ == '__main__':
    print("Testing JSON Lines Logger...")
    # Ensure the logger is initialized (it should be by module import)
    if log_file_handle is None:
        print("Re-initializing logger for test...")
        initialize_logger() # Explicitly call if running standalone might bypass initial import call

    if log_file_handle:
        print(f"Logging test events to: {LOG_FILE_PATH}")
        # Log a single event
        test_event_single = {'frame_id': 0, 'camera_id': 0, 'track_id': 1, 'global_id': 10, 'bbox': [10, 20, 50, 80]}
        print(f"Logging single: {test_event_single}")
        log_event(test_event_single)

        # Log a batch of events
        test_events_batch = [
            {'frame_id': 1, 'camera_id': 0, 'track_id': 1, 'global_id': 10, 'bbox': [12, 22, 52, 82]},
            {'frame_id': 1, 'camera_id': 1, 'track_id': 5, 'global_id': 15, 'bbox': [100, 120, 150, 180]},
            {'frame_id': 1, 'camera_id': 0, 'track_id': 2, 'global_id': 11, 'bbox': [200, 210, 240, 250]},
            {'frame_id': 1, 'camera_id': 1, 'track_id': 'invalid'} # Malformed event for testing
        ]
        print(f"Logging batch: {test_events_batch}")
        log_batch_events(test_events_batch)

        print("Closing logger (will happen automatically on exit, but calling explicitly for test)...")
        # close_logger() # atexit handles this, but can call manually

        print(f"\nTest finished. Check the content of '{LOG_FILE_PATH}'.")
        print("Note: Run this script multiple times to see append behavior.")
    else:
        print("Logger initialization failed. Cannot run tests.")
