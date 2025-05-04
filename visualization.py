import cv2
import numpy as np
from collections import deque, defaultdict

# Define a list of colors for drawing (BGR format)
# Using a simple set of distinct colors
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (192, 192, 192), (128, 128, 128), (128, 0, 0),
    (128, 128, 0), (0, 128, 0), (128, 0, 128),
    (0, 128, 128), (0, 0, 128), (75, 0, 130),
    (255, 165, 0), (255, 215, 0), (184, 134, 11)
]

# Dictionary to store motion trails for each global ID
# Using defaultdict simplifies adding new IDs
# Max trail length can be adjusted here or via config
TRAIL_MAXLEN = 30
trail_buffers = defaultdict(lambda: deque(maxlen=TRAIL_MAXLEN))

def get_color(id_num):
    """Gets a consistent color for a given ID."""
    if id_num < 0: # Use a default color for invalid/local IDs
        return (0, 0, 255) # Red
    return COLORS[id_num % len(COLORS)]

def draw_bounding_box(frame, bbox, label, color):
    """Draws a bounding box and label on the frame."""
    try:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    except Exception as e:
        print(f"Error drawing bounding box: {e}") # Use logging in main app

def draw_motion_trail(frame, global_id, bbox):
    """Adds current center to the trail buffer and draws the trail."""
    if global_id < 0: # Don't draw trails for non-global IDs
        return

    try:
        x1, y1, x2, y2 = map(int, bbox)
        center = ((x1 + x2) // 2, (y1 + y2) // 2) # Calculate center point
        color = get_color(global_id)

        buffer = trail_buffers[global_id]
        buffer.appendleft(center) # Add new center to the left of the deque

        # Draw lines connecting points in the buffer
        for i in range(1, len(buffer)):
            if buffer[i-1] is None or buffer[i] is None:
                continue
            # Calculate thickness based on position in buffer (fades out)
            thickness = int(np.sqrt(TRAIL_MAXLEN / float(i + 1)) * 1.5) # Non-linear fade
            # thickness = int(5 * (1 - i / len(buffer))) # Linear fade from guide
            cv2.line(frame, buffer[i-1], buffer[i], color, thickness)
    except Exception as e:
        print(f"Error drawing motion trail: {e}") # Use logging in main app

def visualize_tracks(frame, tracks_data):
    """
    Draws bounding boxes and motion trails for all tracks in the frame.
    'tracks_data' should be a list of dicts, each containing at least:
    {'bbox': [x1,y1,x2,y2], 'track_id': local_id, 'global_id': gid}
    """
    for track in tracks_data:
        bbox = track.get('bbox')
        local_id = track.get('track_id', -1)
        global_id = track.get('global_id', -1)

        if bbox is None:
            continue

        color = get_color(global_id)
        label = f"GID:{global_id}" if global_id != -1 else f"TID:{local_id}"

        draw_bounding_box(frame, bbox, label, color)
        draw_motion_trail(frame, global_id, bbox)

    return frame # Return the modified frame
