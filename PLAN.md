# Plan: Multi-Camera Pedestrian Tracking System (Core Logic & Prototyping)

**1. Goal:** Develop a prototype system that takes multiple prerecorded video files as input, detects and tracks pedestrians within each video stream using YOLOv8 + ByteTrack (via Ultralytics), extracts appearance features (Re-ID), assigns consistent global IDs across cameras, and logs the results.

**2. Core Components & Approach:**

*   **Detection & Tracking:** Utilize the integrated `model.track()` method from the `ultralytics` library with the `bytetrack.yaml` configuration. This simplifies the pipeline by combining detection and per-camera tracking into a single step. We'll start with a lightweight model like `yolov8n.pt` or `yolov8s.pt` for CPU performance.
*   **Re-Identification (Re-ID):** Employ a lightweight pretrained Re-ID model (e.g., OSNet) via the `torchreid` library to extract appearance embeddings for each tracked person.
*   **Global ID Management:** Implement a manager that compares Re-ID embeddings from different camera tracks using cosine similarity to assign consistent global IDs.
*   **Input:** Process multiple prerecorded video files (e.g., `.mp4`).
*   **Environment:** Target a CPU-based environment initially (like local machine without a dedicated GPU) for prototyping.

**3. Proposed Code Structure (Modules):**

We'll structure the code into the following Python modules:

*   **`config.py`:** Stores configuration parameters (model paths, Re-ID thresholds, video file paths, etc.).
*   **`video_loader.py`:**
    *   Defines a class/functions to handle opening multiple video files (`cv2.VideoCapture`).
    *   Provides a method to read the next synchronized frame (or frame batch) from all video sources. Basic synchronization can be achieved by reading one frame from each video per iteration.
*   **`tracker.py`:**
    *   Defines a class `IntegratedTracker`.
    *   `__init__`: Loads the YOLOv8 model (`YOLO('yolov8s.pt')`).
    *   `process_frame(frame)`: Takes a single frame, runs `model.track(frame, persist=True, tracker='bytetrack.yaml', conf=0.3, classes=[0])` (class 0 is 'person' in COCO), and returns track data (track IDs, bounding boxes). `persist=True` is important for tracking across frames.
*   **`reid.py`:**
    *   Defines a class `ReIDExtractor`.
    *   `__init__`: Loads a pretrained Re-ID model using `torchreid.utils.FeatureExtractor` (e.g., `model_name='osnet_x0_25'`, device='cpu').
    *   `extract(frame, bbox)`: Crops the person from the frame using the bounding box, preprocesses the crop, extracts the feature embedding, normalizes it, and returns the 1D NumPy array.
*   **`global_id.py`:**
    *   Defines a class `GlobalIDManager`.
    *   Maintains a dictionary mapping `global_id` to representative feature embeddings.
    *   `match(feature)`: Compares the input feature to stored embeddings using cosine similarity. Returns an existing `global_id` if similarity > threshold; otherwise, assigns a new `global_id`, stores the feature, and returns the new ID.
*   **`logger.py`:**
    *   Defines a class/function `CsvLogger`.
    *   Handles writing tracking results (frame index, camera ID, local track ID, global ID, bounding box) to a CSV file.
*   **`main.py`:**
    *   Initializes all modules (VideoLoader, Tracker, ReIDExtractor, GlobalIDManager, Logger).
    *   Contains the main processing loop:
        *   Read synchronized frames from VideoLoader.
        *   For each camera's frame:
            *   Get tracks using `IntegratedTracker.process_frame()`.
            *   For each track:
                *   Extract Re-ID feature using `ReIDExtractor.extract()`.
                *   Get/assign global ID using `GlobalIDManager.match()`.
                *   Log the results (frame, cam\_id, track\_id, global\_id, bbox) using `CsvLogger`.
        *   (Optional) Add basic visualization using OpenCV (`cv2.imshow`).
        *   Handle loop termination (end of videos).

**4. Development & Prototyping Steps:**

1.  **Environment Setup:** Create a Python environment and install necessary libraries: `ultralytics`, `torch`, `torchvision`, `opencv-python`, `torchreid`, `numpy`, `scipy`.
2.  **Implement `config.py`:** Define initial paths and parameters.
3.  **Implement `video_loader.py`:** Test loading and reading frames from 2+ sample video files.
4.  **Implement `tracker.py`:** Test `model.track()` on single frames from a video. Verify it returns track IDs and bounding boxes for persons.
5.  **Implement `reid.py`:** Test feature extraction on sample cropped person images. Ensure it returns consistent-shape embeddings.
6.  **Implement `global_id.py`:** Unit test the matching logic with dummy feature vectors.
7.  **Implement `logger.py`:** Test writing sample data to a CSV file.
8.  **Integrate in `main.py`:** Combine all modules in the main loop.
9.  **Initial Test Run:** Use very short sample videos (e.g., 5-10 seconds) to test the end-to-end pipeline on CPU. Debug synchronization, data flow, and ID assignment.
10. **Refine & Iterate:** Adjust Re-ID thresholds, tracker confidence, etc., based on initial results.

**5. Pipeline Visualization (Mermaid):**

```mermaid
graph TD
    subgraph System Pipeline
        A[Video Input Files] --> B(Video Loader);
        B -- Frames per Camera --> C{Main Loop};

        subgraph Per-Camera Processing in Loop
            C -- Frame --> D[Integrated Tracker (YOLOv8 + ByteTrack)];
            D -- Tracks (Local ID, BBox) --> E{Process Each Track};
            C -- Frame --> E;
            E -- Frame, BBox --> F[Re-ID Extractor];
            F -- Feature Embedding --> G[Global ID Manager];
            G -- Global ID --> H{Combine Results};
            E -- Local ID, BBox --> H;
        end

        H -- Log Entry --> I[Logger];
        I -- Write --> J[Output CSV Log];
        C -- (Optional) Frame, BBox, Global ID --> K[Visualization (OpenCV)];
    end

    style D fill:#f9f,stroke:#333,stroke-width:2px
    style F fill:#ccf,stroke:#333,stroke-width:2px
    style G fill:#cfc,stroke:#333,stroke-width:2px
```

**6. Next Steps:**

*   Review this plan. Does it align with your expectations for the prototype? Are there any adjustments you'd like to make?
*   Once approved, I can write this plan to a `PLAN.md` file in the workspace.
*   After saving the plan, I will suggest switching to "Code" mode to begin implementing the modules according to these steps.