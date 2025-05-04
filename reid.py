import torch
import numpy as np
import cv2
import config
import logging
from torchreid.reid.utils import FeatureExtractor

logging.basicConfig(level=config.LOG_LEVEL)

class ReIDExtractor:
    """
    Extracts appearance features from image crops using a pre-trained Re-ID model.
    """
    def __init__(self):
        """
        Initializes the FeatureExtractor from torchreid.
        """
        self.model_name = config.REID_MODEL_NAME
        self.device = config.REID_DEVICE

        # Check for CUDA availability if requested
        if self.device == 'cuda' and not torch.cuda.is_available():
            logging.warning("CUDA device requested but not available. Falling back to CPU.")
            self.device = 'cpu'

        try:
            logging.info(f"Loading Re-ID model '{self.model_name}' on device '{self.device}'...")
            # Initialize the FeatureExtractor
            # Note: FeatureExtractor expects a list of image paths or numpy arrays.
            # We will pass individual numpy arrays in the extract method.
            self.extractor = FeatureExtractor(
                model_name=self.model_name,
                model_path=None, # Use default pretrained weights
                device=self.device,
                verbose=False # Suppress torchreid loading messages
            )
            logging.info(f"Re-ID model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Re-ID model '{self.model_name}': {e}")
            raise

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        Basic preprocessing (currently just ensures correct format).
        TorchReID's extractor handles resizing and normalization internally.
        """
        # Ensure the crop is in BGR format if needed (OpenCV default)
        # The extractor expects RGB, but handles conversion if input is BGR numpy array.
        return crop

    def extract(self, frame: np.ndarray, bbox: list) -> np.ndarray | None:
        """
        Extracts a feature embedding from a cropped region of the frame.

        Args:
            frame (np.ndarray): The full input frame (BGR format).
            bbox (list): Bounding box coordinates [x1, y1, x2, y2].

        Returns:
            np.ndarray | None: A 1D NumPy array representing the normalized feature
                               embedding, or None if extraction fails.
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)

            # Basic validation for bounding box coordinates
            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # Check if the bounding box is valid (has width and height)
            if x1 >= x2 or y1 >= y2:
                logging.warning(f"Invalid bounding box received: {bbox}. Skipping feature extraction.")
                return None

            # Crop the person image from the frame
            crop = frame[y1:y2, x1:x2]

            # Check if crop is empty (can happen with invalid boxes)
            if crop.size == 0:
                 logging.warning(f"Empty crop generated from bbox: {bbox}. Skipping feature extraction.")
                 return None

            # Preprocess the crop (if necessary, though extractor handles much of it)
            processed_crop = self._preprocess(crop)

            # Extract features - extractor expects a list of images
            # It returns a torch tensor, shape (1, feature_dim)
            feature_tensor = self.extractor([processed_crop]) # Pass as a list

            # Convert to numpy array, remove batch dimension, and ensure it's 1D
            feature_np = feature_tensor.cpu().numpy().flatten()

            # Normalize the feature vector (L2 normalization)
            norm = np.linalg.norm(feature_np)
            if norm == 0:
                logging.warning("Feature vector norm is zero. Cannot normalize.")
                # Return zero vector or handle as appropriate
                return np.zeros_like(feature_np)

            normalized_feature = feature_np / norm
            # logging.debug(f"Extracted feature of shape {normalized_feature.shape}")
            return normalized_feature

        except Exception as e:
            logging.error(f"Error during feature extraction for bbox {bbox}: {e}", exc_info=True)
            return None

# Example Usage (can be run standalone for testing)
if __name__ == '__main__':
    print("Initializing ReIDExtractor...")
    try:
        extractor = ReIDExtractor()
        print("Extractor initialized.")

        # Create a dummy frame and bounding box
        dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        # A sample bounding box within the frame
        # Make sure the box has reasonable width and height
        sample_bbox = [100, 150, 160, 280] # x1, y1, x2, y2 (width=60, height=130)

        print(f"Extracting feature for bbox {sample_bbox}...")
        feature = extractor.extract(dummy_frame, sample_bbox)

        if feature is not None:
            print(f"Successfully extracted feature vector.")
            print(f"  Shape: {feature.shape}")
            print(f"  Norm: {np.linalg.norm(feature):.4f}") # Should be close to 1.0
            print(f"  First 5 elements: {feature[:5]}")
        else:
            print("Feature extraction failed.")

        # Test with an invalid bbox (zero width)
        invalid_bbox = [100, 150, 100, 280]
        print(f"\nTesting with invalid bbox {invalid_bbox}...")
        feature_invalid = extractor.extract(dummy_frame, invalid_bbox)
        if feature_invalid is None:
            print("Correctly handled invalid bbox: returned None.")
        else:
            print("Error: Did not handle invalid bbox correctly.")

    except Exception as e:
        print(f"An error occurred during testing: {e}")

    print("\nReIDExtractor test finished.")