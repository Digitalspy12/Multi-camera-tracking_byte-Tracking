import numpy as np
from scipy.spatial.distance import cosine
import config
import logging

logging.basicConfig(level=config.LOG_LEVEL)

class GlobalIDManager:
    """
    Manages global identities based on feature embeddings. Assigns a consistent
    global ID to tracks across cameras based on feature similarity.
    """
    def __init__(self):
        """
        Initializes the global ID manager.
        """
        self.global_features = {}  # Stores {global_id: representative_feature_vector}
        self.next_id = 0
        self.threshold = config.REID_COSINE_THRESHOLD
        logging.info(f"GlobalIDManager initialized with threshold: {self.threshold}")

    def match(self, feature: np.ndarray) -> int:
        """
        Matches an input feature vector against the stored global features.
        Assigns an existing global ID if similarity is high enough, otherwise
        creates a new global ID.

        Args:
            feature (np.ndarray): The normalized 1D feature vector for the current track.

        Returns:
            int: The assigned global ID.
        """
        if feature is None:
            logging.warning("Received None feature vector. Cannot perform matching.")
            # Decide how to handle this - perhaps return a special ID like -1?
            # For now, let's log and potentially skip assignment, but returning an ID is needed.
            # Returning a new ID might be problematic. Let's return -1 for now.
            return -1 # Indicate failure or inability to match

        # Ensure feature is a 1D numpy array
        if not isinstance(feature, np.ndarray) or feature.ndim != 1:
             logging.error(f"Invalid feature type or dimension received: {type(feature)}, ndim={feature.ndim if isinstance(feature, np.ndarray) else 'N/A'}")
             return -1 # Indicate error

        # Handle the case where this is the first feature seen
        if not self.global_features:
            gid = self.next_id
            self.global_features[gid] = feature
            self.next_id += 1
            logging.debug(f"First feature seen. Assigned new Global ID: {gid}")
            return gid

        # Calculate cosine similarity with all stored features
        similarities = {}
        for gid, stored_feat in self.global_features.items():
            # Cosine distance = 1 - cosine similarity
            # Ensure features are valid before calculating distance
            if stored_feat is not None and stored_feat.shape == feature.shape:
                try:
                    # distance = cosine(feature, stored_feat) # Returns distance
                    # similarity = 1 - distance
                    # Using dot product for normalized vectors is equivalent and potentially faster
                    similarity = np.dot(feature, stored_feat)
                    similarities[gid] = similarity
                except Exception as e:
                    logging.error(f"Error calculating similarity between input feature and stored feature for GID {gid}: {e}")
                    similarities[gid] = -1 # Assign low similarity on error
            else:
                 logging.warning(f"Skipping comparison with GID {gid} due to invalid stored feature or shape mismatch.")
                 similarities[gid] = -1


        if not similarities: # No valid comparisons could be made
             logging.warning("No valid stored features to compare against. Assigning new ID.")
             gid = self.next_id
             self.global_features[gid] = feature
             self.next_id += 1
             return gid

        # Find the best match
        best_gid = max(similarities, key=similarities.get)
        best_sim = similarities[best_gid]

        # Decide whether to assign existing ID or create a new one
        if best_sim >= self.threshold:
            # Match found! Assign existing global ID.
            # Optional: Update the stored feature (e.g., moving average)
            self.global_features[best_gid] = self._update_feature(self.global_features[best_gid], feature)
            logging.debug(f"Match found. Assigning existing Global ID: {best_gid} (Similarity: {best_sim:.4f})")
            return best_gid
        else:
            # No sufficiently similar feature found. Create new global ID.
            gid = self.next_id
            self.global_features[gid] = feature
            self.next_id += 1
            logging.debug(f"No match above threshold ({self.threshold}). Assigned new Global ID: {gid} (Best sim: {best_sim:.4f} for GID {best_gid})")
            return gid

    # Optional: Method to update stored features over time
    def _update_feature(self, old_feature, new_feature, alpha=0.5):
        """ Simple moving average update. """
        updated_feature = alpha * new_feature + (1 - alpha) * old_feature
        # Re-normalize after averaging
        norm = np.linalg.norm(updated_feature)
        return updated_feature / norm if norm > 0 else updated_feature


# Example Usage (can be run standalone for testing)
if __name__ == '__main__':
    print("Initializing GlobalIDManager...")
    manager = GlobalIDManager()
    # Use a threshold from config or override for testing
    manager.threshold = 0.7
    print(f"Manager initialized with threshold: {manager.threshold}")

    # Simulate some feature vectors (assuming 512-dim from osnet_x0_25)
    feature_dim = 512
    feat1 = np.random.rand(feature_dim)
    feat1 /= np.linalg.norm(feat1)

    feat2 = np.random.rand(feature_dim)
    feat2 /= np.linalg.norm(feat2)

    # Create a feature similar to feat1
    feat1_similar = feat1 + np.random.rand(feature_dim) * 0.1 # Add small noise
    feat1_similar /= np.linalg.norm(feat1_similar)

    print("\nMatching first feature...")
    gid1 = manager.match(feat1)
    print(f"Assigned Global ID: {gid1}")
    print(f"Current registry size: {len(manager.global_features)}")

    print("\nMatching second (different) feature...")
    gid2 = manager.match(feat2)
    print(f"Assigned Global ID: {gid2}")
    print(f"Current registry size: {len(manager.global_features)}")
    assert gid1 != gid2, "Different features should get different IDs initially"

    print("\nMatching a feature similar to the first one...")
    gid1_again = manager.match(feat1_similar)
    print(f"Assigned Global ID: {gid1_again}")
    print(f"Current registry size: {len(manager.global_features)}")
    assert gid1 == gid1_again, f"Similar feature should match existing ID {gid1}, but got {gid1_again}"
    print(f"Similarity between feat1 and feat1_similar: {np.dot(feat1, feat1_similar):.4f}")


    print("\nMatching None feature...")
    gid_none = manager.match(None)
    print(f"Assigned Global ID for None: {gid_none}")
    assert gid_none == -1, "None feature should result in ID -1"


    print("\nGlobalIDManager test finished.")