import numpy as np
import cv2

class Homography:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape")
        
        if source.shape[1] != 2:
            raise ValueError("Source and target must have 2 arrays")
        
        source = source.astype(np.float32)
        target = target.astype(np.float32)

        self.matrix, _ = cv2.findHomography(source, target, cv2.RANSAC, 0.3)
        if self.matrix is None:
            raise ValueError("Could not compute homography")

    def transform_points(self, points):
        if points.size == 0:
            return points
        if points.shape[1] != 2:
            raise ValueError("Points must be 2D arrays")
        
        points = points.reshape(-1,1,2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(points, self.matrix)
        return transformed_points.reshape(-1,2).astype(np.float32)
