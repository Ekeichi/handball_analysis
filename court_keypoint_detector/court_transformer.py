import json
import cv2
import numpy as np
import sys
sys.path.append("../")
from utils import get_foot_position


class CourtTransformer:
    """Converts image pixels to court meters (and back) with a homography computed from clicked keypoints.

    Court frame, in meters: x along the court (0 = left goal line, 40 = right goal line),
    y across the court (0 = far sideline, 20 = camera-side sideline).
    Only valid for points on the floor: a ball in the air is projected several meters too far.
    """

    def __init__(self, keypoints_path, court_length=40, court_width=20):
        self.court_length = court_length
        self.court_width = court_width

        # Margins around the court lines, in meters
        # camera side: small, so substitutes and staff along the bench are excluded
        # far side and behind the goal lines: larger, for the referees standing near the corners
        self.margin_camera_side = 0.5
        self.margin_far_side = 0.5
        self.margin_goal_lines = 0.5

        with open(keypoints_path) as f:
            keypoints = json.load(f)["keypoints"]

        pixels = np.array([keypoint["pixel"] for keypoint in keypoints], dtype=np.float32)
        meters = np.array([keypoint["meters"] for keypoint in keypoints], dtype=np.float32)
        if len(pixels) < 4:
            raise ValueError(f"At least 4 keypoints are needed for a homography, {keypoints_path} has {len(pixels)}")

        # RANSAC drops a badly clicked keypoint (more than 0.3 m off) instead of letting it bend the whole homography
        self.image_to_court_matrix, _ = cv2.findHomography(pixels, meters, cv2.RANSAC, 0.3)
        self.court_to_image_matrix = np.linalg.inv(self.image_to_court_matrix)

    def transform_points(self, points, matrix):
        points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(points, matrix).reshape(-1, 2)

    def image_to_court(self, points):
        return self.transform_points(points, self.image_to_court_matrix)

    def court_to_image(self, points):
        return self.transform_points(points, self.court_to_image_matrix)

    def is_on_court(self, court_position):
        x, y = court_position
        return (-self.margin_goal_lines <= x <= self.court_length + self.margin_goal_lines
                and -self.margin_far_side <= y <= self.court_width + self.margin_camera_side)

    def filter_player_tracks(self, player_tracks, image_height=None):
        """Keeps only the players whose feet are on the court, and stores their position in meters.

        image_height: when given, boxes cut by the bottom of the image are dropped. Their feet are out of the frame,
        so the bottom of the box is not where they stand (on extrait.mp4 these are the substitutes on the bench,
        projected right onto the sideline).
        """
        filtered_tracks = []
        for frame_tracks in player_tracks:
            filtered_tracks.append({})
            if image_height is not None:
                frame_tracks = {track_id: track for track_id, track in frame_tracks.items() if track["bbox"][3] < image_height - 5}
            if not frame_tracks:
                continue

            track_ids = list(frame_tracks.keys())
            feet = [get_foot_position(frame_tracks[track_id]["bbox"]) for track_id in track_ids]
            court_positions = self.image_to_court(feet)

            for track_id, court_position in zip(track_ids, court_positions):
                if self.is_on_court(court_position):
                    filtered_tracks[-1][track_id] = {**frame_tracks[track_id], "position_m": court_position.tolist()}

        return filtered_tracks
