import numpy as np
import supervision as sv

class CourtKeypointDrawer:
    def __init__(self):
        self.keypoint_color = "#ff2c2c"

    def _to_sv_keypoints(self, court_keypoints_json):
        """Convert the JSON list of keypoint dicts to an sv.KeyPoints object."""
        xy = np.array([kp["pixel"] for kp in court_keypoints_json], dtype=np.float32)
        return sv.KeyPoints(xy=xy[np.newaxis])  # shape (1, N, 2) — one "instance"

    def draw(self, frames, court_keypoints_json):
        # court_keypoints_json is a flat list of dicts (same for every frame)
        keypoints = self._to_sv_keypoints(court_keypoints_json)

        vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex(self.keypoint_color), radius=8
        )
        vertex_label_annotator = sv.VertexLabelAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            text_color=sv.Color.WHITE,
            text_scale=0.5,
            text_thickness=1,
        )

        # Draw in place and yield frame by frame to avoid keeping the whole video in memory
        for frame in frames:
            frame = vertex_annotator.annotate(frame, keypoints)
            frame = vertex_label_annotator.annotate(frame, keypoints)
            yield frame

        
        


        