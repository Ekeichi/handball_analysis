from ultralytics import YOLO
import supervision as sv
import sys
sys.path.append("../utils")
from utils import read_stub, save_stub, batch_frames

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

        # The model's player/referee/goalkeeper split is unreliable on our footage (it learned other jersey colors),
        # so all classes are treated as one "person" class; roles are decided later from jersey colors
        cls_names_inv = {value: key for key, value in self.model.names.items()}
        self.person_class_ids = [cls_names_inv["player"], cls_names_inv["referee"]]
        self.person_class_id = cls_names_inv["player"]

    def detect_frames(self, frames):
        # Keep batches small: at imgsz=1280, big batches exceed YOLO's NMS time limit and whole batches are silently dropped
        batch_size = 4

        # Yield detections batch by batch so we never keep all the results (and their images) in memory
        for batch in batch_frames(frames, batch_size):
            # classes: only persons, so a "goal" box can't suppress the goalkeeper standing inside it
            # agnostic_nms: one box per person even when the model outputs both a "player" and a "referee" box on it
            # conf=0.3: the score is split between player and referee, so real players often score 0.3-0.5
            batch_detections = self.model.predict(batch, conf=0.3, imgsz=1280, classes=self.person_class_ids, agnostic_nms=True, device="mps", verbose=False)
            for detection in batch_detections:
                yield detection
    
    def get_object_tracks(self, frames, frame_count, read_from_stub=False, stub_path=None):
        
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None:
            if len(tracks) == frame_count:
                return tracks

        detections = self.detect_frames(frames)
        tracks = []

        for frame_num, detection in enumerate(detections):
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Same class for everyone, so a player/referee label flip between frames can't break a track
            detection_supervision.class_id[:] = self.person_class_id

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks.append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                track_id = frame_detection[4]

                tracks[frame_num][track_id] = {"bbox" : bbox}

        save_stub(stub_path, tracks)

        return tracks
