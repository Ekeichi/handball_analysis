from ultralytics import YOLO
import supervision as sv
import sys
sys.path.append("../utils")
from utils import read_stub, save_stub, batch_frames

class PlayerTracker:
    def __init__(self, model_path, fps=30):
        self.model = YOLO(model_path)

        # frame_rate: ByteTrack counts lost_track_buffer in 30 fps frames, so it must know the real fps
        # lost_track_buffer=60: a hidden player is kept for 2 s before getting a new id (default was 0.5 s at 60 fps)
        # track_activation_threshold=0.25: detections between 0.1 and 0.25 can only extend existing tracks (ByteTrack's
        #   second association, for half-hidden players); new tracks need 0.35 (the threshold + 0.1)
        # minimum_consecutive_frames=5: no new id for a detection that only flickers for a few frames
        # minimum_matching_threshold=0.9: accept a match down to 10% box overlap (default 20%), so a player who reappears
        #   slightly away from the predicted position keeps the same id. On extrait.mp4 this took ids from 116 to 85
        #   without more green/blue mixed tracks; lower it back to 0.8 if players start swapping ids
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=60,
            frame_rate=fps,
            minimum_consecutive_frames=5,
            minimum_matching_threshold=0.9,
        )

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
            # conf=0.1: keep low-confidence detections so ByteTrack can use them to follow half-hidden players;
            # it only creates tracks from confident ones (see track_activation_threshold)
            batch_detections = self.model.predict(batch, conf=0.1, imgsz=1280, classes=self.person_class_ids, agnostic_nms=True, device="mps", verbose=False)
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
