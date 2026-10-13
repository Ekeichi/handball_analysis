from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import sys
sys.path.append("../utils")
from utils import read_stub, save_stub, batch_frames, get_center_of_bbox, measure_distance

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def detect_frames(self, frames):
        # Keep batches small: the model runs at imgsz=1280 (its training size), and big batches exceed YOLO's NMS time limit and whole batches are silently dropped
        batch_size = 2

        # Yield detections batch by batch so we never keep all the results (and their images) in memory
        for batch in batch_frames(frames, batch_size):
            batch_detections = self.model.predict(batch, conf=0.1, device="mps", imgsz=1280, verbose=False)
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
            cls_names = detection.names
            
            cls_names_inv = {}
            for key, value in cls_names.items():
                cls_names_inv[value] = key

            detection_supervision = sv.Detections.from_ultralytics(detection)
            tracks.append({})
            chosen_bbox = None
            max_confidence = 0
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = frame_detection[2]
                if cls_id == cls_names_inv["ball"]:
                    if max_confidence < confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence
            
            if chosen_bbox is not None:
                tracks[frame_num][1] = {"bbox" : chosen_bbox}

        save_stub(stub_path, tracks)

        return tracks
    
    def is_reachable(self, detection_a, detection_b, max_speed):
        frame_a, center_a = detection_a
        frame_b, center_b = detection_b
        return measure_distance(center_a, center_b) <= max_speed * abs(frame_b - frame_a)

    def remove_wrong_detection(self, ball_positions):
        # At 60 fps a fast pass moves 30-40 px per frame, so 50 still rejects jumps to another part of the image
        max_speed = 70
        # Segments shorter than this are suspicious: a false ball (head, shoe) rarely lasts more than a couple of frames
        min_segment_length = 3

        detections = []
        for frame_num, position in enumerate(ball_positions):
            bbox = position.get(1, {}).get("bbox", [])
            if len(bbox) > 0:
                detections.append((frame_num, get_center_of_bbox(bbox)))

        # Split detections into segments of consecutive, physically reachable positions.
        # Each detection is compared to its direct predecessor, so one bad detection can't become the reference for the next ones
        segments = []
        for detection in detections:
            if segments and self.is_reachable(segments[-1][-1], detection, max_speed):
                segments[-1].append(detection)
            else:
                segments.append([detection])

        # A short segment is a detour when the ball's path makes sense without it:
        # the segments around it connect to each other directly, or it sits at the start/end next to a long segment
        wrong_frames = []
        kept_segments = []
        for i, segment in enumerate(segments):
            if len(segment) < min_segment_length:
                previous_segment = kept_segments[-1] if kept_segments else None
                next_segment = segments[i + 1] if i + 1 < len(segments) else None

                if previous_segment and next_segment:
                    is_detour = self.is_reachable(previous_segment[-1], next_segment[0], max_speed)
                elif previous_segment:
                    is_detour = len(previous_segment) >= min_segment_length
                elif next_segment:
                    is_detour = len(next_segment) >= min_segment_length
                else:
                    is_detour = False

                if is_detour:
                    wrong_frames += [frame_num for frame_num, _ in segment]
                    continue

            kept_segments.append(segment)

        for frame_num in wrong_frames:
            ball_positions[frame_num] = {}

        return ball_positions
    
    def interpolate_ball_positions(self, ball_positions, max_gap=30):
        # max_gap=30 frames is half a second at 60 fps: beyond that a straight line between
        # the two detections no longer says where the ball really was, so the gap stays empty
        bboxes = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        bboxes = [bbox if len(bbox) > 0 else [np.nan] * 4 for bbox in bboxes]
        df_ball_positions = pd.DataFrame(bboxes, columns=["x1", "y1", "x2", "y2"])

        # Length of the gap each missing frame belongs to (each gap starts right after a detection)
        detected = df_ball_positions["x1"].notna()
        gap_length = (~detected).groupby(detected.cumsum()).transform("sum")

        # limit_area="inside": only fill between two detections, never before the first or after the last one
        df_ball_positions = df_ball_positions.interpolate(limit_area="inside")
        df_ball_positions[~detected & (gap_length > max_gap)] = np.nan

        ball_positions = []
        for bbox, is_detected in zip(df_ball_positions.to_numpy().tolist(), detected):
            if np.isnan(bbox[0]):
                ball_positions.append({})
            else:
                # Flag filled positions: they are a guess, so the possession detector trusts them less
                ball_positions.append({1: {"bbox": bbox, "interpolated": not is_detected}})
        return ball_positions
        
                

    


    