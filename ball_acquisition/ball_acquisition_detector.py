import sys
sys.path.append("../")
from utils import measure_distance, get_center_of_bbox

class BallAcquisitionDetector:
    def __init__(self):
        self.position_threshold = 50
        self.min_frames = 7
        self.containment_threshold = 0.8
        # Frames without candidate tolerated before the current streak is dropped (occlusion, missed detection)
        self.max_missed_frames = 4
        # Frames the confirmed holder keeps the ball without being seen (ball in flight, out of play)
        self.max_holder_absence = 25
        # Ball speed (px/frame) relative to the candidate's bbox height above which the ball is in flight, not held
        self.max_ball_speed_ratio = 0.25
        # Relative height in the candidate's bbox (0 = head, 1 = feet) below which the ball is at leg level
        self.low_ball_ratio = 0.6

    def get_key_player_assignments_points(self, player_bbox, ball_center):
        ball_center_x = ball_center[0]
        ball_center_y = ball_center[1]
        
        x1, y1, x2, y2 = player_bbox
        width = x2 - x1
        height = y2 - y1
        
        output_points = []
        
        if ball_center_y > y1 and ball_center_y < y2:
            output_points.append((x1, ball_center_y))
            output_points.append((x2, ball_center_y))

        if ball_center_x > x1 and ball_center_x < x2:
            output_points.append((ball_center_x, y1))
            output_points.append((ball_center_x, y2))
        
        output_points += [
            (x1, y1), #top left corner
            (x1, y2), #bottom left corner
            (x2, y1), #top right corner
            (x2, y2), #bottom right corner
            (x1 + width//2, y1), #top center
            (x1 + width//2, y2), #bottom center
            (x1, y1 + height//2), #left center
            (x2, y1 + height//2) #right center
        ]

        return output_points
    
    def find_minimum_distance_to_ball(self,ball_center,player_bbox):
        key_points = self.get_key_player_assignments_points(player_bbox,ball_center)
        return min(measure_distance(key_point,ball_center) for key_point in key_points)
    
    def calculate_ball_containment_ratio(self, player_bbox, ball_bbox):
        px1, py1, px2, py2 = player_bbox
        bx1, by1, bx2, by2 = ball_bbox

        ball_area = max(0, bx2 - bx1) * max(0, by2 - by1)
        if ball_area == 0:
            return 0.0

        inter_w = max(0, min(px2, bx2) - max(px1, bx1))
        inter_h = max(0, min(py2, by2) - max(py1, by1))
        
        return (inter_w * inter_h) / ball_area

    def find_best_candidate_for_possession(self, ball_center, player_tracks_frame, ball_bbox):
        high_containment_players = []
        regular_distance_players = []
        
        for player_id, player_info in player_tracks_frame.items():
            player_bbox = player_info.get('bbox', [])
            if not player_bbox:
                continue

            containment = self.calculate_ball_containment_ratio(player_bbox, ball_bbox)
            min_distance = self.find_minimum_distance_to_ball(ball_center, player_bbox)

            if containment > self.containment_threshold:
                high_containment_players.append((player_id, containment, player_bbox))
            else:
                regular_distance_players.append((player_id, min_distance, player_bbox))
        
        # First priority high containment players
        if high_containment_players:
            best_candidate = max(high_containment_players, key=lambda x:x[1])
            return best_candidate[0]
        
        # Second priority regular distance players with position threshold
        if regular_distance_players:
            best_candidate = min(regular_distance_players, key=lambda x:x[1])
            if best_candidate[1] < self.position_threshold:
                return best_candidate[0]
        
        return -1

    def get_ball_speeds(self, ball_tracks):
        centers = []
        for frame in ball_tracks:
            bbox = frame.get(1, {}).get('bbox', [])
            centers.append(get_center_of_bbox(bbox) if bbox else None)
        speeds = [0.0] * len(centers)

        for frame_num, center in enumerate(centers):
            if center is None:
                continue
            # Centered difference over the neighbours to smooth the detection jitter
            neighbours = [centers[f] for f in (frame_num - 1, frame_num + 1) if 0 <= f < len(centers) and centers[f] is not None]
            if len(neighbours) == 2:
                speeds[frame_num] = measure_distance(neighbours[0], neighbours[1]) / 2
            elif len(neighbours) == 1:
                speeds[frame_num] = measure_distance(neighbours[0], center)

        return speeds

    def detect_ball_possession(self, player_tracks, ball_tracks):
        num_frames = len(ball_tracks)
        possession_list = [-1] * num_frames
        ball_speeds = self.get_ball_speeds(ball_tracks)

        current_holder = -1
        holder_last_seen = -1
        candidate = -1
        candidate_start = -1
        candidate_count = 0
        missed_frames = 0

        for frame_num in range(num_frames):
            # No ball detected and no player close enough are handled the same way: no candidate
            best_player_id = -1
            ball_info = ball_tracks[frame_num].get(1, {})
            ball_bbox = ball_info.get('bbox', [])
            if ball_bbox:
                ball_center = get_center_of_bbox(ball_bbox)
                best_player_id = self.find_best_candidate_for_possession(ball_center, player_tracks[frame_num], ball_bbox)

            if best_player_id != -1:
                _, y1, _, y2 = player_tracks[frame_num][best_player_id]['bbox']
                height = y2 - y1

                # A fast ball is a pass or a shot going past the player, nobody holds it
                if ball_speeds[frame_num] > self.max_ball_speed_ratio * height:
                    best_player_id = -1

                # A guessed position or a ball at leg level (bounce pass, ball rolling past) is a weak clue:
                # it can extend the current candidate or holder (dribble) but not start a new possession
                elif best_player_id not in (candidate, current_holder):
                    is_low_ball = (ball_center[1] - y1) > self.low_ball_ratio * height
                    if ball_info.get('interpolated', False) or is_low_ball:
                        best_player_id = -1

            if best_player_id == -1:
                missed_frames += 1
                if missed_frames > self.max_missed_frames:
                    candidate = -1
                    candidate_count = 0
            elif best_player_id == candidate:
                candidate_count += 1
                missed_frames = 0
            else:
                # A player switch restarts the confirmation
                candidate = best_player_id
                candidate_start = frame_num
                candidate_count = 1
                missed_frames = 0

            if best_player_id != -1 and best_player_id == current_holder:
                holder_last_seen = frame_num

            if candidate_count == self.min_frames:
                current_holder = candidate
                holder_last_seen = frame_num
                # Backfill: the possession started at the first frame of the streak, not when it got confirmed
                for f in range(candidate_start, frame_num):
                    possession_list[f] = current_holder

            # Hysteresis: the holder keeps the ball until another player is confirmed or they have been gone too long
            if current_holder != -1 and frame_num - holder_last_seen > self.max_holder_absence:
                current_holder = -1

            possession_list[frame_num] = current_holder

        return possession_list
                
            

