from copy import deepcopy
import sys
import numpy as np
import cv2
from .homography import Homography
sys.path.append("../")
from utils import measure_distance, get_foot_position

class TacticalViewConverter:
    def __init__(self, court_image_path):
        self.court_image_path = court_image_path
        self.width = 400
        self.height = 235

        self.actual_width_in_meters = 40
        self.actual_height_in_meters = 20

        # images/court.jpg is 612 x 360 px with a border around the court: the goal lines are at x=51.5 and 559.5,
        # the sidelines at y=53.5 (far side) and 305.5 (camera side). Everything is scaled to width x height
        self.left_goal_line = self.width * 51.5 / 612
        self.far_sideline = self.height * 53.5 / 360
        self.pixels_per_meter_x = self.width * (559.5 - 51.5) / 612 / self.actual_width_in_meters
        self.pixels_per_meter_y = self.height * (305.5 - 53.5) / 360 / self.actual_height_in_meters
        left_goal_line, far_sideline = self.left_goal_line, self.far_sideline
        pixels_per_meter_x, pixels_per_meter_y = self.pixels_per_meter_x, self.pixels_per_meter_y

        # Keypoints in pixels of the tactical view, same names as input_videos/homography.py
        # so they can be matched with the keypoints clicked on the video
        self.key_points = {
            "coin gauche cote camera": (left_goal_line, far_sideline + 20 * pixels_per_meter_y),
            "coin gauche au fond": (left_goal_line, far_sideline),
            "pied du poteau cote camera": (left_goal_line, far_sideline + 11.5 * pixels_per_meter_y),
            "pied du poteau du fond": (left_goal_line, far_sideline + 8.5 * pixels_per_meter_y),
            "arc des 6 m sur la ligne de but, cote camera": (left_goal_line, far_sideline + 17.5 * pixels_per_meter_y),
            "arc des 6 m sur la ligne de but, au fond": (left_goal_line, far_sideline + 2.5 * pixels_per_meter_y),
            "ligne des 9 m sur la touche, cote camera": (left_goal_line + 2.96 * pixels_per_meter_x, far_sideline + 20 * pixels_per_meter_y),
            "ligne des 9 m sur la touche, au fond": (left_goal_line + 2.96 * pixels_per_meter_x, far_sideline),
            "ligne des 7 m, bout cote fond": (left_goal_line + 7 * pixels_per_meter_x, far_sideline + 9.5 * pixels_per_meter_y),
            "ligne des 7 m, bout cote camera": (left_goal_line + 7 * pixels_per_meter_x, far_sideline + 10.5 * pixels_per_meter_y),
        }

    def meters_to_tactical(self, position_m):
        x_meters, y_meters = position_m
        return (self.left_goal_line + x_meters * self.pixels_per_meter_x,
                self.far_sideline + y_meters * self.pixels_per_meter_y)

    def transform_players_to_tactical_view(self, player_tracks):
        # The camera is fixed: CourtTransformer already gives each player's position in meters ("position_m"),
        # so the tactical position is just meters scaled to the tactical view pixels
        tactical_players_positions = []
        for frame_tracks in player_tracks:
            tactical_positions = {}
            for player_id, player_data in frame_tracks.items():
                if "position_m" in player_data:
                    tactical_positions[player_id] = self.meters_to_tactical(player_data["position_m"])
            tactical_players_positions.append(tactical_positions)

        return tactical_players_positions

    # Only needed with a court keypoint detector (moving camera): cleans keypoints detected on each frame (sv.KeyPoints)
    def validate(self, keypoints_list):

        keypoints_list = deepcopy(keypoints_list)  
        for frame_idx, frame_keypoints in enumerate(keypoints_list):
            frame_keypoints = frame_keypoints.xy.tolist()[0]

            detected_indicies = [i for i, kp in enumerate(frame_keypoints) if kp[0]>0 and kp[1]>0]

            if len(detected_indicies) < 3:
                continue

            invalid_keypoints = []

            for i in detected_indicies:
                if frame_keypoints[i][0] == 0 and frame_keypoints[i][1] == 0:
                    continue

                other_indicies = [idx for idx in detected_indicies if idx != i and idx not in invalid_keypoints]
                
                if len(other_indicies) < 2:
                    continue

                j,k = other_indicies[0], other_indicies[1]

                d_ij = measure_distance(frame_keypoints[i], frame_keypoints[j])
                d_ik = measure_distance(frame_keypoints[i], frame_keypoints[k])
                
                t_ij = measure_distance(self.key_points[i], self.key_points[j])
                t_ik = measure_distance(self.key_points[i], self.key_points[k])

                if t_ij > 0 and t_ik > 0:
                    prop_detected = d_ij/d_ik if d_ik > 0 else float('inf')
                    prop_tactical = t_ij/t_ik if t_ik > 0 else float('inf')

                    error = (prop_detected - prop_tactical) / prop_tactical

                    if error > 0.8:
                        keypoints_list[frame_idx].xy[0][i] *= 0
                        keypoints_list[frame_idx].xy[0][j] *= 0
                        invalid_keypoints.append(i)
                
        return keypoints_list
