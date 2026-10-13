import cv2

class TacticalViewDrawer:
    def __init__(self, team_1_color=(235, 206, 135), team_2_color=(0, 180, 0), referee_color=(255, 255, 255)):
        self.start_x = 20
        self.start_y = 40
        
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color
        self.referee_color = referee_color

    def draw(self, video_frames, court_image_path, width, height, tactical_court_keypoints, tactical_player_positions, player_assignment=None, ball_aquisition_point=None):
        court_image = cv2.imread(court_image_path)
        if court_image is None:
            raise FileNotFoundError(f"Court image not found: {court_image_path}")
        court_image = cv2.resize(court_image, (width, height))
        
        # Draw in place and yield frame by frame to avoid keeping the whole video in memory
        for frame_idx, frame in enumerate(video_frames):
            y1 = self.start_y
            y2 = y1 + height
            x1 = self.start_x
            x2 = x1 + width

            alpha = 0.6
            overlay = frame[y1:y2, x1:x2].copy()
            cv2.addWeighted(court_image, alpha, overlay, 1 - alpha, 0, frame[y1:y2, x1:x2])

            for x, y in tactical_court_keypoints.values():
                cv2.circle(frame, (int(x + self.start_x), int(y + self.start_y)), 3, (0, 0, 255), -1)

            frame_positions = tactical_player_positions[frame_idx]
            frame_assignment = player_assignment[frame_idx] if player_assignment is not None else {}
            player_with_ball = ball_aquisition_point[frame_idx] if ball_aquisition_point is not None else -1

            for player_id, player_position in frame_positions.items():
                team_id = frame_assignment.get(player_id, 1)
                if team_id == 1:
                    color = self.team_1_color
                elif team_id == 2:
                    color = self.team_2_color
                else:
                    color = self.referee_color

                x, y = int(player_position[0] + self.start_x), int(player_position[1] + self.start_y)

                player_radius = 6
                cv2.circle(frame, (x, y), radius=player_radius, color=color, thickness=-1)

                if player_id == player_with_ball:
                    cv2.circle(frame, (x, y), radius=player_radius + 3, color=(0, 0, 255), thickness=2)

            yield frame
