from .utils import draw_ellipse, draw_triangle

class PlayerTracksDrawer:
    
    def __init__(self, team_1_color=(235, 206, 135), team_2_color=(0, 180, 0), referee_color=(255,255,255)):
        self.default_player_team_id = 1
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color

        self.referee_color = referee_color

    def draw(self, video_frames, tracks, player_assignment, ball_acquisition):
        # Draw in place and yield frame by frame to avoid copying the whole video
        for frame_num, frame in enumerate(video_frames):
            player_dict = tracks[frame_num]

            player_assignment_for_frame = player_assignment[frame_num]

            player_id_has_ball = ball_acquisition[frame_num]

            # draw players tracks
            for track_id, player in player_dict.items():
                if player_assignment_for_frame.get(track_id, self.default_player_team_id) == 1:
                    frame = draw_ellipse(frame, player['bbox'], self.team_1_color, track_id)
                elif player_assignment_for_frame.get(track_id, self.default_player_team_id) == 2:
                    frame = draw_ellipse(frame, player['bbox'], self.team_2_color, track_id)
                elif player_assignment_for_frame.get(track_id, self.default_player_team_id) == 0:
                    frame = draw_ellipse(frame, player['bbox'], self.referee_color, track_id)

                # Highlight the player with the ball
                if track_id == player_id_has_ball:
                    frame = draw_triangle(frame, player['bbox'], (0,0,255))

            yield frame
