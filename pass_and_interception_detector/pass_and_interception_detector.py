class PassAndInterceptionDetector:
    def __init__(self):
        pass

    def get_player_team(self, player_assignment, player_id, frame):
        # The holder can be kept on frames where the tracker lost them (ball in flight, occlusion),
        # so use the team from the closest frame where the player was assigned one
        for offset in range(len(player_assignment)):
            for f in (frame - offset, frame + offset):
                if 0 <= f < len(player_assignment):
                    team = player_assignment[f].get(player_id, -1)
                    if team != -1:
                        return team
        return -1

    def detect_passes(self, ball_acquisition, player_assignment):
        passes = [-1] * len(ball_acquisition)
        prev_holder = -1
        prev_frame = -1

        for frame in range(1, len(ball_acquisition)):
            if ball_acquisition[frame - 1] != -1:
                prev_holder = ball_acquisition[frame - 1]
                prev_frame = frame - 1
            
            current_holder = ball_acquisition[frame]

            if prev_holder != -1 and current_holder != -1 and prev_holder != current_holder:
                prev_team = self.get_player_team(player_assignment, prev_holder, prev_frame)
                current_team = self.get_player_team(player_assignment, current_holder, frame)

                if prev_team == current_team and prev_team != -1:
                    passes[frame] = prev_team

        return passes
            
    def detect_interceptions(self, ball_acquisition, player_assignment):
        interceptions = [-1] * len(ball_acquisition)
        prev_holder = -1
        prev_frame = -1

        for frame in range(1, len(ball_acquisition)):
            if ball_acquisition[frame - 1] != -1:
                prev_holder = ball_acquisition[frame - 1]
                prev_frame = frame - 1
            
            current_holder = ball_acquisition[frame]

            if prev_holder != -1 and current_holder != -1 and prev_holder != current_holder:
                prev_team = self.get_player_team(player_assignment, prev_holder, prev_frame)
                current_team = self.get_player_team(player_assignment, current_holder, frame)

                if prev_team != current_team and prev_team != -1 and current_team != -1:
                    interceptions[frame] = current_team

        return interceptions

         