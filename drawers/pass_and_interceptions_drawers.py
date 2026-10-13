
import cv2

class PassAndInterceptionsDrawer:
    def __init__(self):
        pass

    def get_stats(self, passes, interceptions):
        team_1_passes = []
        team_2_passes = []
        team_1_interceptions = []
        team_2_interceptions = []

        for frame_num in range(len(passes)):
            if passes[frame_num] == 1:
                team_1_passes.append(frame_num)
            elif passes[frame_num] == 2:
                team_2_passes.append(frame_num)

            if interceptions[frame_num] == 1:
                team_1_interceptions.append(frame_num)
            elif interceptions[frame_num] == 2:
                team_2_interceptions.append(frame_num)

        return len(team_1_passes), len(team_2_passes), len(team_1_interceptions), len(team_2_interceptions)

    def draw(self, video_frames, passes, interceptions):
        # Draw in place and yield frame by frame to avoid keeping the whole video in memory
        for frame_num, frame in enumerate(video_frames):
            yield self.draw_frame(frame, frame_num, passes, interceptions)
    
    def draw_frame(self, frame, frame_num, passes, interceptions):
        overlay = frame.copy()
        font_scale = 0.7
        font_thickness = 2
        
        #overlay position
        frame_height, frame_width, _ = overlay.shape
        rect_x1 = int(frame_width * 0.16)
        rect_y1 = int(frame_height * 0.75)
        rect_x2 = int(frame_width * 0.55)
        rect_y2 = int(frame_height * 0.90)

        #text position
        text_x = int(frame_width * 0.19)
        text_y1 = int(frame_height * 0.80)
        text_y2 = int(frame_height * 0.88)

        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        
        passes_till_frame = passes[:frame_num+1]
        interceptions_till_frame = interceptions[:frame_num+1]

        team_1_passes, team_2_passes, team_1_interceptions, team_2_interceptions = self.get_stats(passes_till_frame, interceptions_till_frame)

        cv2.putText(frame, f"Team 1 passes: {team_1_passes} / interceptions: {team_1_interceptions}", (text_x, text_y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
        cv2.putText(frame, f"Team 2 passes: {team_2_passes} / interceptions: {team_2_interceptions}", (text_x, text_y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
        
        return frame