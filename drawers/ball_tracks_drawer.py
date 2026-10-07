from .utils import draw_triangle

class BallTracksDrawer:
    def __init__(self):
        self.ball_pointer_color = (0,255,0)

    def draw(self, video_frames, tracks):
        # Draw in place and yield frame by frame to avoid copying the whole video
        for frame_num, frame in enumerate(video_frames):
            ball_dict = tracks[frame_num]

            for _, track in ball_dict.items():
                bbox = track['bbox']
                if bbox is None:
                    continue

                frame = draw_triangle(frame, bbox, self.ball_pointer_color)

            yield frame
