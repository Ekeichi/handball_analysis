import cv2
import os
from itertools import islice

def get_video_info(video_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # CAP_PROP_FRAME_COUNT is unreliable (metadata), so count the frames actually readable
    frame_count = 0
    while cap.grab():
        frame_count += 1
    cap.release()
    return fps, frame_count

def read_video(video_path: str):
    # Yield frames one by one instead of loading the whole video in memory
    cap = cv2.VideoCapture(video_path)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()

def batch_frames(frames, batch_size):
    frames = iter(frames)
    while True:
        batch = list(islice(frames, batch_size))
        if not batch:
            break
        yield batch

def save_video(output_video_frames, output_video_path: str, fps: int):
    if not os.path.exists(os.path.dirname(output_video_path)):
        os.mkdir(os.path.dirname(output_video_path))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = None

    for frame in output_video_frames:
        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
        out.write(frame)

    if out is not None:
        out.release()
