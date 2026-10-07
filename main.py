import teams_assigner
from utils import read_video, save_video, get_video_info
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer, TeamBallControlDrawer
from teams_assigner import TeamAssigner
from ball_acquisition.ball_acquisition_detector import BallAcquisitionDetector


def main():

	video_path = "input_videos/extrait.mp4"

	# Frames are streamed from the file (read_video is a generator), never all loaded in memory
	fps, frame_count = get_video_info(video_path)

	# Initialise tracker
	player_tracker = PlayerTracker(model_path="models/detector_1280.pt")
	ball_tracker = BallTracker(model_path="models/ball_detector_v2.pt")

	# Run tracker (each tracker reads the video again only if its stub is missing)
	player_tracks = player_tracker.get_object_tracks(read_video(video_path), frame_count, read_from_stub=True, stub_path="stubs/player_track_stubs.pkl")
	ball_tracks = ball_tracker.get_object_tracks(read_video(video_path), frame_count, read_from_stub=True, stub_path="stubs/ball_track_stubs.pkl")

	#remove wrong ball detection
	ball_tracks = ball_tracker.remove_wrong_detection(ball_tracks)

	#interpolate ball tracks
	ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

	# Assign player teams
	teams_assigner = TeamAssigner()
	player_assignment = teams_assigner.get_player_teams_across_frames(read_video(video_path), frame_count, player_tracks, read_from_stub=True, stub_path="stubs/player_teams_stubs.pkl")

	# Ball acquisition
	ball_acquisition_detector = BallAcquisitionDetector()
	ball_acquisition = ball_acquisition_detector.detect_ball_possession(player_tracks, ball_tracks)

	# Draw output video frames
	player_tracks_drawer = PlayerTracksDrawer()
	ball_tracks_drawer = BallTracksDrawer()
	team_ball_control_drawer = TeamBallControlDrawer()

	# draw player tracks
	output_video_frames = player_tracks_drawer.draw(read_video(video_path), player_tracks, player_assignment, ball_acquisition)
	
	# draw ball tracks
	output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)
	
	# draw team ball control
	output_video_frames = team_ball_control_drawer.draw(output_video_frames, player_assignment, ball_acquisition)


	# Save video (frames are read, drawn and written one at a time)
	save_video(output_video_frames=output_video_frames, output_video_path="output_videos/output_video.avi", fps=fps)


if __name__ == "__main__":
    main()
