from drawers import PassAndInterceptionsDrawer
import json
from court_keypoint_detector import CourtTransformer
import teams_assigner
from utils import read_video, save_video, get_video_info
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer, TeamBallControlDrawer, CourtKeypointDrawer, TacticalViewDrawer
from teams_assigner import TeamAssigner
from ball_acquisition.ball_acquisition_detector import BallAcquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from tactical_view_converter import TacticalViewConverter
from track_stitcher import TrackStitcher

def main():

	video_path = "input_videos/extrait.mp4"

	# Frames are streamed from the file (read_video is a generator), never all loaded in memory
	fps, frame_count = get_video_info(video_path)

	# Initialise tracker
	player_tracker = PlayerTracker(model_path="models/detector_1280.pt", fps=fps)
	ball_tracker = BallTracker(model_path="models/ball_best.pt")
	court_transformer = CourtTransformer("input_videos/extrait_keypoints.json")

	# Run tracker (each tracker reads the video again only if its stub is missing)
	player_tracks = player_tracker.get_object_tracks(read_video(video_path), frame_count, read_from_stub=True, stub_path="stubs/player_track_stubs.pkl")
	ball_tracks = ball_tracker.get_object_tracks(read_video(video_path), frame_count, read_from_stub=True, stub_path="stubs/ball_track_stubs.pkl")

	#remove wrong ball detection
	ball_tracks = ball_tracker.remove_wrong_detection(ball_tracks)

	#interpolate ball tracks
	ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

	# filter player tracks outside the court
	player_tracks = court_transformer.filter_player_tracks(player_tracks, image_height=1080)

	# Assign player teams
	teams_assigner = TeamAssigner()
	player_assignment = teams_assigner.get_player_teams_across_frames(read_video(video_path), frame_count, player_tracks, read_from_stub=True, stub_path="stubs/player_teams_stubs.pkl")

	# Join the track pieces split by occlusions (after the team assignment: its stub is keyed by the original ids)
	track_stitcher = TrackStitcher(fps=fps)
	player_tracks, player_assignment = track_stitcher.stitch(player_tracks, player_assignment)

	# Ball acquisition
	ball_acquisition_detector = BallAcquisitionDetector()
	ball_acquisition = ball_acquisition_detector.detect_ball_possession(player_tracks, ball_tracks)

	# Detect passes and interceptions
	pass_and_interception_detector = PassAndInterceptionDetector()
	passes = pass_and_interception_detector.detect_passes(ball_acquisition, player_assignment)
	interceptions = pass_and_interception_detector.detect_interceptions(ball_acquisition, player_assignment)
	tactical_view_converter = TacticalViewConverter("images/court.jpg")
	
	# load court keypoints
	with open("input_videos/extrait_keypoints.json") as f:
		court_keypoints = json.load(f)["keypoints"]

	# Player positions in the tactical view (from the "position_m" computed by court_transformer)
	tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(player_tracks)
	
	# Draw output video frames
	player_tracks_drawer = PlayerTracksDrawer()
	ball_tracks_drawer = BallTracksDrawer()
	team_ball_control_drawer = TeamBallControlDrawer()
	passes_and_interceptions_drawer = PassAndInterceptionsDrawer()
	court_keypoint_drawer = CourtKeypointDrawer()
	tactical_view_drawer = TacticalViewDrawer()

	# draw player tracks
	output_video_frames = player_tracks_drawer.draw(read_video(video_path), player_tracks, player_assignment, ball_acquisition)
	
	# draw ball tracks
	output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)
	
	# draw team ball control
	output_video_frames = team_ball_control_drawer.draw(output_video_frames, player_assignment, ball_acquisition)

	# draw passes and interceptions numbers
	output_video_frames = passes_and_interceptions_drawer.draw(output_video_frames, passes, interceptions)

	# draw court keypoints
	output_video_frames = court_keypoint_drawer.draw(output_video_frames, court_keypoints)
	
	# draw tactical view
	output_video_frames = tactical_view_drawer.draw(output_video_frames, tactical_view_converter.court_image_path, tactical_view_converter.width, tactical_view_converter.height, tactical_view_converter.key_points, tactical_player_positions, player_assignment, ball_acquisition)
	
	# Save video (frames are read, drawn and written one at a time)
	save_video(output_video_frames=output_video_frames, output_video_path="output_videos/output_video.avi", fps=fps)


if __name__ == "__main__":
    main()
