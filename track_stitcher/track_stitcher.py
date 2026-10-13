import sys
sys.path.append("../")
from utils import measure_distance


class TrackStitcher:
    """Joins the pieces of track that ByteTrack splits when players cross or hide each other.

    When an id disappears and a new id appears a bit later where the player could have run to,
    both pieces are the same player: the new id is renamed to the old one.
    Works in court meters ("position_m"), which is reliable because the camera is fixed.
    """

    def __init__(self, fps, max_gap_seconds=1.5, max_player_speed=8, position_margin=1.0):
        # Longest hole (in frames) between the end of a piece and the start of the next one
        self.max_gap = int(max_gap_seconds * fps)
        self.fps = fps
        # A handball player sprints at 7-8 m/s: beyond that, the new piece can't be the same player
        self.max_player_speed = max_player_speed
        # Extra meters allowed for the position noise (box swallowed by a teammate, feet hidden)
        self.position_margin = position_margin

    def get_tracklets(self, player_tracks, player_assignment):
        tracklets = {}
        for frame_num, frame_tracks in enumerate(player_tracks):
            for track_id, track in frame_tracks.items():
                if "position_m" not in track:
                    continue
                tracklet = tracklets.setdefault(track_id, {"start": frame_num, "start_position": track["position_m"], "team": -1})
                tracklet["end"] = frame_num
                tracklet["end_position"] = track["position_m"]
                if tracklet["team"] == -1:
                    tracklet["team"] = player_assignment[frame_num].get(track_id, -1)
        return tracklets

    def find_links(self, tracklets):
        candidate_links = []
        for previous_id, previous in tracklets.items():
            for next_id, following in tracklets.items():
                gap = following["start"] - previous["end"]
                if gap <= 0 or gap > self.max_gap:
                    continue
                if previous["team"] == -1 or previous["team"] != following["team"]:
                    continue

                distance = measure_distance(previous["end_position"], following["start_position"])
                if distance <= self.max_player_speed * gap / self.fps + self.position_margin:
                    candidate_links.append((distance, gap, previous_id, next_id))

        # Closest pairs first; each piece can only be continued once and continue one piece
        links = {}
        linked_next = set()
        for _, _, previous_id, next_id in sorted(candidate_links):
            if previous_id in links or next_id in linked_next:
                continue
            links[previous_id] = next_id
            linked_next.add(next_id)
        return links

    def stitch(self, player_tracks, player_assignment):
        tracklets = self.get_tracklets(player_tracks, player_assignment)
        links = self.find_links(tracklets)

        # Follow each chain from its first piece and give every piece the id of the first one
        new_ids = {}
        for first_id in tracklets:
            if first_id in links.values():
                continue
            track_id = first_id
            while track_id in links:
                track_id = links[track_id]
                new_ids[track_id] = first_id

        stitched_tracks = [{new_ids.get(track_id, track_id): track for track_id, track in frame_tracks.items()}
                           for frame_tracks in player_tracks]
        stitched_assignment = [{new_ids.get(track_id, track_id): team for track_id, team in frame_assignment.items()}
                               for frame_assignment in player_assignment]
        return stitched_tracks, stitched_assignment
