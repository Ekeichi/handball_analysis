from collections import Counter
from PIL import Image
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
import sys
sys.path.append("../")
from utils import read_stub, save_stub


class TeamAssigner:
    def __init__(self, team1_class_name ="sky blue shirt", team2_class_name ="green shirt", goalkeeper_team1_class_name ="red shirt", referee_class_name ="black shirt", sample_every=10):
        self.team1_class_name = team1_class_name
        self.team2_class_name = team2_class_name

        self.goalkeeper_team1_class_name = goalkeeper_team1_class_name
        self.referee_class_name = referee_class_name

        # CLIP looks at each player every sample_every frames (1/6 s at 60 fps): a jersey barely changes in between
        self.sample_every = sample_every

        # Votes collected for each track id: {player_id: [team_id, ...]}
        self.player_votes = {}

    def load_model(self):
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    def get_jersey_crop(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # Keep only the jersey: skip the head and everything below the waist, trim the sides
        top = int(y1 + 0.15 * height)
        bottom = int(y1 + 0.5 * height)
        left = int(x1 + 0.15 * width)
        right = int(x2 - 0.15 * width)

        image = frame[max(top, 0):bottom, max(left, 0):right]
        if image.size == 0:
            # Box too small for the jersey crop: fall back to the whole box
            image = frame[max(int(y1), 0):int(y2), max(int(x1), 0):int(x2)]
        if image.size == 0:
            return None

        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def get_players_colors(self, frame, bboxes):
        """Asks CLIP for the jersey of every given player of the frame in a single call."""
        classes = [self.team1_class_name, self.team2_class_name, self.goalkeeper_team1_class_name, self.referee_class_name]

        crops = [self.get_jersey_crop(frame, bbox) for bbox in bboxes]
        valid = [i for i, crop in enumerate(crops) if crop is not None]
        colors = [None] * len(bboxes)
        if not valid:
            return colors

        inputs = self.processor(text=classes, images=[crops[i] for i in valid], return_tensors="pt", padding=True)
        with torch.no_grad():
            probs = self.model(**inputs).logits_per_image.softmax(dim=1)

        for i, class_index in zip(valid, probs.argmax(dim=1).tolist()):
            colors[i] = classes[class_index]
        return colors

    def color_to_team(self, player_color):
        if player_color in (self.team1_class_name, self.goalkeeper_team1_class_name):
            return 1
        if player_color == self.team2_class_name:
            return 2
        return 0  # referee

    def get_player_teams_across_frames(self, video_frames, frame_count, player_tracks, read_from_stub=False, stub_path=None):

        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None:
            if len(player_assignment) == frame_count:
                return player_assignment

        self.load_model()

        # 1. Collect votes: CLIP looks at each player on several frames of its track
        self.player_votes = {}
        last_vote_frame = {}

        # video_frames is a generator: iterate it alongside the tracks instead of indexing it
        for frame_num, (frame, player_track) in enumerate(zip(video_frames, player_tracks)):
            players_to_look_at = [player_id for player_id in player_track
                                  if frame_num - last_vote_frame.get(player_id, -self.sample_every) >= self.sample_every]
            if not players_to_look_at:
                continue

            colors = self.get_players_colors(frame, [player_track[player_id]["bbox"] for player_id in players_to_look_at])
            for player_id, color in zip(players_to_look_at, colors):
                if color is None:
                    continue
                self.player_votes.setdefault(player_id, []).append(self.color_to_team(color))
                last_vote_frame[player_id] = frame_num

        # 2. Each track id gets the team most voted over its whole life, so a bad frame
        # (occlusion, player bent over, teammate in front) can't flip its color
        player_teams = {player_id: Counter(votes).most_common(1)[0][0] for player_id, votes in self.player_votes.items()}

        player_assignment = []
        for player_track in player_tracks:
            player_assignment.append({player_id: player_teams[player_id] for player_id in player_track if player_id in player_teams})

        save_stub(stub_path, player_assignment)

        return player_assignment
