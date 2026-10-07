from transformers.models import axk1
from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel
import sys
sys.path.append("../")
from utils import read_stub, save_stub


class TeamAssigner:
    def __init__(self, team1_class_name ="sky blue shirt", team2_class_name ="green shirt", goalkeeper_team1_class_name ="red shirt", referee_class_name ="black shirt"):
        self.team1_class_name = team1_class_name
        self.team2_class_name = team2_class_name

        self.goalkeeper_team1_class_name = goalkeeper_team1_class_name
        self.referee_class_name = referee_class_name

        self.player_team_dict = {}
        
    def load_model(self):
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")   
    
    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        top = int(y1 + 0.15 * height)
        bottom = int(y1 + 0.5 * height)
        left = int(x1 + 0.15 * width)
        right = int(x2 - 0.15 * width)
        
        image = frame[top:bottom, left:right]
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        classes = [self.team1_class_name, self.team2_class_name, self.goalkeeper_team1_class_name, self.referee_class_name]

        inputs = self.processor(text=classes, images=pil_image, return_tensors="pt", padding=True)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        class_name = classes[probs.argmax(dim=1)[0]]
        return class_name

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        if player_color == self.team1_class_name:
            team_id = 1
        elif player_color == self.team2_class_name:
            team_id = 2
        elif player_color == self.goalkeeper_team1_class_name:
            team_id = 1
        elif player_color == self.referee_class_name:
            team_id = 0

        self.player_team_dict[player_id] = team_id
        return team_id 

    def get_player_teams_across_frames(self, video_frames, frame_count, player_tracks, read_from_stub=False, stub_path=None):

        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None:
            if len(player_assignment) == frame_count:
                return player_assignment

        self.load_model()

        player_assignment = []

        # video_frames is a generator: iterate it alongside the tracks instead of indexing it
        for frame_num, (frame, player_track) in enumerate(zip(video_frames, player_tracks)):
            player_assignment.append({})

            if frame_num % 50 == 0:
                self.player_team_dict = {}

            for player_id, track in player_track.items():
                team = self.get_player_team(frame, track["bbox"], player_id)
                player_assignment[frame_num][player_id] = team

        save_stub(stub_path, player_assignment)
    
        return player_assignment
