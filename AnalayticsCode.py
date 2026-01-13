import cv2
import numpy as np
import pandas as pd
import pickle
import os
import sys
import torch
from ultralytics import YOLO
import supervision as sv
from sklearn.cluster import KMeans

# ==============================================================================
# SECTION 1: UTILITY FUNCTIONS
# ==============================================================================

def check_gpu():
    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        return 0
    else:
        print("⚠️ GPU NOT detected. Running on CPU.")
        return 'cpu'

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    if not output_video_frames: return
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 24,
                          (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

def measure_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def measure_xy_distance(p1, p2):
    return p1[0] - p2[0], p1[1] - p2[1]

def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)

# ==============================================================================
# SECTION 2: TRACKER CLASS (Dual Model + Resolution Fix)
# ==============================================================================

class Tracker:
    def __init__(self, model_path_players, model_path_ball, device):
        print(f"Loading Player Model: {model_path_players}...")
        self.model_players = YOLO(model_path_players)

        print(f"Loading Ball Model: {model_path_ball}...")
        self.model_ball = YOLO(model_path_ball)

        self.tracker = sv.ByteTrack()
        self.device = device

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_boxes = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_boxes, columns=['x1', 'y1', 'x2', 'y2'])

        # Filter Outliers
        df_ball_positions['center_x'] = (df_ball_positions['x1'] + df_ball_positions['x2']) / 2
        df_ball_positions['center_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2

        df_ball_positions['dy'] = df_ball_positions['center_y'].diff()
        df_ball_positions['dx'] = df_ball_positions['center_x'].diff()
        df_ball_positions['distance'] = (df_ball_positions['dx']**2 + df_ball_positions['dy']**2)**0.5

        MAX_PIXEL_JUMP = 20
        df_ball_positions.loc[df_ball_positions['distance'] > MAX_PIXEL_JUMP, ['x1', 'y1', 'x2', 'y2']] = np.nan

        # Interpolate
        df_ball_positions = df_ball_positions.interpolate(limit=15)
        df_ball_positions = df_ball_positions.bfill(limit=5)

        ball_positions = [{1: {"bbox": x}} if not np.isnan(x[0]) else {} for x in df_ball_positions[['x1', 'y1', 'x2', 'y2']].to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames, model, img_size=640, conf_thresh=0.1):
        """
        Generic detection function that accepts custom resolution and confidence.
        """
        batch_size = 16
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = model.predict(
                frames[i:i+batch_size],
                conf=conf_thresh,
                device=self.device,
                imgsz=img_size, # Use high res for ball
                verbose=False
            )
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        tracks = {"players": [], "referees": [], "ball": []}

        # --- STEP 1: DETECT PLAYERS (Standard Res, Normal Confidence) ---
        print(f"Step 1/2: Detecting PLAYERS (YOLOv5)...")
        player_detections = self.detect_frames(frames, self.model_players, img_size=640, conf_thresh=0.1)

        # --- STEP 2: DETECT BALL (High Res, Low Confidence) ---
        print(f"Step 2/2: Detecting BALL (YOLOv12x - High Res)...")
        # img_size=1280 is CRITICAL for seeing the ball
        # conf_thresh=0.05 allows catching blurry balls
        ball_detections_raw = self.detect_frames(frames, self.model_ball, img_size=1280, conf_thresh=0.05)

        print("Processing Tracks...")

        for frame_num in range(len(frames)):
            # --- PROCESS PLAYERS ---
            det_p = player_detections[frame_num]
            cls_names_p = det_p.names
            cls_names_inv_p = {v: k for k, v in cls_names_p.items()}
            sv_det_p = sv.Detections.from_ultralytics(det_p)

            # Filter out ball from player model if present
            non_ball_mask = np.array([cls_id != cls_names_inv_p.get('ball', 0) for cls_id in sv_det_p.class_id])

            if len(non_ball_mask) > 0 and np.any(non_ball_mask):
                valid_detections = sv_det_p[non_ball_mask]
                detection_with_tracks = self.tracker.update_with_detections(valid_detections)
            else:
                detection_with_tracks = []

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv_p.get('player', 2):
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv_p.get('goalkeeper', 1):
                    tracks["players"][frame_num][track_id] = {"bbox": bbox, "role": "GK"}
                elif cls_id == cls_names_inv_p.get('referee', 3):
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # --- PROCESS BALL (Fixed for ID 1 and 2) ---
            det_b = ball_detections_raw[frame_num]
            sv_det_b = sv.Detections.from_ultralytics(det_b)

            # Look for BOTH 'ball' (id 1) and 'balls' (id 2) based on your logs
            ball_mask = np.array([cls_id in [1, 2] for cls_id in sv_det_b.class_id])

            if np.any(ball_mask):
                ball_hits = sv_det_b[ball_mask]
                if len(ball_hits) > 0:
                    # Pick highest confidence
                    best_ball_idx = np.argmax(ball_hits.confidence)
                    ball_bbox = ball_hits.xyxy[best_ball_idx].tolist()
                    tracks["ball"][frame_num][1] = {"bbox": ball_bbox}

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_bbox_with_label(self, frame, bbox, color, label, distance=None, is_ball=False):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if not is_ball:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            (w_lbl, h_lbl), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(frame, (x1, y1 - h_lbl - 5), (x1 + w_lbl + 5, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5), font, font_scale, (0,0,0), thickness)

            if distance is not None:
                dist_txt = f"{distance:.1f}m"
                (w_dist, h_dist), _ = cv2.getTextSize(dist_txt, font, font_scale, thickness)
                dist_y2 = y1 - h_lbl - 5
                dist_y1 = dist_y2 - h_dist - 5
                cv2.rectangle(frame, (x1, dist_y1), (x1 + w_dist + 5, dist_y2), (255, 255, 255), -1)
                cv2.putText(frame, dist_txt, (x1 + 2, dist_y2 - 3), font, font_scale, (0,0,0), thickness)
        return frame

    def draw_clean_stats(self, frame, team_ball_control, pass_counts, distances):
        t1_frames = (team_ball_control == 1).sum()
        t2_frames = (team_ball_control == 2).sum()
        total = t1_frames + t2_frames

        t1_poss = (t1_frames / total) * 100 if total > 0 else 0
        t2_poss = (t2_frames / total) * 100 if total > 0 else 0

        t1_dist = distances.get(1, 0) / 1000
        t2_dist = distances.get(2, 0) / 1000

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        color = (0, 0, 0)
        thickness = 2
        h, w, _ = frame.shape
        y = h - 100

        cv2.putText(frame, f"Possession: T1 {t1_poss:.0f}% | T2 {t2_poss:.0f}%", (20, y), font, scale, color, thickness)
        y += 35
        cv2.putText(frame, f"Passes: T1 {pass_counts.get(1,0)} | T2 {pass_counts.get(2,0)}", (20, y), font, scale, color, thickness)
        y += 35
        cv2.putText(frame, f"Distance: T1 {t1_dist:.1f}km | T2 {t2_dist:.1f}km", (20, y), font, scale, color, thickness)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control, pass_counts_history, distances_history):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            for track_id, player in player_dict.items():
                color = player.get("team_color", (200, 200, 200))
                label = f"P{track_id}"
                distance = player.get("distance", 0)

                if player.get('role') == 'GK':
                    label = "GK"
                    color = (0, 255, 255)

                frame = self.draw_bbox_with_label(frame, player["bbox"], color, label, distance=distance)

                if player.get('has_ball', False):
                     x1, y1, x2, y2 = player["bbox"]
                     cx = int((x1+x2)/2)
                     cv2.arrowedLine(frame, (cx, int(y1)-45), (cx, int(y1)-20), (0,0,255), 3, tipLength=0.5)

            for _, referee in referee_dict.items():
                frame = self.draw_bbox_with_label(frame, referee["bbox"], (0,0,0), "REF")
                cv2.putText(frame, "REF", (int(referee['bbox'][0]), int(referee['bbox'][1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

            for track_id, ball in ball_dict.items():
                frame = self.draw_bbox_with_label(frame, ball["bbox"], (0, 255, 0), "Ball", is_ball=True)

            current_passes = pass_counts_history[frame_num]
            current_dists = distances_history[frame_num]
            current_possession = team_ball_control[:frame_num+1]
            frame = self.draw_clean_stats(frame, current_possession, current_passes, current_dists)

            output_video_frames.append(frame)
        return output_video_frames

# ==============================================================================
# SECTION 3: OTHER MODULES (Team, Camera, View, Speed)
# ==============================================================================

class TeamAssigner:
    def __init__(self):
        self.team_centers = None
        self.player_team_dict = {}
        self.kmeans = None

    def get_jersey_color_and_feature(self, frame, box_xyxy):
        x1, y1, x2, y2 = map(int, box_xyxy)
        h, w = y2 - y1, x2 - x1
        if h < 5 or w < 5: return (255, 255, 255), np.zeros(3)

        crop_y1 = y1 + int(h * 0.1)
        crop_y2 = y1 + int(h * 0.5)
        crop_x1 = x1 + int(w * 0.3)
        crop_x2 = x1 + int(w * 0.7)
        crop = frame[max(0,crop_y1):min(frame.shape[0],crop_y2), max(0,crop_x1):min(frame.shape[1],crop_x2)]

        if crop.size == 0: return (255, 255, 255), np.zeros(3)
        mean_bgr = np.mean(crop, axis=(0, 1))
        bgr_uint8 = np.uint8([[mean_bgr]])
        lab = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2LAB)[0, 0]
        feature_vec = lab.astype(np.float32)
        return tuple(map(int, mean_bgr)), feature_vec

    def init_team_centers(self, features):
        if len(features) < 2: return
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=5)
        kmeans.fit(features)
        self.team_centers = kmeans.cluster_centers_.astype(np.float32)
        print("✅ Team colors learned.")

    def assign_team(self, player_id, feature_vec):
        if player_id in self.player_team_dict: return self.player_team_dict[player_id]
        if self.team_centers is None: return 1
        dists = np.linalg.norm(self.team_centers - feature_vec[None, :], axis=1)
        team_idx = int(np.argmin(dists))
        self.player_team_dict[player_id] = team_idx + 1
        return team_idx + 1

class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 70
    def assign_ball_to_player(self, players, ball_bbox):
        if not ball_bbox: return -1
        ball_position = get_center_of_bbox(ball_bbox)
        min_dist = 99999
        assigned_player = -1
        for player_id, player in players.items():
            player_bbox = player['bbox']
            d_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            d_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(d_left, d_right)
            if distance < self.max_player_ball_distance:
                if distance < min_dist:
                    min_dist = distance
                    assigned_player = player_id
        return assigned_player

class CameraMovementEstimator:
    def __init__(self, frame):
        self.minimum_distance = 5
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        first_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(first_gray)
        mask[:, 0:20] = 1; mask[:, 900:1050] = 1
        self.features = dict(maxCorners=100, qualityLevel=0.3, minDistance=3, blockSize=7, mask=mask)

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f: return pickle.load(f)
        camera_movement = [[0, 0]] * len(frames)
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            if old_features is None or len(old_features) == 0:
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                old_gray = frame_gray.copy(); continue

            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)
            max_dist = 0; cam_x, cam_y = 0, 0
            if new_features is not None and len(new_features) > 0:
                for i, (new, old) in enumerate(zip(new_features, old_features)):
                    new_pt = new.ravel(); old_pt = old.ravel()
                    dist = measure_distance(new_pt, old_pt)
                    if dist > max_dist:
                        max_dist = dist; cam_x, cam_y = measure_xy_distance(old_pt, new_pt)
            if max_dist > self.minimum_distance:
                camera_movement[frame_num] = [cam_x, cam_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            if old_features is None or len(old_features) < 10:
                 old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            old_gray = frame_gray.copy()
        if stub_path:
            with open(stub_path, 'wb') as f: pickle.dump(camera_movement, f)
        return camera_movement

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        return frames # Skip visual

class ViewTransformer:
    def __init__(self):
        court_width, court_length = 68, 23.32
        self.pixel_vertices = np.array([[110, 1035], [265, 275], [910, 260], [1640, 915]]).astype(np.float32)
        self.target_vertices = np.array([[0, court_width], [0, 0], [court_length, 0], [court_length, court_width]]).astype(np.float32)
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)
    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        if cv2.pointPolygonTest(self.pixel_vertices, p, False) < 0: return None
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        return transform_point.reshape(-1, 2)
    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = np.array(track_info['position_adjusted'])
                    pos_trans = self.transform_point(position)
                    if pos_trans is not None: pos_trans = pos_trans.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = pos_trans

class SpeedAndDistance_Estimator:
    def __init__(self): self.frame_window = 5
    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}
        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees": continue
            num_frames = len(object_tracks)
            for frame_num in range(0, num_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, num_frames - 1)
                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]: continue
                    start = object_tracks[frame_num][track_id]['position_transformed']
                    end = object_tracks[last_frame][track_id]['position_transformed']
                    if start is None or end is None: continue
                    dist = measure_distance(start, end)
                    if object not in total_distance: total_distance[object] = {}
                    if track_id not in total_distance[object]: total_distance[object][track_id] = 0
                    total_distance[object][track_id] += dist
                    for f in range(frame_num, last_frame):
                        if track_id in tracks[object][f]:
                            tracks[object][f][track_id]['distance'] = total_distance[object][track_id]
        return total_distance

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def generate_summary(tracks, team_ball_control, team_pass_counts, player_pass_counts, output_file='output_stats.txt'):
    t1_frames = (team_ball_control == 1).sum(); t2_frames = (team_ball_control == 2).sum(); total = t1_frames + t2_frames
    t1_poss = (t1_frames/total)*100 if total>0 else 0; t2_poss = (t2_frames/total)*100 if total>0 else 0
    player_distances = {}
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, info in player_track.items():
            team = info.get('team', 0); dist = info.get('distance', 0)
            if team not in player_distances: player_distances[team] = {}
            if dist > player_distances[team].get(player_id, 0): player_distances[team][player_id] = dist
    t1_total_dist = sum(player_distances.get(1, {}).values()); t2_total_dist = sum(player_distances.get(2, {}).values())

    text = f"T1 Poss: {t1_poss:.1f}% | T2 Poss: {t2_poss:.1f}%\n"
    text += f"T1 Dist: {t1_total_dist/1000:.2f}km | T2 Dist: {t2_total_dist/1000:.2f}km"
    print(text)
    with open(output_file, "w") as f: f.write(text)

def main():
    device = check_gpu()
    # PATHS
    video_path = 'input_videos/test9.mp4'
    model_path_players = 'models/best.pt'
    model_path_ball = 'models/best3.pt'
    output_path = 'output_videos/final_output_dual_fixed.mp4'

    print("Reading video...")
    video_frames = read_video(video_path)

    print("Tracking (Dual Model)...")
    tracker = Tracker(model_path_players, model_path_ball, device=device)
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path='stubs/track_stubs_dual.pkl')
    tracker.add_position_to_tracks(tracks)

    print("Camera Movement...")
    camera_estimator = CameraMovementEstimator(video_frames[0])
    cam_movement = camera_estimator.get_camera_movement(video_frames, read_from_stub=False, stub_path='stubs/cam_stub.pkl')
    camera_estimator.add_adjust_positions_to_tracks(tracks, cam_movement)

    print("View Transform...")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    print("Ball Interpolation...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    print("Speed & Distance...")
    speed_estimator = SpeedAndDistance_Estimator()
    speed_estimator.add_speed_and_distance_to_tracks(tracks)

    print("Team Assignment...")
    team_assigner = TeamAssigner()
    all_features = []
    for frame_num in range(min(30, len(video_frames))):
        frame = video_frames[frame_num]
        for _, track in tracks['players'][frame_num].items():
             _, feat = team_assigner.get_jersey_color_and_feature(frame, track['bbox'])
             all_features.append(feat)
    if len(all_features) > 2: team_assigner.init_team_centers(np.array(all_features))

    for frame_num, player_track in enumerate(tracks['players']):
        frame = video_frames[frame_num]
        for player_id, track in player_track.items():
            display_color, feat = team_assigner.get_jersey_color_and_feature(frame, track['bbox'])
            team_id = team_assigner.assign_team(player_id, feat)
            tracks['players'][frame_num][player_id]['team'] = team_id
            tracks['players'][frame_num][player_id]['team_color'] = display_color

    print("Possession Logic...")
    player_assigner = PlayerBallAssigner()
    team_ball_control = []; team_pass_counts = {1:0, 2:0}; player_pass_counts = {}; pass_counts_history = []; distances_history = []
    last_player = -1; last_team = -1

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox')
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            curr_team = tracks['players'][frame_num][assigned_player]['team']
            team_ball_control.append(curr_team)
            if last_team == curr_team and last_player != assigned_player and last_player != -1:
                team_pass_counts[curr_team] += 1
                player_pass_counts[last_player] = player_pass_counts.get(last_player, 0) + 1
            last_player = assigned_player; last_team = curr_team
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

        pass_counts_history.append(team_pass_counts.copy())
        d1 = sum([p.get('distance', 0) for p in player_track.values() if p.get('team') == 1])
        d2 = sum([p.get('distance', 0) for p in player_track.values() if p.get('team') == 2])
        distances_history.append({1: d1, 2: d2})

    print("Drawing...")
    output_video_frames = tracker.draw_annotations(video_frames, tracks, np.array(team_ball_control), pass_counts_history, distances_history)
    save_video(output_video_frames, output_path)
    generate_summary(tracks, np.array(team_ball_control), team_pass_counts, player_pass_counts)
    print("Done!")

if __name__ == '__main__':
    main()
