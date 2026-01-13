import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.cluster import KMeans

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_PATH = 'models/best.pt'
INPUT_IMAGE_PATH = 'input1.jpg'
OUTPUT_IMAGE_PATH = 'final_offside_result.jpg'

# ------------------------------------------------------------------------------
# PERSPECTIVE MAPPING (Standardized for Right-Attack)
# ------------------------------------------------------------------------------
# We map the 4 corners of the 18-yard box.
# We set the Left Line (18-yard) to X=0.
# We set the Right Line (Goal Line) to X=16.5.
# This means X INCREASES towards the goal (Right).
DST_POINTS = np.array([
    [0, 40.32],    # Click 1: Top-Left (Far 18-yard corner) -> X=0
    [0, 0],        # Click 2: Bottom-Left (Near 18-yard corner) -> X=0
    [16.5, 0],     # Click 3: Bottom-Right (Near Goal line corner) -> X=16.5
    [16.5, 40.32]  # Click 4: Top-Right (Far Goal line corner) -> X=16.5
], dtype=np.float32)

# ==============================================================================
# 1. UTILS & CLASSES
# ==============================================================================

def check_gpu():
    return 0 if torch.cuda.is_available() else 'cpu'

def get_foot_position(bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return int((x1 + x2) / 2), int(y2)

class PerspectiveManager:
    def __init__(self):
        self.M = None
        self.M_inv = None

    def compute_transform(self, src_points):
        self.M = cv2.getPerspectiveTransform(np.array(src_points, dtype=np.float32), DST_POINTS)
        self.M_inv = cv2.invert(self.M)[1]

    def transform_point(self, point):
        if self.M is None: return None
        p = np.array([[[point[0], point[1]]]], dtype=np.float32)
        world_pt = cv2.perspectiveTransform(p, self.M)[0][0]
        return world_pt # [x, y]

    def project_line(self, x_depth):
        # Draw line at constant X
        world_line = np.array([
            [[x_depth, -20]],
            [[x_depth, 100]]
        ], dtype=np.float32)

        pixel_line = cv2.perspectiveTransform(world_line, self.M_inv)
        pt1 = tuple(map(int, pixel_line[0][0]))
        pt2 = tuple(map(int, pixel_line[1][0]))
        return pt1, pt2

class TeamAssigner:
    def __init__(self):
        self.team_centers = None
        self.kmeans = None

    def get_jersey_color_and_feature(self, frame, box_xyxy):
        x1, y1, x2, y2 = map(int, box_xyxy)
        h, w = y2 - y1, x2 - x1
        if h < 5 or w < 5: return (255, 255, 255), np.zeros(3)

        # Center Crop
        crop = frame[y1+int(h*0.2):y1+int(h*0.5), x1+int(w*0.3):x1+int(w*0.7)]
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

    def assign_team(self, feature_vec):
        if self.team_centers is None: return 0
        dists = np.linalg.norm(self.team_centers - feature_vec[None, :], axis=1)
        return int(np.argmin(dists))

# ==============================================================================
# 2. INTERACTIVE MOUSE LOGIC
# ==============================================================================

# Global State
original_frame = None
annotated_base = None
display_img = None
perspective_mgr = None
players_data = [] # Stores {bbox, foot, team}
calibration_points = []
state = "CALIBRATE"

def mouse_callback(event, x, y, flags, param):
    global state, calibration_points, display_img, annotated_base, perspective_mgr

    if event == cv2.EVENT_LBUTTONDOWN:
        if state == "CALIBRATE":
            calibration_points.append((x, y))

            # Draw point
            cv2.circle(display_img, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(display_img, str(len(calibration_points)), (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cv2.imshow("Offside Tool", display_img)

            if len(calibration_points) == 4:
                print("Computing Perspective...")
                pts = np.array(calibration_points, dtype=np.float32)
                perspective_mgr.compute_transform(pts)

                # Visual feedback
                pts_int = pts.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(display_img, [pts_int], True, (0, 255, 0), 2)
                cv2.imshow("Offside Tool", display_img)

                state = "SELECT_LAST_MAN"
                print("\n>>> PHASE 2: Click the LAST MAN (Defender) <<<")

        elif state == "SELECT_LAST_MAN":
            # Refresh to clear old lines
            temp_img = annotated_base.copy()

            # 1. Identify Clicked Player
            click_pt = np.array((x, y))
            last_man = None
            min_dist = 9999

            for p in players_data:
                dist = np.linalg.norm(click_pt - np.array(p['foot']))
                if dist < min_dist:
                    min_dist = dist
                    last_man = p

            if last_man:
                # 2. Draw Line
                last_man_world = perspective_mgr.transform_point(last_man['foot'])
                line_x = last_man_world[0]
                pt1, pt2 = perspective_mgr.project_line(line_x)

                cv2.line(temp_img, pt1, pt2, (0, 255, 255), 3) # Yellow

                # Mark Defender
                lx, ly = last_man['foot']
                cv2.circle(temp_img, (lx, ly), 8, (0, 255, 255), -1)
                cv2.putText(temp_img, "LAST MAN", (lx+10, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # 3. Check Offside (Attack Right -> X > Line = Offside)
                def_team = last_man['team']

                for p in players_data:
                    # Logic: Only check opponents
                    if p['team'] != def_team:
                        p_world = perspective_mgr.transform_point(p['foot'])
                        p_x = p_world[0]
                        bx1, by1, bx2, by2 = map(int, p['bbox'])

                        # Tolerance 0.2m
                        if p_x > (line_x + 0.2):
                            color = (0, 0, 255) # Red
                            # Draw OFFSIDE Label
                            cv2.rectangle(temp_img, (bx1, by1 - 25), (bx1 + 100, by1), color, -1)
                            cv2.putText(temp_img, "OFFSIDE", (bx1 + 5, by1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                            cv2.rectangle(temp_img, (bx1, by1), (bx2, by2), color, 2)
                        else:
                            color = (0, 255, 0) # Green
                            # Just draw box, no text
                            cv2.rectangle(temp_img, (bx1, by1), (bx2, by2), color, 2)

            cv2.imshow("Offside Tool", temp_img)
            cv2.imwrite(OUTPUT_IMAGE_PATH, temp_img)
            print(f"Updated line. Saved to {OUTPUT_IMAGE_PATH}")

# ==============================================================================
# 4. MAIN
# ==============================================================================

def main():
    global original_frame, annotated_base, display_img, perspective_mgr, players_data

    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO(MODEL_PATH)

    print(f"Loading {INPUT_IMAGE_PATH}...")
    original_frame = cv2.imread(INPUT_IMAGE_PATH)
    if original_frame is None: return

    perspective_mgr = PerspectiveManager()
    team_assigner = TeamAssigner()

    # 1. Detect & Cluster
    print("Detecting players...")
    results = model(original_frame, conf=0.25, device=device)[0]

    feats = []
    temp_boxes = []

    for box in results.boxes:
        if int(box.cls[0]) == 2: # Player
            bbox = box.xyxy[0].cpu().numpy()
            _, feat = team_assigner.get_jersey_color_and_feature(original_frame, bbox)
            feats.append(feat)
            temp_boxes.append(bbox)

    if len(feats) > 1:
        team_assigner.init_team_centers(np.array(feats))

    # 2. Build Data & Draw Base
    annotated_base = original_frame.copy()

    for i, bbox in enumerate(temp_boxes):
        tid = team_assigner.assign_team(feats[i])
        foot = get_foot_position(bbox)

        players_data.append({
            'bbox': bbox,
            'foot': foot,
            'team': tid
        })

        # Base Visual (Grey Box)
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated_base, (x1, y1), (x2, y2), (200, 200, 200), 1)

    # 3. Interactive Loop
    display_img = annotated_base.copy()
    cv2.namedWindow("Offside Tool")
    cv2.setMouseCallback("Offside Tool", mouse_callback)

    print("\n--- INSTRUCTIONS ---")
    print("1. Click 4 corners of the PENALTY BOX:")
    print("   Order: Top-Left -> Bottom-Left -> Bottom-Right -> Top-Right")
    print("2. Then Click the LAST MAN (Defender).")

    while True:
        cv2.imshow("Offside Tool", display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
