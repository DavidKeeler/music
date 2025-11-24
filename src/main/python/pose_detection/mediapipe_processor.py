import mediapipe as mp
import numpy as np
import cv2
from .mediapipe_landmarks import CONDUCTING_INDICES

mp_pose = mp.solutions.pose

def extract_keypoints(frame, pose):
    """Run MediaPipe and return (14, 2) conducting keypoints or None."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(frame_rgb)
    
    if not res.pose_landmarks:
        return None
    
    # Extract all 33 landmarks first
    all_pts = np.array([[lm.x, lm.y] for lm in res.pose_landmarks.landmark])
    
    # Select only conducting-relevant keypoints
    pts = all_pts[CONDUCTING_INDICES]
    return pts  # shape (14, 2)

def normalize_translation(pts):
    """Center at hip midpoint."""
    hip_center = (pts[12] + pts[13]) / 2.0  # left_hip=12, right_hip=13 in our 14-point array
    return pts - hip_center

def normalize_scale(pts):
    """Divide by shoulder width."""
    shoulder_width = np.linalg.norm(pts[0] - pts[1]) + 1e-6  # left_shoulder=0, right_shoulder=1
    return pts / shoulder_width

def estimate_body_angle(pts):
    """Robust angle estimation using shoulders + hips."""
    shoulder_vec = pts[0] - pts[1]  # left_shoulder - right_shoulder
    hip_vec = pts[12] - pts[13]     # left_hip - right_hip
    vec = (shoulder_vec + hip_vec) / 2.0
    return np.arctan2(vec[1], vec[0])

def normalize_rotation(pts):
    """Align body horizontally using robust angle estimation."""
    angle = estimate_body_angle(pts)
    
    # 2D rotation matrix
    rot = np.array([
        [ np.cos(-angle), -np.sin(-angle)],
        [ np.sin(-angle),  np.cos(-angle)]
    ])
    
    return pts @ rot.T

def process_frame(frame, pose):
    """Full normalization pipeline with fixes."""
    pts = extract_keypoints(frame, pose)
    if pts is None:
        return None
    
    # Convert normalized coords → pixel coords
    h, w = frame.shape[:2]
    pts[:, 0] *= w
    pts[:, 1] *= h
    
    pts = normalize_translation(pts)
    pts = normalize_rotation(pts)
    pts = normalize_scale(pts)
    
    return pts  # (14,2) normalized keypoints

def run_pose_sequence(video_path):
    """Extract normalized pose sequence from video."""
    cap = cv2.VideoCapture(video_path)
    seq = []
    
    with mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            pts = process_frame(frame, pose)
            if pts is not None:
                seq.append(pts)
    
    cap.release()
    return np.stack(seq) if seq else None  # (T, 14, 2)
