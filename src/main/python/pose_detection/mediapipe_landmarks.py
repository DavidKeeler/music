# MediaPipe Pose Landmarks (33 total points)
# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

MEDIAPIPE_POSE_LANDMARKS = {
    # Face (5 points)
    0: 'nose',
    1: 'left_eye_inner',
    2: 'left_eye',
    3: 'left_eye_outer', 
    4: 'right_eye_inner',
    5: 'right_eye',
    6: 'right_eye_outer',
    7: 'left_ear',
    8: 'right_ear',
    9: 'mouth_left',
    10: 'mouth_right',
    
    # Arms and shoulders (12 points)
    11: 'left_shoulder',
    12: 'right_shoulder',
    13: 'left_elbow',
    14: 'right_elbow',
    15: 'left_wrist',
    16: 'right_wrist',
    17: 'left_pinky',
    18: 'right_pinky',
    19: 'left_index',
    20: 'right_index',
    21: 'left_thumb',
    22: 'right_thumb',
    
    # Torso and hips (4 points)
    23: 'left_hip',
    24: 'right_hip',
    
    # Legs (10 points)
    25: 'left_knee',
    26: 'right_knee',
    27: 'left_ankle',
    28: 'right_ankle',
    29: 'left_heel',
    30: 'right_heel',
    31: 'left_foot_index',
    32: 'right_foot_index'
}

# Conducting-relevant subset - Arms and torso (14 points)
CONDUCTING_KEYPOINTS = {
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24
}

# Index list for easy extraction
CONDUCTING_INDICES = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

def get_landmark_name(index):
    """Get landmark name by index."""
    return MEDIAPIPE_POSE_LANDMARKS.get(index, f'unknown_{index}')

def get_conducting_keypoints():
    """Get conducting-relevant keypoint indices and names."""
    return {name: idx for name, idx in CONDUCTING_KEYPOINTS.items()}
