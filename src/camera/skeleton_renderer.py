#!/usr/bin/env python3
"""
Skeleton rendering module for the Hand Gesture Application.
Draws facial landmarks, hand skeleton, and body joints on video frames.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, Optional, Any

class SkeletonRenderer:
    """
    Handles skeleton rendering using MediaPipe.
    """
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        
        # Create instances
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Colors for different skeleton parts
        self.colors = {
            'face': (0, 255, 0),      # Green
            'hands': (255, 0, 0),     # Blue
            'pose': (0, 0, 255),      # Red
            'landmarks': (255, 255, 255)  # White
        }
        
        # Landmark connections
        self.face_connections = self.mp_face_mesh.FACEMESH_TESSELATION
        self.hand_connections = self.mp_hands.HAND_CONNECTIONS
        self.pose_connections = self.mp_pose.POSE_CONNECTIONS

    def process_frame(self, frame: np.ndarray, show_face: bool = True,
                     show_hand: bool = True, show_pose: bool = True) -> Tuple[np.ndarray, Dict, Any]:
        """
        Process a frame and draw skeletons.
        
        Args:
            frame: RGB image frame
            show_face: Whether to draw face skeleton
            show_hand: Whether to draw hand skeleton
            show_pose: Whether to draw body pose skeleton
            
        Returns:
            Tuple of (processed frame, landmarks dict, raw_results)
            landmarks dict contains:
                - 'face': list of (x, y) pixel coordinates
                - 'hands': list of lists of (x, y) pixel coordinates
                - 'pose': list of (x, y) pixel coordinates
                - 'raw_hands': list of raw MediaPipe hand landmark objects
                - 'handedness': list of handedness info
        """
        # Convert to RGB if needed
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
            
        # Process with MediaPipe
        image_height, image_width, _ = frame.shape
        
        # Process face mesh
        face_results = self.face_mesh.process(frame) if show_face else None
        
        # Process hands
        hand_results = self.hands.process(frame) if show_hand else None
        
        # Process pose
        pose_results = self.pose.process(frame) if show_pose else None
        
        # Create landmarks dictionary
        landmarks = {
            'face': [],
            'hands': [],
            'pose': [],
            'raw_hands': [],
            'handedness': []
        }
        
        # Draw face landmarks
        if show_face and face_results and face_results.multi_face_landmarks:
            for face_landmark in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmark,
                    connections=self.face_connections,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_style()
                )
                # Store landmarks
                landmarks['face'] = [
                    (int(lm.x * image_width), int(lm.y * image_height))
                    for lm in face_landmark.landmark
                ]
        
        # Draw hand landmarks
        if show_hand and hand_results and hand_results.multi_hand_landmarks:
            for idx, hand_landmark in enumerate(hand_results.multi_hand_landmarks):
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmark,
                    connections=self.hand_connections,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
                )
                # Store landmarks
                landmarks['hands'].append([
                    (int(lm.x * image_width), int(lm.y * image_height))
                    for lm in hand_landmark.landmark
                ])
                # Store raw landmarks for gesture recognition
                landmarks['raw_hands'].append(hand_landmark)
                # Store handedness
                if hand_results.multi_handedness and idx < len(hand_results.multi_handedness):
                    landmarks['handedness'].append(
                        hand_results.multi_handedness[idx].classification[0].label
                    )
                else:
                    landmarks['handedness'].append("Unknown")
        
        # Draw pose landmarks
        if show_pose and pose_results and pose_results.multi_pose_landmarks:
            for pose_landmark in pose_results.multi_pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=pose_landmark,
                    connections=self.pose_connections,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                # Store landmarks
                landmarks['pose'] = [
                    (int(lm.x * image_width), int(lm.y * image_height))
                    for lm in pose_landmark.landmark
                ]
        
        return frame, landmarks, (face_results, hand_results, pose_results)

    def extract_hand_crop(self, frame: np.ndarray, landmarks: Dict, 
                         padding: int = 30) -> Optional[np.ndarray]:
        """
        Extract and crop the hand region from landmarks.
        
        Args:
            frame: Original RGB frame
            landmarks: Dictionary containing landmark coordinates
            padding: Padding around the hand crop
            
        Returns:
            Cropped hand image (150x150) or None if no hand detected
        """
        if not landmarks['hands']:
            return None
            
        # Use the first detected hand
        hand_landmarks = landmarks['hands'][0]
        
        # Calculate bounding box
        x_coords = [lm[0] for lm in hand_landmarks]
        y_coords = [lm[1] for lm in hand_landmarks]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding
        height, width, _ = frame.shape
        x_min = max(0, x_min - padding)
        x_max = min(width, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(height, y_max + padding)
        
        # Extract crop
        crop = frame[y_min:y_max, x_min:x_max]
        
        if crop.size == 0:
            return None
        
        # Resize to 150x150
        crop_resized = cv2.resize(crop, (150, 150))
        
        return crop_resized

    def get_eye_closure(self, landmarks: Dict) -> float:
        """
        Calculate eye closure ratio from facial landmarks.
        
        Args:
            landmarks: Dictionary containing facial landmarks
            
        Returns:
            Eye closure ratio (0.0 to 1.0)
        """
        if not landmarks['face']:
            return 0.0
            
        # Get eye landmarks (MediaPipe face mesh indices)
        # Left eye: landmarks 474, 475, 476, 477, 478, 479
        # Right eye: landmarks 469, 470, 471, 472, 473, 474
        left_eye = landmarks['face'][474:480]
        right_eye = landmarks['face'][469:474]
        
        if len(left_eye) < 2 or len(right_eye) < 2:
            return 0.0
            
        # Calculate eye aspect ratio (EAR)
        def eye_aspect_ratio(eye):
            # Vertical distances
            v1 = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
            v2 = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
            # Horizontal distance
            h = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
            # EAR formula
            return (v1 + v2) / (2.0 * h) if h > 0 else 0.0
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        
        return (left_ear + right_ear) / 2.0

    def release(self):
        """Release MediaPipe resources."""
        # MediaPipe doesn't require explicit cleanup
        pass
