#!/usr/bin/env python3
"""
Gesture recognition module for the Hand Gesture Application.
Classifies hand gestures using a trained model or rule-based fallback.
"""

import numpy as np
from typing import Tuple, Optional, List, Any

# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


class GestureRecognizer:
    """
    Recognizes hand gestures from preprocessed hand crops or landmarks.
    Uses trained model if available, otherwise falls back to rule-based recognition.
    """
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.model = model_manager.get_current_model()
        self.has_model = self.model is not None
        
        # Default gesture names for rule-based recognition
        self.rule_based_gestures = [
            "fist", "open_palm", "point", "victory", "thumbs_up",
            "thumbs_down", "ok_sign", "rock", "peace", "three",
            "four", "five", "call_me", "gun", "pinch",
            "unknown"
        ]
        
    def recognize_gesture(self, hand_crop: Optional[np.ndarray] = None, 
                         raw_landmarks: Optional[Any] = None) -> Tuple[str, str]:
        """
        Recognize gesture from hand crop and/or raw landmarks.
        
        Args:
            hand_crop: Preprocessed hand crop image (for model-based recognition)
            raw_landmarks: Raw MediaPipe hand landmark object (for rule-based)
            
        Returns:
            Tuple of (gesture_name, palm_side)
        """
        # Use model if available and hand_crop provided
        if self.has_model and hand_crop is not None:
            try:
                gesture_name = self._model_recognize(hand_crop)
                palm_side = self._determine_palm_side(raw_landmarks, hand_crop)
                return gesture_name, palm_side
            except Exception as e:
                print(f"Model recognition failed, using rule-based: {e}")
        
        # Fall back to rule-based recognition
        if raw_landmarks is not None:
            gesture_name = self._rule_based_recognize(raw_landmarks)
            palm_side = self._determine_palm_side(raw_landmarks, hand_crop)
            return gesture_name, palm_side
        
        return "Unknown", "Unknown"
        
    def _model_recognize(self, hand_crop: np.ndarray) -> str:
        """Run model inference on hand crop."""
        input_tensor = self.model_manager.preprocess_input(hand_crop)
        predictions = self.model_manager.run_inference(input_tensor)
        gesture_index = np.argmax(predictions)
        
        gesture_names = self.model_manager.get_gesture_names()
        if gesture_index < len(gesture_names):
            return gesture_names[gesture_index]
        return "Unknown"
        
    def _rule_based_recognize(self, landmarks: Any) -> str:
        """
        Rule-based gesture recognition using MediaPipe hand landmarks.
        
        Args:
            landmarks: MediaPipe hand landmark object with .landmark list
            
        Returns:
            Gesture name string
        """
        lm = landmarks.landmark
        
        # Calculate finger states (extended or curled)
        fingers_extended = self._get_finger_states(lm)
        
        # Count extended fingers
        num_extended = sum(fingers_extended)
        
        # Thumb position
        thumb_extended = fingers_extended[0]
        
        # Special cases
        # OK sign: thumb and index form circle, others extended
        if self._is_ok_sign(lm):
            return "ok_sign"
        
        # Rock sign (horns): index and pinky extended, others curled
        if fingers_extended[1] and fingers_extended[4] and not fingers_extended[2] and not fingers_extended[3]:
            return "rock"
        
        # Gun sign: thumb, index extended, middle curled, ring+pinky curled
        if thumb_extended and fingers_extended[1] and not fingers_extended[2] and not fingers_extended[3] and not fingers_extended[4]:
            return "gun"
        
        # Call me: thumb and pinky extended, others curled
        if thumb_extended and not fingers_extended[1] and not fingers_extended[2] and not fingers_extended[3] and fingers_extended[4]:
            return "call_me"
        
        # Pinch: thumb tip close to index tip
        if self._is_pinch(lm):
            return "pinch"
        
        # Based on number of extended fingers
        if num_extended == 0:
            return "fist"
        elif num_extended == 5:
            return "open_palm"
        elif num_extended == 1:
            if fingers_extended[1]:
                return "point"
            elif thumb_extended:
                return "thumbs_up" if self._is_thumb_up(lm) else "thumbs_down"
        elif num_extended == 2:
            if fingers_extended[1] and fingers_extended[2]:
                return "victory"
            elif fingers_extended[2] and fingers_extended[3]:
                return "peace"
            elif fingers_extended[1] and fingers_extended[3]:
                return "three"  # approximation
        elif num_extended == 3:
            if fingers_extended[1] and fingers_extended[2] and fingers_extended[3]:
                return "three"
        elif num_extended == 4:
            if all(fingers_extended[1:5]):
                return "four"
        
        return "unknown"
        
    def _get_finger_states(self, lm: Any) -> List[bool]:
        """
        Determine which fingers are extended.
        
        Returns:
            List of 5 booleans: [thumb, index, middle, ring, pinky]
        """
        fingers = []
        
        # Thumb: compare tip x to IP joint x (accounting for handedness)
        # For simplicity, use distance from wrist
        thumb_tip = lm[THUMB_TIP]
        thumb_mcp = lm[THUMB_MCP]
        # Thumb extended if tip is far from palm
        thumb_dist = np.linalg.norm([thumb_tip.x - thumb_mcp.x, thumb_tip.y - thumb_mcp.y])
        thumb_extended = thumb_dist > 0.1
        fingers.append(thumb_extended)
        
        # Other fingers: tip should be higher (smaller y) than PIP joint
        # Index
        index_tip = lm[INDEX_TIP]
        index_pip = lm[INDEX_PIP]
        fingers.append(index_tip.y < index_pip.y)
        
        # Middle
        middle_tip = lm[MIDDLE_TIP]
        middle_pip = lm[MIDDLE_PIP]
        fingers.append(middle_tip.y < middle_pip.y)
        
        # Ring
        ring_tip = lm[RING_TIP]
        ring_pip = lm[RING_PIP]
        fingers.append(ring_tip.y < ring_pip.y)
        
        # Pinky
        pinky_tip = lm[PINKY_TIP]
        pinky_pip = lm[PINKY_PIP]
        fingers.append(pinky_tip.y < pinky_pip.y)
        
        return fingers
        
    def _is_ok_sign(self, lm: Any) -> bool:
        """Check if gesture is OK sign (thumb and index form circle)."""
        thumb_tip = lm[THUMB_TIP]
        index_tip = lm[INDEX_TIP]
        # Distance between thumb tip and index tip
        dist = np.linalg.norm([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y])
        # Other fingers should be extended
        fingers = self._get_finger_states(lm)
        return dist < 0.05 and fingers[2] and fingers[3] and fingers[4]
        
    def _is_pinch(self, lm: Any) -> bool:
        """Check if thumb and index are pinched together."""
        thumb_tip = lm[THUMB_TIP]
        index_tip = lm[INDEX_TIP]
        dist = np.linalg.norm([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y])
        return dist < 0.03
        
    def _is_thumb_up(self, lm: Any) -> bool:
        """Check if thumb is pointing up."""
        thumb_tip = lm[THUMB_TIP]
        wrist = lm[WRIST]
        # Thumb tip should be above wrist (smaller y)
        return thumb_tip.y < wrist.y
        
    def _determine_palm_side(self, raw_landmarks: Optional[Any] = None, 
                            hand_crop: Optional[np.ndarray] = None) -> str:
        """
        Determine if palm is facing front (palm) or back (back of hand).
        
        Args:
            raw_landmarks: MediaPipe hand landmark object
            hand_crop: Hand crop image (unused fallback)
            
        Returns:
            "front" or "back"
        """
        if raw_landmarks is not None:
            lm = raw_landmarks.landmark
            # Use the relative depth (z) of middle finger MCP vs wrist
            # In MediaPipe, negative z is closer to camera
            wrist_z = lm[WRIST].z
            middle_mcp_z = lm[MIDDLE_MCP].z
            
            # If middle MCP is closer to camera than wrist, palm is facing camera
            # This is a heuristic - actual orientation depends on hand pose
            if middle_mcp_z < wrist_z:
                return "front"
            else:
                return "back"
        
        # Fallback: random (should not happen in practice)
        return "front" if np.random.rand() > 0.5 else "back"
        
    def get_gesture_names(self) -> list:
        """
        Get list of known gesture names.
        """
        if self.has_model:
            return self.model_manager.get_gesture_names()
        return self.rule_based_gestures
