#!/usr/bin/env python3
"""
Hand crop processing module for the Hand Gesture Application.
Extracts and centers hand regions from video frames.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List, Union

class HandCropProcessor:
    """
    Processes hand crops for gesture recognition.
    """
    def __init__(self, target_size: int = 150, padding: int = 30):
        self.target_size = target_size
        self.padding = padding
        
    def extract_hand_crop(self, frame: np.ndarray, 
                         hand_landmarks: Union[List, Dict]) -> Optional[np.ndarray]:
        """
        Extract hand crop from frame using hand landmarks.
        
        Args:
            frame: RGB image frame
            hand_landmarks: Either a list of (x, y) tuples for hand landmarks,
                           or a landmarks dict with 'hands' key
            
        Returns:
            Cropped and centered hand image or None
        """
        # Handle dict input (from skeleton_renderer)
        if isinstance(hand_landmarks, dict):
            if not hand_landmarks.get('hands'):
                return None
            hand_landmarks = hand_landmarks['hands'][0]
        
        if not hand_landmarks:
            return None
            
        height, width, _ = frame.shape
        
        # Calculate bounding box
        x_coords = [lm[0] for lm in hand_landmarks]
        y_coords = [lm[1] for lm in hand_landmarks]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding
        x_min = max(0, x_min - self.padding)
        x_max = min(width, x_max + self.padding)
        y_min = max(0, y_min - self.padding)
        y_max = min(height, y_max + self.padding)
        
        # Extract crop
        crop = frame[y_min:y_max, x_min:x_max]
        
        if crop.size == 0:
            return None
            
        # Resize to target size
        crop_resized = cv2.resize(crop, (self.target_size, self.target_size))
        
        return crop_resized
    
    def center_hand(self, crop: np.ndarray) -> np.ndarray:
        """
        Center the hand within the crop using moments.
        
        Args:
            crop: Hand crop image
            
        Returns:
            Centered hand image
        """
        # Convert to grayscale for moment calculation
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        
        # Calculate moments
        moments = cv2.moments(gray)
        if moments["m00"] == 0:
            return crop
            
        # Calculate centroid
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        
        # Calculate shift to center
        height, width = gray.shape
        shift_x = width // 2 - cx
        shift_y = height // 2 - cy
        
        # Apply translation
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        centered = cv2.warpAffine(crop, M, (width, height))
        
        return centered
    
    def preprocess_for_model(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess hand crop for model inference.
        
        Args:
            crop: Hand crop image
            
        Returns:
            Preprocessed image array
        """
        # Normalize to [0, 1]
        processed = crop.astype(np.float32) / 255.0
        
        # Add batch dimension
        processed = np.expand_dims(processed, axis=0)
        
        return processed
