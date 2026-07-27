#!/usr/bin/env python3
"""
Unit test for rule-based gesture recognition logic.
Tests the GestureRecognizer with synthetic MediaPipe-style landmarks.

Note: In image coordinates, y increases downward.
- Extended finger: tip.y < pip.y (tip is higher up)
- Curled finger: tip.y > pip.y (tip is lower, near palm)
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.model_manager import ModelManager
from hand_processing.gesture_recognizer import GestureRecognizer


class MockLandmark:
    """Mock MediaPipe landmark with x, y, z attributes."""
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class MockHandLandmarks:
    """Mock MediaPipe hand landmarks object."""
    def __init__(self, points):
        self.landmark = [MockLandmark(*p) for p in points]


def create_fist_landmarks():
    """Create landmarks for a fist (all fingers curled, tips below PIP)."""
    points = [(0.5, 0.8)]  # 0 wrist
    # Thumb (curled across palm)
    points += [(0.4, 0.7), (0.38, 0.68), (0.36, 0.66), (0.34, 0.65)]
    # Index: MCP, PIP, DIP, TIP (curled - tip below pip)
    points += [(0.45, 0.6), (0.45, 0.5), (0.45, 0.53), (0.45, 0.56)]
    # Middle
    points += [(0.5, 0.58), (0.5, 0.48), (0.5, 0.51), (0.5, 0.54)]
    # Ring
    points += [(0.55, 0.6), (0.55, 0.5), (0.55, 0.53), (0.55, 0.56)]
    # Pinky
    points += [(0.6, 0.62), (0.6, 0.52), (0.6, 0.55), (0.6, 0.58)]
    return MockHandLandmarks(points)


def create_open_palm_landmarks():
    """Create landmarks for open palm (all fingers extended, tips above PIP)."""
    points = [(0.5, 0.8)]  # 0 wrist
    # Thumb (extended to side)
    points += [(0.4, 0.7), (0.35, 0.65), (0.3, 0.6), (0.25, 0.55)]
    # Index: extended (tip above pip)
    points += [(0.45, 0.6), (0.45, 0.45), (0.45, 0.35), (0.45, 0.25)]
    # Middle
    points += [(0.5, 0.58), (0.5, 0.43), (0.5, 0.33), (0.5, 0.23)]
    # Ring
    points += [(0.55, 0.6), (0.55, 0.45), (0.55, 0.35), (0.55, 0.25)]
    # Pinky
    points += [(0.6, 0.62), (0.6, 0.47), (0.6, 0.37), (0.6, 0.27)]
    return MockHandLandmarks(points)


def create_point_landmarks():
    """Create landmarks for pointing (index extended, others curled)."""
    points = [(0.5, 0.8)]  # 0 wrist
    # Thumb (curled)
    points += [(0.4, 0.7), (0.38, 0.68), (0.36, 0.66), (0.34, 0.65)]
    # Index: extended
    points += [(0.45, 0.6), (0.45, 0.45), (0.45, 0.35), (0.45, 0.25)]
    # Middle: curled
    points += [(0.5, 0.58), (0.5, 0.48), (0.5, 0.51), (0.5, 0.54)]
    # Ring: curled
    points += [(0.55, 0.6), (0.55, 0.5), (0.55, 0.53), (0.55, 0.56)]
    # Pinky: curled
    points += [(0.6, 0.62), (0.6, 0.52), (0.6, 0.55), (0.6, 0.58)]
    return MockHandLandmarks(points)


def create_victory_landmarks():
    """Create landmarks for victory (index + middle extended, others curled)."""
    points = [(0.5, 0.8)]  # 0 wrist
    # Thumb (curled)
    points += [(0.4, 0.7), (0.38, 0.68), (0.36, 0.66), (0.34, 0.65)]
    # Index: extended
    points += [(0.45, 0.6), (0.45, 0.45), (0.45, 0.35), (0.45, 0.25)]
    # Middle: extended
    points += [(0.5, 0.58), (0.5, 0.43), (0.5, 0.33), (0.5, 0.23)]
    # Ring: curled
    points += [(0.55, 0.6), (0.55, 0.5), (0.55, 0.53), (0.55, 0.56)]
    # Pinky: curled
    points += [(0.6, 0.62), (0.6, 0.52), (0.6, 0.55), (0.6, 0.58)]
    return MockHandLandmarks(points)


def test_recognition():
    """Test rule-based gesture recognition."""
    print("Testing rule-based gesture recognition...")
    
    # Create model manager (no model - should use rule-based)
    mm = ModelManager()
    print(f"Model loaded: {mm.get_current_model() is not None}")
    
    recognizer = GestureRecognizer(mm)
    print(f"Has model: {recognizer.has_model}")
    
    # Test fist
    fist_lm = create_fist_landmarks()
    gesture, palm = recognizer.recognize_gesture(None, fist_lm)
    print(f"Fist test -> Gesture: {gesture}, Palm: {palm}")
    assert gesture == "fist", f"Expected 'fist', got '{gesture}'"
    
    # Test open palm
    open_lm = create_open_palm_landmarks()
    gesture, palm = recognizer.recognize_gesture(None, open_lm)
    print(f"Open palm test -> Gesture: {gesture}, Palm: {palm}")
    assert gesture == "open_palm", f"Expected 'open_palm', got '{gesture}'"
    
    # Test point
    point_lm = create_point_landmarks()
    gesture, palm = recognizer.recognize_gesture(None, point_lm)
    print(f"Point test -> Gesture: {gesture}, Palm: {palm}")
    assert gesture == "point", f"Expected 'point', got '{gesture}'"
    
    # Test victory
    victory_lm = create_victory_landmarks()
    gesture, palm = recognizer.recognize_gesture(None, victory_lm)
    print(f"Victory test -> Gesture: {gesture}, Palm: {palm}")
    assert gesture == "victory", f"Expected 'victory', got '{gesture}'"
    
    print("\nAll gesture recognition tests passed!")


if __name__ == "__main__":
    test_recognition()
