#!/usr/bin/env python3
"""
Process dynamics data extracted from hand landmarks.

This script is automatically triggered by CameraWorker after extracting
21 hand landmarks for dynamic gesture recognition. It processes the data
and performs further analysis.
"""

import os
import sys
import json
import time
import threading
from typing import List, Dict, Any

# Add the hand_gesture_app directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.hand_processing.gesture_fusion import GestureFusion

# Global configuration
DYNAMICS_DATA_DIR = "F:\\MODELS\\Model_lerning\\dynamics_data"
MODEL_CONFIG_PATH = "hand_gesture_app\\model_config.json"


def load_model_config() -> Dict[str, Any]:
    """Load model configuration from JSON file."""
    try:
        with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ProcessDynamics] Failed to load model config: {e}")
        return {}


def extract_dynamics_from_landmarks(landmarks: List[Any]) -> Dict[str, Any]:
    """
    Extract dynamics features from hand landmarks.
    
    Args:
        landmarks: List of MediaPipe hand landmarks
        
    Returns:
        Dictionary containing extracted dynamics features
    """
    # Extract 21 hand landmarks
    landmark_data = []
    for lm in landmarks:
        landmark_data.append({
            'x': lm.x,
            'y': lm.y,
            'z': lm.z
        })
    
    # Calculate dynamics features
    features = {
        'landmarks': landmark_data,
        'timestamp': time.time(),
        'landmark_count': len(landmark_data),
        'hand_size': calculate_hand_size(landmark_data),
        'finger_spread': calculate_finger_spread(landmark_data),
        'gesture_velocity': 0.0,  # Would be calculated from sequence
        'confidence': 0.85  # Placeholder
    }
    
    return features


def calculate_hand_size(landmarks: List[Dict[str, float]]) -> float:
    """Calculate the size of the hand based on landmarks."""
    if not landmarks:
        return 0.0
    
    # Calculate bounding box
    x_coords = [lm['x'] for lm in landmarks]
    y_coords = [lm['y'] for lm in landmarks]
    
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    
    return (width + height) / 2.0


def calculate_finger_spread(landmarks: List[Dict[str, float]]) -> float:
    """Calculate the spread between fingers."""
    if len(landmarks) < 5:
        return 0.0
    
    # Use tip landmarks (4, 8, 12, 16, 20) for finger spread calculation
    tip_indices = [4, 8, 12, 16, 20]
    tip_landmarks = [landmarks[i] for i in tip_indices if i < len(landmarks)]
    
    if len(tip_landmarks) < 2:
        return 0.0
    
    # Calculate average distance between fingertips
    distances = []
    for i in range(len(tip_landmarks)):
        for j in range(i + 1, len(tip_landmarks)):
            dist = ((tip_landmarks[i]['x'] - tip_landmarks[j]['x']) ** 2 +
                    (tip_landmarks[i]['y'] - tip_landmarks[j]['y']) ** 2 +
                    (tip_landmarks[i]['z'] - tip_landmarks[j]['z']) ** 2) ** 0.5
            distances.append(dist)
    
    return sum(distances) / len(distances) if distances else 0.0


def save_dynamics_data(features: Dict[str, Any]) -> str:
    """
    Save extracted dynamics data to file.
    
    Args:
        features: Dictionary containing dynamics features
        
    Returns:
        Path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(DYNAMICS_DATA_DIR, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = int(time.time())
    filename = f"dynamics_{timestamp}.json"
    filepath = os.path.join(DYNAMICS_DATA_DIR, filename)
    
    # Save data
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(features, f, indent=2, ensure_ascii=False)
    
    print(f"[ProcessDynamics] Saved dynamics data to {filepath}")
    return filepath


def process_dynamics_file(filepath: str):
    """
    Process a saved dynamics data file.
    
    Args:
        filepath: Path to the dynamics data file
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"[ProcessDynamics] Processing dynamics data from {filepath}")
        
        # Extract features
        landmarks = data.get('landmarks', [])
        if landmarks:
            # Convert to MediaPipe landmark format (simplified)
            class SimpleLandmark:
                def __init__(self, x, y, z):
                    self.x = x
                    self.y = y
                    self.z = z
            
            mp_landmarks = [SimpleLandmark(lm['x'], lm['y'], lm['z']) for lm in landmarks]
            
            # Load model config and create GestureFusion instance
            model_config = load_model_config()
            current_model_id = model_config.get('current_static_model_id', 'mobilenet')
            
            # Create GestureFusion with current model
            fusion = GestureFusion(
                ha_grid_model_path=None,  # Will use default from config
                hand_landmarker_path=None,  # Will use default
                device=model_config.get('device', 'cpu')
            )
            
            # Load the appropriate static model
            fusion.load_static_model_by_id(current_model_id, model_config)
            
            # Process the landmarks
            if fusion._model is not None:
                # Convert frame to numpy array (placeholder)
                import numpy as np
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Get prediction
                name, conf, prob_dict = fusion.predict(
                    frame, mp_landmarks, w=640, h=480
                )
                
                print(f"[ProcessDynamics] Predicted gesture: {name} (confidence: {conf:.2f})")
                
                # Save results
                results = {
                    'input_file': filepath,
                    'predicted_gesture': name,
                    'confidence': conf,
                    'probability_distribution': prob_dict,
                    'processing_time': time.time(),
                    'model_used': current_model_id
                }
                
                results_file = filepath.replace('.json', '_result.json')
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"[ProcessDynamics] Saved results to {results_file}")
            else:
                print("[ProcessDynamics] Warning: Could not load model for prediction")
        
    except Exception as e:
        print(f"[ProcessDynamics] Error processing file {filepath}: {e}")


def monitor_dynamics_directory():
    """Monitor the dynamics data directory for new files and process them."""
    print(f"[ProcessDynamics] Monitoring directory: {DYNAMICS_DATA_DIR}")
    
    processed_files = set()
    
    while True:
        try:
            if os.path.exists(DYNAMICS_DATA_DIR):
                files = os.listdir(DYNAMICS_DATA_DIR)
                for filename in files:
                    if filename.endswith('.json') and filename not in processed_files:
                        filepath = os.path.join(DYNAMICS_DATA_DIR, filename)
                        processed_files.add(filename)
                        
                        # Process the file in a separate thread
                        thread = threading.Thread(
                            target=process_dynamics_file,
                            args=(filepath,),
                            daemon=True
                        )
                        thread.start()
            
            time.sleep(1)  # Check every second
            
        except Exception as e:
            print(f"[ProcessDynamics] Error in monitoring loop: {e}")
            time.sleep(5)


def main():
    """Main function."""
    print("[ProcessDynamics] Starting dynamics processing service...")
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_dynamics_directory, daemon=True)
    monitor_thread.start()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[ProcessDynamics] Shutting down...")


if __name__ == "__main__":
    main()
