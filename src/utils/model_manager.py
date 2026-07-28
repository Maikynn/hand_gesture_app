#!/usr/bin/env python3
"""
Model management utilities for the Hand Gesture Application.
Supports loading TFLite and ONNX models for gesture recognition.
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Any

# Import inference engines conditionally to avoid hard dependencies
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False


class ModelManager:
    """
    Manages model loading, inference, and switching between local/cloud models.
    """
    def __init__(self, model_dir: str = "models"):
        """
        Initialize model manager with default model directory.
        """
        self.model_dir = Path(model_dir)
        self.current_model = None
        self.loaded_models = {}
        self.model_type = None  # 'tflite' or 'onnx'
        self.gesture_names = []
        
        # Try to load built-in model (non-fatal if missing)
        try:
            self.load_built_in_model()
        except Exception as e:
            print(f"Warning: Could not load built-in model: {e}")
            print("Application will use rule-based gesture recognition fallback.")
    
    def load_built_in_model(self) -> bool:
        """
        Load the default gesture recognition model.
        Supports both .tflite and .onnx formats.
        """
        # Check for TFLite model first
        built_in_tflite = self.model_dir / "built_in.tflite"
        if built_in_tflite.exists():
            if not TFLITE_AVAILABLE:
                raise RuntimeError("TensorFlow Lite not available for model loading")
            self.current_model = "built_in"
            self.model_type = "tflite"
            self.loaded_models["built_in"] = tf.lite.Interpreter(model_path=str(built_in_tflite))
            self.loaded_models["built_in"].allocate_tensors()
            self._load_gesture_names()
            return True
        
        # Check for ONNX model
        built_in_onnx = self.model_dir / "built_in.onnx"
        if built_in_onnx.exists():
            if not ONNX_AVAILABLE:
                raise RuntimeError("ONNX Runtime not available for model loading")
            self.current_model = "built_in"
            self.model_type = "onnx"
            self.loaded_models["built_in"] = ort.InferenceSession(str(built_in_onnx))
            self._load_gesture_names()
            return True
        
        raise FileNotFoundError("Built-in model not found (expected built_in.tflite or built_in.onnx)")
    
    def load_custom_model(self, model_path: str) -> bool:
        """
        Load a custom model from the custom directory.
        """
        custom_model = self.model_dir / "custom" / model_path
        if not custom_model.exists():
            return False
        try:
            if custom_model.suffix == ".tflite":
                if not TFLITE_AVAILABLE:
                    return False
                self.loaded_models[model_path] = tf.lite.Interpreter(
                    model_path=str(custom_model)
                )
                self.loaded_models[model_path].allocate_tensors()
                self.model_type = "tflite"
            elif custom_model.suffix == ".onnx":
                if not ONNX_AVAILABLE:
                    return False
                self.loaded_models[model_path] = ort.InferenceSession(str(custom_model))
                self.model_type = "onnx"
            else:
                return False
            self.current_model = model_path
            self._load_gesture_names()
            return True
        except Exception as exc:
            print(f"Error loading custom model: {exc}")
            return False

    def load_external_model(self, model_path: str) -> bool:
        """Load an absolute .onnx/.tflite path selected in the camera UI."""
        path = Path(model_path).expanduser()
        if not path.is_absolute() or not path.is_file():
            return False
        try:
            if path.suffix.lower() == ".tflite":
                if not TFLITE_AVAILABLE:
                    return False
                model = tf.lite.Interpreter(model_path=str(path))
                model.allocate_tensors()
                self.model_type = "tflite"
            elif path.suffix.lower() == ".onnx":
                if not ONNX_AVAILABLE:
                    return False
                model = ort.InferenceSession(str(path))
                self.model_type = "onnx"
            else:
                return False
            key = f"external:{path}"
            self.loaded_models[key] = model
            self.current_model = key
            self._load_gesture_names()
            return True
        except Exception as exc:
            print(f"Error loading external model: {exc}")
            return False
    
    def get_current_model(self) -> Optional[Any]:
        """
        Get the currently loaded model session/interpreter.
        """
        return self.loaded_models.get(self.current_model)
    
    def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different model.
        """
        if model_name in self.loaded_models:
            self.current_model = model_name
            return True
        elif model_name == "built_in" and self.loaded_models.get("built_in"):
            self.current_model = "built_in"
            return True
        else:
            return False
    
    def preprocess_input(self, hand_crop: np.ndarray) -> np.ndarray:
        """
        Preprocess hand crop for model inference.
        
        Args:
            hand_crop: RGB image array (150x150x3)
            
        Returns:
            Preprocessed tensor with batch dimension
        """
        # Normalize to [0, 1]
        processed = hand_crop.astype(np.float32) / 255.0
        # Add batch dimension
        processed = np.expand_dims(processed, axis=0)
        return processed
    
    def run_inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Run inference on the current model.
        
        Args:
            input_tensor: Preprocessed input with batch dimension
            
        Returns:
            Model output predictions
        """
        model = self.get_current_model()
        if model is None:
            raise RuntimeError("No model loaded")
        
        if self.model_type == "tflite":
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            model.set_tensor(input_details[0]['index'], input_tensor.astype(input_details[0]['dtype']))
            model.invoke()
            return model.get_tensor(output_details[0]['index'])
        elif self.model_type == "onnx":
            input_name = model.get_inputs()[0].name
            return model.run(None, {input_name: input_tensor})[0]
        else:
            raise RuntimeError("Unknown model type")
    
    def get_gesture_names(self) -> List[str]:
        """
        Get list of known gesture names from model metadata or defaults.
        """
        if self.gesture_names:
            return self.gesture_names
        # Default 40 gestures + unknown
        return [f"Gesture_{i}" for i in range(40)] + ["Unknown"]
    
    def _load_gesture_names(self):
        """Load gesture names from model metadata file if available."""
        names_file = self.model_dir / "gesture_classes.txt"
        if names_file.exists():
            try:
                with open(names_file, "r", encoding="utf-8") as f:
                    self.gesture_names = [line.strip() for line in f if line.strip()]
            except Exception:
                self.gesture_names = []
    
    def get_model_info(self) -> dict:
        """
        Get information about loaded models.
        """
        return {
            "current_model": self.current_model,
            "model_type": self.model_type,
            "available_models": list(self.loaded_models.keys())
        }
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """
        Get the file path of a specific model.
        """
        if model_name == "built_in":
            tflite_path = self.model_dir / "built_in.tflite"
            onnx_path = self.model_dir / "built_in.onnx"
            return tflite_path if tflite_path.exists() else (onnx_path if onnx_path.exists() else None)
        elif model_name.startswith("custom/"):
            return self.model_dir / "custom" / model_name
        else:
            return None


if __name__ == "__main__":
    # Example usage
    manager = ModelManager()
    print(manager.get_model_info())
