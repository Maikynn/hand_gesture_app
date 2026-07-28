#!/usr/bin/env python3
"""
Training script for the Hand Gesture Recognition model.
Trains a CNN on hand crop images to recognize 40+ gestures.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from typing import Tuple

# Gesture classes (40 gestures + unknown)
GESTURE_CLASSES = [
    "fist", "open_palm", "point", "victory", "thumbs_up",
    "thumbs_down", "ok_sign", "rock", "peace", "call_me",
    "stop", "come_here", "gun", "three", "four",
    "five", "six", "seven", "eight", "nine",
    "ten", "love", "horns", "shaka", "cross",
    "pray", "wave", "pinch", "snap", "point_left",
    "point_right", "point_up", "point_down", "fist_vertical", "open_vertical",
    "c_hand", "l_hand", "y_hand", "w_hand", "x_hand",
    "unknown"
]

def create_model(input_shape: Tuple[int, int, int] = (150, 150, 3), 
                 num_classes: int = len(GESTURE_CLASSES)) -> keras.Model:
    """
    Create a CNN model for gesture classification.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from directory structure.
    Expected: data_dir/gesture_name/*.jpg
    """
    images = []
    labels = []
    
    for idx, gesture in enumerate(GESTURE_CLASSES):
        gesture_dir = Path(data_dir) / gesture
        if not gesture_dir.exists():
            continue
            
        for img_path in gesture_dir.glob("*.jpg"):
            img = keras.preprocessing.image.load_img(
                img_path, target_size=(150, 150)
            )
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0
            
            images.append(img_array)
            labels.append(idx)
            
    return np.array(images), np.array(labels)

def train(data_dir: str, model_path: str = "models/built_in.tflite", 
          epochs: int = 50, batch_size: int = 32):
    """
    Train the gesture recognition model.
    """
    print("Loading dataset...")
    X, y = load_dataset(data_dir)
    
    if len(X) == 0:
        print("No data found! Please collect data first.")
        return
        
    print(f"Loaded {len(X)} samples")
    
    # Convert labels to categorical
    y_cat = keras.utils.to_categorical(y, num_classes=len(GESTURE_CLASSES))
    
    # Split into train/validation
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y_cat[:split], y_cat[split:]
    
    # Create model
    model = create_model()
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            "models/best_model.h5", save_best_only=True
        )
    ]
    
    # Train
    print("Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Convert to TFLite
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save
    os.makedirs("models", exist_ok=True)
    with open(model_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"Model saved to {model_path}")
    
    # Save gesture classes
    with open("models/gesture_classes.txt", "w") as f:
        f.write("\n".join(GESTURE_CLASSES))
        
    print("Training complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train gesture recognition model")
    parser.add_argument("--data_dir", default="datasets/asl_alphabet", 
                       help="Directory containing training data")
    parser.add_argument("--model_path", default="models/built_in.tflite",
                       help="Output model path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    
    train(args.data_dir, args.model_path, args.epochs, args.batch_size)