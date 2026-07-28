# Models Directory

This directory contains gesture recognition models for the Hand Gesture Application.

## Built-in Model
- `built_in.tflite` or `built_in.onnx` - The default gesture recognition model
- If not present, the app falls back to **rule-based gesture recognition** using MediaPipe hand landmarks

## YOLO HaGRID
- `yolo/hagrid_best.pt` — bundled 20-class YOLO gesture classifier
- Select **YOLO HaGRID · нейросеть** on the Camera page
- A different compatible `.pt` model can be selected with the file button
- `setup_venv.bat` installs the CPU-only PyTorch/Ultralytics runtime

## Custom Models
- Place custom models in the `custom/` subdirectory
- Supported formats: `.tflite`, `.onnx`
- Load from **Camera → Model → Custom ONNX/TFLite**

## Gesture Classes
- `gesture_classes.txt` - List of 40 gesture names + "unknown" class
- Order matches the model output indices

## Training
See `train_model.py` in the project root for training a custom model.
The training script expects data in `datasets/gesture_name/*.jpg` format.

## Supported Gestures (40 + unknown)
fist, open_palm, point, victory, thumbs_up, thumbs_down, ok_sign, rock,
peace, call_me, stop, come_here, gun, three, four, five, six, seven,
eight, nine, ten, love, horns, shaka, cross, pray, wave, pinch, snap,
point_left, point_right, point_up, point_down, fist_vertical,
open_vertical, c_hand, l_hand, y_hand, w_hand, x_hand, unknown
