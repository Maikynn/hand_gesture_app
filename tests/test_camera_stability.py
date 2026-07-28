import os
import sys
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# This order is intentional and mirrors the application entry point.
# MediaPipe's Windows native runtime must be loaded before Qt5.
from camera.skeleton_renderer import SkeletonRenderer
from hand_processing.gesture_fusion import HAGRID_CLASSES
from hand_processing.static_yolo import DEFAULT_YOLO_MODEL, YOLOStaticModel

from PyQt5.QtWidgets import QApplication, QLabel

from ui.main_window import CameraWidget


class CameraStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_qimage_preview_owns_contiguous_memory(self):
        label = QLabel()
        label.resize(320, 180)
        source = np.random.default_rng(42).integers(
            0, 256, size=(480, 640, 3), dtype=np.uint8
        )
        for index in range(250):
            # Deliberately pass a non-contiguous view. set_image must detach it
            # before the temporary NumPy object leaves scope.
            CameraWidget.set_image(label, source[::2, ::2, :])
            if index % 25 == 0:
                self.app.processEvents()
        self.assertIsNotNone(label.pixmap())

    def test_mediapipe_graph_can_process_repeated_frames(self):
        renderer = SkeletonRenderer()
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        try:
            for _ in range(15):
                output, landmarks, _ = renderer.process_frame(frame.copy())
                self.assertEqual(output.shape, frame.shape)
                self.assertIn("raw_hands", landmarks)
                self.assertTrue(output.flags.writeable)
        finally:
            renderer.release()

    def test_bundled_yolo_model_loads_and_returns_hagrid_probabilities(self):
        model = YOLOStaticModel(DEFAULT_YOLO_MODEL, HAGRID_CLASSES)
        self.assertTrue(model.ensure_loaded())
        probabilities = model.predict_probs(
            np.zeros((224, 224, 3), dtype=np.uint8),
            input_is_rgb=True,
        )
        self.assertIsNotNone(probabilities)
        self.assertEqual(probabilities.shape, (len(HAGRID_CLASSES),))
        self.assertTrue(np.isfinite(probabilities).all())


if __name__ == "__main__":
    unittest.main()
