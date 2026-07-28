import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.hardware_profiler import AdaptiveLoadController, _build_profile


class HardwareProfilerTests(unittest.TestCase):
    @patch("utils.hardware_profiler._cuda_available", return_value=False)
    @patch("utils.hardware_profiler._gpu_info", return_value=("", 0.0))
    @patch("utils.hardware_profiler._ram_gb", return_value=4.0)
    @patch("utils.hardware_profiler.os.cpu_count", return_value=4)
    def test_weak_pc_uses_lightweight_profile(self, *_mocks):
        profile = _build_profile()
        self.assertEqual(profile.tier, "eco")
        self.assertEqual(profile.camera_model, "rules")
        self.assertEqual(profile.camera_width, 640)

    @patch("utils.hardware_profiler._cuda_available", return_value=True)
    @patch(
        "utils.hardware_profiler._gpu_info",
        return_value=("NVIDIA GeForce RTX", 8.0),
    )
    @patch("utils.hardware_profiler._ram_gb", return_value=32.0)
    @patch("utils.hardware_profiler.os.cpu_count", return_value=16)
    def test_powerful_pc_uses_quality_profile(self, *_mocks):
        profile = _build_profile()
        self.assertEqual(profile.tier, "performance")
        self.assertEqual(profile.camera_model, "yolo")
        self.assertEqual(profile.whisper_model, "base")

    def test_adaptive_controller_backs_off_under_load(self):
        controller = AdaptiveLoadController(100, "balanced")
        overloaded = controller.update(8, 300, 10.0)
        self.assertGreater(overloaded, 100)
        recovered = controller.update(30, 40, 14.0)
        self.assertLess(recovered, overloaded)


if __name__ == "__main__":
    unittest.main()
