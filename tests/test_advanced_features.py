import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assistant.actions import ActionExecutor
from assistant.memory_store import MemoryStore
from camera.advanced_features import (
    CustomGestureLibrary,
    GestureHistory,
    GestureSequenceMatcher,
)


class _Point:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class _Hand:
    def __init__(self, shift=0.0):
        self.landmark = [
            _Point(index * 0.012 + shift, (index % 5) * 0.018, index * 0.001)
            for index in range(21)
        ]


class AdvancedFeatureTests(unittest.TestCase):
    def test_custom_gesture_can_be_recorded_and_recognized(self):
        with tempfile.TemporaryDirectory() as directory:
            library = CustomGestureLibrary(Path(directory) / "custom.json")
            from camera.advanced_features import normalized_landmarks

            samples = [
                normalized_landmarks(_Hand(shift=index * 0.001))
                for index in range(12)
            ]
            library.train("мой жест", samples)
            result = library.predict(_Hand())
            self.assertIsNotNone(result)
            self.assertEqual(result[0], "мой жест")
            self.assertGreater(result[1], 0.9)

    def test_sequence_matches_transitions_inside_timeout(self):
        matcher = GestureSequenceMatcher(
            [
                {
                    "id": "demo",
                    "steps": ["fist", "palm", "two_up"],
                    "timeout_sec": 2,
                    "enabled": True,
                }
            ]
        )
        self.assertIsNone(matcher.feed("fist", 1.0))
        self.assertIsNone(matcher.feed("palm", 1.5))
        self.assertEqual(matcher.feed("two_up", 2.0)["id"], "demo")

    def test_history_marks_mistakes_for_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            history = GestureHistory(Path(directory) / "history.json")
            index = history.add("Left", "palm", 0.9, "yolo")
            history.mark_incorrect(index)
            loaded = GestureHistory(history.path)
            self.assertTrue(loaded.items[0].incorrect)

    def test_memory_is_persistent_and_deletable(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = MemoryStore(Path(directory) / "memory.json")
            memory.remember("Любимый профиль — работа")
            self.assertIn("Любимый профиль", memory.prompt_context())
            memory.delete(0)
            self.assertEqual(memory.items, [])

    def test_scenario_and_undo_execute_through_safe_actions(self):
        keys = []
        executor = ActionExecutor(
            hotkey_sender=lambda value: keys.append(value),
            scenarios=[
                {
                    "id": "music",
                    "steps": [
                        {"type": "system", "target": "volume_up"},
                        {"type": "system", "target": "play_pause"},
                    ],
                }
            ],
        )
        executor.execute({"type": "scenario", "target": "music"})
        self.assertEqual(keys, [["volumeup"], ["playpause"]])
        executor.undo_last()
        self.assertEqual(keys[-1], ["playpause"])


if __name__ == "__main__":
    unittest.main()
