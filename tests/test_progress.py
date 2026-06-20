import io
import json
import tempfile
import unittest
from pathlib import Path

from fair_mia.progress import ProgressReporter, format_duration


class ProgressReporterTests(unittest.TestCase):
    def test_progress_writes_percentage_and_eta_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            now = [0.0]
            stream = io.StringIO()
            path = Path(temp_dir) / "progress.json"
            reporter = ProgressReporter(
                label="job",
                total=2,
                path=path,
                min_interval=0,
                stream=stream,
                clock=lambda: now[0],
            )
            reporter.phase("scoring", "starting")
            now[0] = 5.0
            reporter.update(detail="sample 1 attack=loss")

            snapshot = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(snapshot["completed"], 1)
            self.assertEqual(snapshot["total"], 2)
            self.assertEqual(snapshot["percent"], 50.0)
            self.assertEqual(snapshot["eta_seconds"], 5.0)
            self.assertIn("[progress] job", stream.getvalue())
            self.assertIn("ETA 00:05", stream.getvalue())

    def test_finish_marks_progress_complete(self):
        stream = io.StringIO()
        reporter = ProgressReporter(
            label="job",
            total=3,
            min_interval=0,
            stream=stream,
            clock=lambda: 1.0,
        )
        reporter.finish()
        self.assertEqual(reporter.completed, 3)
        self.assertIn("100.00%", stream.getvalue())

    def test_duration_formatting(self):
        self.assertEqual(format_duration(None), "--:--")
        self.assertEqual(format_duration(65), "01:05")
        self.assertEqual(format_duration(3661), "1:01:01")


if __name__ == "__main__":
    unittest.main()
