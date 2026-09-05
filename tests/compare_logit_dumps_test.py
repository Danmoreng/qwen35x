"""Run with Python + NumPy; validates the public evaluator CLI."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare-logit-dumps.py"


class LogitComparisonTest(unittest.TestCase):
    def compare(self, teacher, candidate, positions=1):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("teacher.bin", "candidate.bin")]
            for path, values in zip(paths, (teacher, candidate)):
                data = struct.pack("<8sIIQ", b"Q35LGT1\0", 1, len(values), positions)
                if positions:
                    data += struct.pack("<i", 0) + struct.pack(f"<{len(values)}f", *values)
                path.write_bytes(data)
            return subprocess.run(
                [sys.executable, str(SCRIPT), "--teacher", str(paths[0]), "--candidate", str(paths[1])],
                capture_output=True, text=True)

    def test_identical_finite_logits(self):
        result = self.compare(list(range(16)), list(range(16)))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["kld"]["mean"], 0)

    def test_nonfinite_teacher_and_candidate(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for side in (0, 1):
                with self.subTest(value=value, side=side):
                    values = [list(range(16)), list(range(16))]
                    values[side][3] = value
                    result = self.compare(*values)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Non-finite raw logit", result.stderr)
                    self.assertIn("output position 0, vocabulary index 3", result.stderr)
                    self.assertIn("teacher.bin" if side == 0 else "candidate.bin", result.stderr)
                    self.assertEqual(result.stdout, "")

    def test_empty_dump(self):
        result = self.compare(list(range(16)), list(range(16)), positions=0)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Empty logit dump", result.stderr)


if __name__ == "__main__":
    unittest.main()
