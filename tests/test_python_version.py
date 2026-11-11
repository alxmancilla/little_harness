"""Unsupported Python versions get a clear message instead of a confusing syntax error."""

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run_as(version, statement):
    code = f"import sys; sys.version_info = {version!r}; {statement}"
    return subprocess.run([sys.executable, "-B", "-c", code], cwd=REPO, capture_output=True, text=True,
                          env={"PATH": os.defpath, "PYTHONPATH": str(REPO)}, timeout=20)


class PythonVersionTests(unittest.TestCase):
    def test_harness_import_explains_the_required_version(self):
        child = run_as((3, 10, 14), "import harness")
        self.assertNotEqual(child.returncode, 0)
        self.assertIn("ImportError: Little Harness needs Python 3.11 or newer", child.stderr)

    def test_lessons_exit_with_the_required_version(self):
        child = run_as((3, 9, 6), "import lessons")
        self.assertEqual(child.returncode, 1)
        self.assertIn("The labs need Python 3.11 or newer", child.stderr)
        self.assertNotIn("Traceback", child.stderr)

    def test_supported_versions_import_normally(self):
        child = run_as((3, 11, 0), "import harness, lessons")
        self.assertEqual(child.returncode, 0, child.stderr)


if __name__ == "__main__":
    unittest.main()
