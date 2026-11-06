"""The optional wheel has no application files, dependencies or build-time packages."""

import base64
import csv
import hashlib
import io
from zipfile import ZipFile

from tests.helpers import REPO, TemporaryDirectoryTestCase
from tools.package_runtime import DIST_INFO, build_wheel


class RuntimeDistributionTests(TemporaryDirectoryTestCase):
    def test_wheel_contains_only_runtime_and_valid_metadata_and_record(self):
        wheel = build_wheel(self.root)
        with ZipFile(wheel) as archive:
            names = set(archive.namelist())
            source_names = {"harness/" + path.relative_to(REPO / "harness").as_posix()
                            for path in (REPO / "harness").rglob("*.py")}
            self.assertEqual(names, source_names | {DIST_INFO + "/" + name for name in ("METADATA", "WHEEL", "RECORD")})
            metadata = archive.read(DIST_INFO + "/METADATA").decode()
            self.assertIn("Requires-Python: >=3.11", metadata)
            self.assertNotIn("Requires-Dist:", metadata)
            rows = list(csv.reader(io.StringIO(archive.read(DIST_INFO + "/RECORD").decode())))
            self.assertEqual({row[0] for row in rows}, names)
            for name, digest, size in rows:
                if name.endswith("/RECORD"):
                    self.assertEqual((digest, size), ("", ""))
                    continue
                data = archive.read(name)
                expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                self.assertEqual(digest, "sha256=" + expected)
                self.assertEqual(int(size), len(data))

    def test_wheel_runs_independently_and_imports_without_application_side_effects(self):
        wheel = build_wheel(self.root)
        child = self.child("""
            import importlib.metadata, importlib.util
            from pathlib import Path
            import harness
            assert importlib.metadata.version('little-harness') == '0.1.0'
            assert importlib.util.find_spec('agents') is None
            assert importlib.util.find_spec('lessons') is None
            assert not list(Path('.').glob('*.jsonl'))
            model = lambda *a, **k: {'content': [{'type': 'text', 'text': 'installed'}]}
            result = harness.Agent(model, 'sys', tools={}, quiet=True).run_result('go')
            assert result.answer == 'installed'
            assert 'go' not in Path('trace.jsonl').read_text()
        """, wheel)
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_build_is_reproducible_and_refuses_to_overwrite(self):
        first = build_wheel(self.root)
        before = first.read_bytes()
        with self.assertRaises(FileExistsError):
            build_wheel(self.root)
        self.assertEqual(first.read_bytes(), before)
        other = self.root / "other"
        other.mkdir()
        self.assertEqual(build_wheel(other).read_bytes(), before)
