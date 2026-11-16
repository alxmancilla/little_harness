"""The wheel has only runtime/store files, no required dependencies or build-time packages."""

import base64
import csv
import hashlib
import io
import subprocess
import sys
import tarfile
import tomllib
from zipfile import ZipFile

from tests.helpers import REPO, TemporaryDirectoryTestCase
from tools import package_runtime
from tools.package_runtime import DIST_INFO, VERSION, write_sdist, write_wheel


def source_names(*packages):
    return {f"{package}/" + path.relative_to(REPO / package).as_posix()
            for package in packages for path in (REPO / package).rglob("*.py")}


class RuntimeDistributionTests(TemporaryDirectoryTestCase):
    def test_wheel_contains_runtime_and_stores_with_valid_metadata_and_record(self):
        wheel = write_wheel(self.root)
        with ZipFile(wheel) as archive:
            names = set(archive.namelist())
            self.assertEqual(names, source_names("harness", "stores") | {DIST_INFO + "/" + name for name in
                                                                         ("METADATA", "WHEEL", "RECORD",
                                                                          "licenses/LICENSE")})
            self.assertEqual(archive.read(DIST_INFO + "/licenses/LICENSE"), (REPO / "LICENSE").read_bytes())
            metadata = archive.read(DIST_INFO + "/METADATA").decode()
            self.assertIn("Requires-Python: >=3.11", metadata)
            self.assertIn("License-Expression: MIT", metadata)
            # Database drivers are optional extras; nothing is required unconditionally.
            extras = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"]
            for extra, requirements in extras.items():
                self.assertIn(f"Provides-Extra: {extra}\n", metadata)
                for requirement in requirements:
                    self.assertIn(f'Requires-Dist: {requirement}; extra == "{extra}"\n', metadata)
            requires = [line for line in metadata.splitlines() if line.startswith("Requires-Dist:")]
            self.assertTrue(requires)
            self.assertTrue(all("; extra ==" in line for line in requires))
            self.assertIn("Description-Content-Type: text/markdown", metadata)
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
        wheel = write_wheel(self.root)
        child = self.child("""
            import importlib.metadata, importlib.util
            from pathlib import Path
            import harness
            import stores.memory, stores.mongodb, stores.postgres  # drivers stay optional
            assert importlib.metadata.version('little-harness') == '""" + VERSION + """'
            assert importlib.util.find_spec('agents') is None
            assert importlib.util.find_spec('lessons') is None
            assert importlib.util.find_spec('tools') is None
            assert not list(Path('.').glob('*.jsonl'))
            model = lambda *a, **k: {'content': [{'type': 'text', 'text': 'installed'}]}
            result = harness.Agent(model, 'sys', tools={}, quiet=True).run_result('go')
            assert result.answer == 'installed'
            assert 'go' not in Path('trace.jsonl').read_text()
        """, wheel)
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_build_is_reproducible_and_refuses_to_overwrite(self):
        for write in (write_wheel, write_sdist):
            with self.subTest(write=write.__name__):
                first = write(self.root)
                before = first.read_bytes()
                with self.assertRaises(FileExistsError):
                    write(self.root)
                self.assertEqual(first.read_bytes(), before)
                other = self.root / ("other-" + write.__name__)
                other.mkdir()
                self.assertEqual(write(other).read_bytes(), before)

    def test_build_creates_a_missing_output_directory(self):
        wheel = write_wheel(self.root / "new" / "dist")
        self.assertTrue(wheel.is_file())

    def test_sdist_rebuilds_the_same_wheel(self):
        sdist = write_sdist(self.root)
        with tarfile.open(sdist) as archive:
            names = {name.split("/", 1)[1] for name in archive.getnames()}
            self.assertEqual(names, source_names("harness", "stores") | {
                "PKG-INFO", "pyproject.toml", "LICENSE", "README.md", "tools/__init__.py", "tools/package_runtime.py"})
            archive.extractall(self.root / "unpacked", filter="data")
        source = self.root / "unpacked" / f"little_harness-{VERSION}"
        output = self.root / "from-sdist"
        child = subprocess.run([sys.executable, "-B", "-m", "tools.package_runtime", "--output", str(output)],
                               cwd=source, capture_output=True, text=True, timeout=30)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual((output / package_runtime.FILENAME).read_bytes(),
                         write_wheel(self.root / "direct").read_bytes())

    def test_pep517_hooks_return_basenames_and_need_no_requirements(self):
        project = tomllib.loads((REPO / "pyproject.toml").read_text())["build-system"]
        self.assertEqual(project, {"requires": [], "build-backend": "tools.package_runtime", "backend-path": ["."]})
        for hook in ("get_requires_for_build_wheel", "get_requires_for_build_sdist",
                     "get_requires_for_build_editable"):
            self.assertEqual(getattr(package_runtime, hook)({}), [])
        self.assertEqual(package_runtime.build_wheel(str(self.root / "w")), f"little_harness-{VERSION}-py3-none-any.whl")
        self.assertEqual(package_runtime.build_sdist(str(self.root / "s")), f"little_harness-{VERSION}.tar.gz")
        editable = package_runtime.build_editable(str(self.root / "e"))
        with ZipFile(self.root / "e" / editable) as archive:
            self.assertEqual(archive.read("little_harness.pth").decode(), "import _little_harness_editable\n")
            self.assertFalse(any(name.startswith(("harness/", "stores/")) for name in archive.namelist()))
            archive.extract("_little_harness_editable.py", self.root / "site")
        # Only the packaged imports resolve to the checkout; applications and labs stay hidden.
        child = self.child(f"""
            import importlib.util
            import _little_harness_editable
            import harness, stores.postgres
            assert harness.__file__ == {str(REPO / "harness" / "__init__.py")!r}
            for name in ("agents", "lessons", "evaluation", "examples", "tools"):
                assert importlib.util.find_spec(name) is None, name
        """, self.root / "site")
        self.assertEqual(child.returncode, 0, child.stderr)
