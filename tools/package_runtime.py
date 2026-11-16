"""Standard-library wheel/sdist builder and PEP 517 build backend.

Packages only the runtime (`harness`) and the optional database adapters (`stores`),
with no build backend, downloads, application packages or runtime dependencies.
pyproject.toml selects this module as the in-tree build backend, so pip and uv can
install the project from a git URL or a source checkout.

Run: python3 -m tools.package_runtime --output DIRECTORY
"""

import argparse
import base64
import csv
import gzip
import hashlib
import io
import tarfile
import tomllib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
NAME, VERSION = "little_harness", PROJECT["project"]["version"]
PACKAGES = ("harness", "stores")
DIST_INFO = f"{NAME}-{VERSION}.dist-info"
FILENAME = f"{NAME}-{VERSION}-py3-none-any.whl"
SDIST = f"{NAME}-{VERSION}"
# Fixed timestamp so identical sources build identical archives.
EPOCH = (2020, 1, 1, 0, 0, 0)
EPOCH_SECONDS = 1577836800


def _metadata():
    """Core metadata shared by the wheel METADATA and the sdist PKG-INFO."""
    project = PROJECT["project"]
    lines = ["Metadata-Version: 2.4", "Name: little-harness", f"Version: {VERSION}",
             f"Summary: {project['description']}", f"License-Expression: {project['license']}",
             "License-File: LICENSE", f"Requires-Python: {project['requires-python']}"]
    for extra, requirements in project.get("optional-dependencies", {}).items():
        lines.append(f"Provides-Extra: {extra}")
        lines.extend(f'Requires-Dist: {requirement}; extra == "{extra}"' for requirement in requirements)
    lines.append("Description-Content-Type: text/markdown")
    readme = (ROOT / project["readme"]).read_text(encoding="utf-8")
    return ("\n".join(lines) + "\n\n" + readme).encode("utf-8")


def _package_files():
    """Map archive names to bytes for every packaged module."""
    files = {}
    for package in PACKAGES:
        source = ROOT / package
        paths = sorted(source.rglob("*.py"))
        if not (source / "__init__.py").is_file() or any(path.is_symlink() for path in paths):
            raise ValueError(f"expected a regular {package} source package")
        files.update({f"{package}/" + path.relative_to(source).as_posix(): path.read_bytes() for path in paths})
    return files


def _write_wheel_archive(output, files):
    """Add dist-info metadata and RECORD, then write the wheel; never overwrite."""
    files = dict(files)
    files[DIST_INFO + "/licenses/LICENSE"] = (ROOT / "LICENSE").read_bytes()
    files[DIST_INFO + "/METADATA"] = _metadata()
    files[DIST_INFO + "/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: little-harness-stdlib\n"
        "Root-Is-Purelib: true\nTag: py3-none-any\n"
    ).encode("utf-8")
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name, data in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
        writer.writerow((name, "sha256=" + digest, len(data)))
    writer.writerow((DIST_INFO + "/RECORD", "", ""))
    files[DIST_INFO + "/RECORD"] = record.getvalue().encode("utf-8")
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = ZipInfo(name, date_time=EPOCH)
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    destination = Path(output) / FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(buffer.getvalue())
    return destination


def write_wheel(output):
    """Build the wheel into output and return its path."""
    return _write_wheel_archive(output, _package_files())


def write_sdist(output):
    """Build a source distribution that rebuilds the same wheel with this backend."""
    names = ["pyproject.toml", "LICENSE", PROJECT["project"]["readme"],
             "tools/__init__.py", "tools/package_runtime.py", *_package_files()]
    destination = Path(output) / f"{SDIST}.tar.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = [("PKG-INFO", _metadata())] + [(name, (ROOT / name).read_bytes()) for name in sorted(set(names))]
    with destination.open("xb") as stream, \
            gzip.GzipFile(fileobj=stream, mode="wb", mtime=EPOCH_SECONDS, filename="") as compressed, \
            tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, data in entries:
            info = tarfile.TarInfo(f"{SDIST}/{name}")
            info.size, info.mtime, info.mode = len(data), EPOCH_SECONDS, 0o644
            archive.addfile(info, io.BytesIO(data))
    return destination


# PEP 517 / PEP 660 hooks. No build requirements: everything is standard library.

def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_sdist(config_settings=None):
    return []


def get_requires_for_build_editable(config_settings=None):
    return []


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return write_wheel(wheel_directory).name


def build_sdist(sdist_directory, config_settings=None):
    return write_sdist(sdist_directory).name


EDITABLE_MODULE = "_little_harness_editable"


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Editable install exposing only the packaged imports from this checkout.

    A .pth line imports a small finder; putting the whole checkout on sys.path would
    also expose labs, examples and evaluations.
    """
    packages = {package: str(ROOT / package) for package in PACKAGES}
    finder = (
        '"""Editable little-harness install: map packaged imports to the source checkout."""\n\n'
        "import importlib.util\nimport sys\nfrom importlib.abc import MetaPathFinder\n\n"
        f"PACKAGES = {packages!r}\n\n\n"
        "class EditableFinder(MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        root = PACKAGES.get(name)\n"
        "        if root is None:\n"
        "            return None\n"
        "        return importlib.util.spec_from_file_location(\n"
        "            name, root + '/__init__.py', submodule_search_locations=[root])\n\n\n"
        "if not any(isinstance(finder, EditableFinder) for finder in sys.meta_path):\n"
        "    sys.meta_path.append(EditableFinder())\n"
    )
    files = {f"{NAME}.pth": f"import {EDITABLE_MODULE}\n".encode("utf-8"),
             f"{EDITABLE_MODULE}.py": finder.encode("utf-8")}
    return _write_wheel_archive(wheel_directory, files).name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True,
                        help="output directory, created if missing; never overwrites a wheel")
    parser.add_argument("--sdist", action="store_true", help="also build the source distribution")
    args = parser.parse_args()
    try:
        print(write_wheel(args.output))
        if args.sdist:
            print(write_sdist(args.output))
    except (ValueError, OSError):
        parser.exit(1, "Could not build; the output directory may already contain this version "
                       "(bump it in pyproject.toml).\n")


if __name__ == "__main__":
    main()
