"""Build a standard pure-Python wheel using only the standard library.

No build backend, downloads, application packages or runtime dependencies.
Run: python3 -m tools.package_runtime --output /existing/output/directory
"""

import argparse
import base64
import csv
import hashlib
import io
import tomllib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


PROJECT = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
NAME, VERSION = "little_harness", PROJECT["project"]["version"]
DIST_INFO = f"{NAME}-{VERSION}.dist-info"
FILENAME = f"{NAME}-{VERSION}-py3-none-any.whl"


def build_wheel(output):
    """Package only runtime source and standard wheel metadata; never overwrite."""
    runtime = Path(__file__).resolve().parents[1] / "harness"
    paths = sorted(runtime.rglob("*.py"))
    if not (runtime / "__init__.py").is_file() or any(path.is_symlink() for path in paths):
        raise ValueError("expected a regular harness source package")
    files = {"harness/" + path.relative_to(runtime).as_posix(): path.read_bytes() for path in paths}
    files[DIST_INFO + "/METADATA"] = (
        f"Metadata-Version: 2.1\nName: little-harness\nVersion: {VERSION}\n"
        f"Summary: {PROJECT['project']['description']}\n"
        f"Requires-Python: {PROJECT['project']['requires-python']}\n\n"
    ).encode("utf-8")
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
            info = ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    destination = Path(output) / FILENAME
    with destination.open("xb") as stream:
        stream.write(buffer.getvalue())
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="existing output directory; no overwrites")
    args = parser.parse_args()
    try:
        destination = build_wheel(args.output)
    except (ValueError, OSError):
        parser.exit(1, "Could not build wheel; use an existing directory without a wheel of the same name.\n")
    print(destination)


if __name__ == "__main__":
    main()