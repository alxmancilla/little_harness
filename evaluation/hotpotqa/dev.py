"""Full HotpotQA development preparation, provenance and bounded question loading.

Only the host reads source/gold. The agent receives one question and its candidate
paragraphs, never the labels or the rest of the development split. Stdlib only.
"""

import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path

from .prepare import LICENSE_URL, MAX_INPUT_BYTES, build_example, json_bytes

DEV_SOURCE_URL = (
    "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/"
    "7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json"
)
DEV_SOURCE_SHA256 = "e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8"
DEV_COUNT = 7405
DEFAULT_DEV_OUTPUT = Path(__file__).resolve().parents[2] / "datasets" / "hotpotqa-dev"


def read_bounded(stream):
    data = stream.read(MAX_INPUT_BYTES + 1)
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError("input exceeds 64 MiB")
    return data


def validate_records(records):
    if not isinstance(records, list) or len(records) != DEV_COUNT:
        raise ValueError(f"expected the complete {DEV_COUNT}-question development split")
    seen = set()
    for record in records:
        identifier = record["_id"]
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("question IDs must be nonempty and unique")
        if record["type"] not in ("bridge", "comparison"):
            raise ValueError("unexpected question type")
        seen.add(identifier)


def prepare_dev(output=DEFAULT_DEV_OUTPUT, input_path=None):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("destination already exists; choose a new --output directory")
    if input_path is None:
        with urllib.request.urlopen(DEV_SOURCE_URL, timeout=60) as response:
            data = read_bounded(response)
    else:
        with Path(input_path).open("rb") as source:
            data = read_bounded(source)
    digest = hashlib.sha256(data).hexdigest()
    if input_path is None and digest != DEV_SOURCE_SHA256:
        raise ValueError("download does not match the pinned source checksum")
    records = json.loads(data)
    validate_records(records)
    questions, answers, issues, counts = [], [], [], Counter()
    for number, record in enumerate(records, 1):
        _, question, answer, _, warnings = build_example(record, number, full_dev=True)
        questions.append(question)
        answers.append(answer)
        issues.extend(warnings)
        counts[len(record["context"])] += 1
    manifest = {
        "dataset": "HotpotQA", "config": "distractor", "split": "validation",
        "homepage": "https://hotpotqa.github.io/", "license": "CC BY-SA 4.0",
        "license_url": LICENSE_URL, "source_url": DEV_SOURCE_URL if input_path is None else None,
        "pinned_mirror_url": DEV_SOURCE_URL, "matches_pinned_source": digest == DEV_SOURCE_SHA256,
        "snapshot_sha256": digest, "source_bytes": len(data), "question_count": len(records),
        "document_count": sum(size * count for size, count in counts.items()),
        "candidate_count_distribution": dict(sorted(counts.items())),
        "question_types": dict(Counter(record["type"] for record in records)),
        "selection": "all development questions in original source order",
        "label_issues": issues,
    }
    files = {"source.json": data, "questions.json": json_bytes(questions),
             "answers.json": json_bytes(answers), "manifest.json": json_bytes(manifest)}
    output.mkdir(parents=True, exist_ok=False)  # no writes until every record has passed validation
    for name, content in files.items():
        with (output / name).open("xb") as target:
            target.write(content)
    return {"questions": len(records), "documents": manifest["document_count"],
            "bytes": sum(map(len, files.values())), "label_issues": len(issues)}


def load_records(dataset=DEFAULT_DEV_OUTPUT):
    """Host/evaluator only: verify once and return the manifest and original records."""
    dataset = Path(dataset)
    with (dataset / "manifest.json").open("rb") as source:
        manifest = json.loads(read_bounded(source))
    with (dataset / "source.json").open("rb") as source:
        data = read_bounded(source)
    if (manifest["snapshot_sha256"] != hashlib.sha256(data).hexdigest()
            or manifest["question_count"] != DEV_COUNT
            or manifest["config"] != "distractor" or manifest["split"] != "validation"):
        raise ValueError("dataset integrity check failed; prepare it again in a new directory")
    records = json.loads(data)
    validate_records(records)
    return manifest, records


def load_question(dataset=DEFAULT_DEV_OUTPUT, number=1, question_id=None):
    """Verify the snapshot, then return only one label-free question and its files."""
    _, records = load_records(dataset)
    if question_id is not None:
        number = next((i for i, record in enumerate(records, 1) if record["_id"] == question_id), 0)
    if type(number) is not int or not 1 <= number <= len(records):
        raise ValueError("question not found; use a valid 1-based number or question ID")
    files, question, _, _, _ = build_example(records[number - 1], number, full_dev=True)
    return number, question, files