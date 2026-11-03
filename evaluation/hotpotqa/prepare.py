"""Prepare HotpotQA exercises with stdlib only; never call an LLM.

Default: fetch a small public preview, checking explicitly for truncated cells.
--full-dev: prepare all 7,405 development questions for isolated per-question runs.
Offline: --input accepts our saved source.json or the original development JSON.
The destination must not exist; no existing corpus or user files are overwritten.
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

SOURCE_URL = ("https://datasets-server.huggingface.co/first-rows?"
              "dataset=hotpotqa/hotpot_qa&config=distractor&split=validation")
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
MAX_INPUT_BYTES = 64 * 1024 * 1024
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "datasets" / "hotpotqa"


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def select_questions(payload):
    """First five of each type, in source order; no answer-dependent selection."""
    if isinstance(payload, dict):                  # Hugging Face preview representation
        rows = payload["rows"]
        if any(item.get("truncated_cells") for item in rows):
            raise ValueError("preview contains truncated cells; use a complete local JSON source")
        records = [item["row"] for item in rows]
    else:
        records = payload
    if not isinstance(records, list):
        raise ValueError("source must contain a list of HotpotQA records")
    selected, seen = [], set()
    counts = {"bridge": 0, "comparison": 0}
    for record in records:
        kind = record.get("type")
        if kind not in counts or counts[kind] >= 5:
            continue
        identifier = record.get("_id", record.get("id"))
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("question IDs must be nonempty and unique")
        context, facts = record["context"], record["supporting_facts"]
        if isinstance(context, dict):
            if len(context["title"]) != len(context["sentences"]):
                raise ValueError("context titles and sentences differ in length")
            context = list(map(list, zip(context["title"], context["sentences"])))
        if isinstance(facts, dict):
            if len(facts["title"]) != len(facts["sent_id"]):
                raise ValueError("supporting-fact titles and sentence IDs differ in length")
            facts = list(map(list, zip(facts["title"], facts["sent_id"])))
        selected.append({"_id": identifier, "question": record["question"],
                         "answer": record["answer"], "type": kind, "level": record["level"],
                         "context": context, "supporting_facts": facts})
        seen.add(identifier)
        counts[kind] += 1
        if len(selected) == 10:
            break
    if len(selected) != 10:
        raise ValueError("source needs at least five bridge and five comparison questions")
    return selected


def build_example(record, number, *, full_dev=False):
    """Render one question without using titles as paths; gold stays separate.

    Full dev preserves short candidate sets and flags invalid original sentence
    labels instead of dropping questions or silently repairing their gold labels.
    """
    if (not isinstance(record["question"], str) or not record["question"].strip()
            or len(record["question"]) > 500 or not isinstance(record["answer"], str)):
        raise ValueError("invalid question/answer or question exceeds the harness's 500-character limit")
    if not isinstance(record["context"], list) or not (
            2 <= len(record["context"]) <= 10 if full_dev else len(record["context"]) == 10):
        raise ValueError("each distractor example must have ten candidate paragraphs (2–10 for full dev)")
    files, documents, candidates, titles, issues = {}, {}, [], {}, []
    for position, (title, sentences) in enumerate(record["context"], 1):
        if (not isinstance(title, str) or not title.strip() or title in titles
                or not isinstance(sentences, list) or not sentences
                or any(not isinstance(sentence, str) for sentence in sentences)):
            raise ValueError("invalid or duplicate paragraph title, or invalid sentences")
        name = f"q{number:02d}-p{position:02d}.txt"
        text = f"# {title}\n\n" + "\n".join(f"[{i}] {sentence}" for i, sentence in enumerate(sentences)) + "\n"
        content = text.encode("utf-8")
        if len(content) > 64 * 1024:
            raise ValueError("paragraph exceeds the harness's 64-KiB document limit")
        files[f"knowledge/{name}"] = content
        documents[name] = {"title": title, "question_id": record["_id"], "sentences": len(sentences)}
        candidates.append(name)
        titles[title] = (name, len(sentences))
    support = []
    for title, sentence in record["supporting_facts"]:
        if title not in titles or type(sentence) is not int or sentence < 0:
            raise ValueError("supporting fact does not refer to a candidate sentence")
        if sentence >= titles[title][1]:
            if not full_dev:
                raise ValueError("supporting fact does not refer to a candidate sentence")
            issues.append({"question_id": record["_id"], "title": title, "sentence": sentence,
                           "sentence_count": titles[title][1], "issue": "sentence_out_of_range"})
        support.append({"document": titles[title][0], "title": title, "sentence": sentence})
    if not support:
        raise ValueError("missing supporting facts")
    question = {"id": record["_id"], "question": record["question"], "candidates": candidates}
    answer = {"id": record["_id"], "answer": record["answer"], "type": record["type"],
              "level": record["level"], "supporting_facts": support}
    return files, question, answer, documents, issues


def build_files(selected, source_url=SOURCE_URL):
    """Keep all candidates; put questions, source snapshots and gold labels outside knowledge/."""
    if len(selected) != 10 or any(sum(row["type"] == kind for row in selected) != 5
                                  for kind in ("bridge", "comparison")):
        raise ValueError("expected exactly five bridge and five comparison questions")
    files, questions, answers, documents = {}, [], [], {}
    for number, record in enumerate(selected, 1):
        paragraphs, question, answer, metadata, _ = build_example(record, number)
        files.update(paragraphs)
        documents.update(metadata)
        questions.append(question)
        answers.append(answer)
    snapshot = json_bytes(selected)
    manifest = {"dataset": "HotpotQA", "config": "distractor", "split": "validation",
                "homepage": "https://hotpotqa.github.io/", "source_url": source_url,
                "license": "CC BY-SA 4.0", "license_url": LICENSE_URL,
                "selection": "first five bridge and first five comparison records, in source order",
                "question_count": len(questions), "document_count": len(documents),
                "snapshot_sha256": hashlib.sha256(snapshot).hexdigest(), "documents": documents}
    files.update({"source.json": snapshot, "questions.json": json_bytes(questions),
                  "answers.json": json_bytes(answers), "manifest.json": json_bytes(manifest)})
    return files


def prepare(output, input_path=None):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("destination already exists; choose a new --output directory")
    if input_path is None:
        with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
            data = response.read(MAX_INPUT_BYTES + 1)
    else:
        with Path(input_path).open("rb") as source:
            data = source.read(MAX_INPUT_BYTES + 1)
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError("input exceeds 64 MiB")
    files = build_files(select_questions(json.loads(data)))
    output.mkdir(parents=True, exist_ok=False)       # all validation precedes filesystem changes
    (output / "knowledge").mkdir()
    for name, content in files.items():
        with (output / name).open("xb") as target:
            target.write(content)
    return {"questions": 10, "documents": len(files) - 4, "bytes": sum(map(len, files.values()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="offline HotpotQA JSON (default: fetch public preview)")
    parser.add_argument("--output", type=Path, help="new dataset directory (default depends on mode)")
    parser.add_argument("--full-dev", action="store_true", help="all 7,405 distractor development questions")
    args = parser.parse_args()
    try:
        if args.full_dev:
            from .dev import DEFAULT_DEV_OUTPUT, prepare_dev
            stats = prepare_dev(args.output or DEFAULT_DEV_OUTPUT, args.input)
        else:
            stats = prepare(args.output or DEFAULT_OUTPUT, args.input)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Preparation failed ({type(error).__name__}); check source, connectivity and destination.\n")
    print(f"Prepared {stats['questions']} questions, {stats['documents']} documents, {stats['bytes']:,} bytes.")
    if args.full_dev:
        print(f"Passages are materialized per question, not as a shared corpus. Label issues: {stats['label_issues']}.")


if __name__ == "__main__":
    main()