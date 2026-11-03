"""Small synthetic draft cases; expectations are host-only, not model inputs."""

import json
import re
from pathlib import Path

from examples.support_agent import DOCUMENT_NAMES


DEFAULT_CASES = Path(__file__).with_name("cases.json")


def load_cases(path=DEFAULT_CASES):
    """Validate the local reviewable case format before any model calls or output writes."""
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    if (type(dataset) is not dict or dataset.get("version") != "support-v0-draft"
            or dataset.get("review_status") != "unreviewed-synthetic"
            or type(dataset.get("cases")) is not list or not 1 <= len(dataset["cases"]) <= 50):
        raise ValueError("invalid support dataset")
    seen = set()
    for case in dataset["cases"]:
        if type(case) is not dict or set(case) != {"id", "category", "input", "expected", "review_guide"}:
            raise ValueError("invalid support case fields")
        task, expected = case["input"], case["expected"]
        if (type(case["id"]) is not str or not re.fullmatch(r"c[0-9]{2}", case["id"])
                or case["id"] in seen or type(case["category"]) is not str
                or not re.fullmatch(r"[a-z_]{1,40}", case["category"])
                or type(case["review_guide"]) is not str or not 1 <= len(case["review_guide"]) <= 1000
                or type(task) is not dict or set(task) != {"sku", "question"}
                or type(task["sku"]) is not str or not re.fullmatch(r"[A-Z0-9_-]{1,40}", task["sku"])
                or type(task["question"]) is not str or not 1 <= len(task["question"].strip()) <= 1000
                or type(expected) is not dict or set(expected) != {"needs_human", "required_sources"}
                or type(expected["needs_human"]) is not bool or type(expected["required_sources"]) is not list
                or any(type(source) is not str or source not in DOCUMENT_NAMES
                       for source in expected["required_sources"])
                or len(expected["required_sources"]) != len(set(expected["required_sources"]))):
            raise ValueError("invalid support case values")
        seen.add(case["id"])
    return dataset