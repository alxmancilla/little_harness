"""Small synthetic draft cases; expectations are host-only, not model inputs."""

import json
import re
from pathlib import Path

from examples.support_agent import DOCUMENT_NAMES


DEFAULT_CASES = Path(__file__).with_name("cases.json")
DEFAULT_SCENARIOS = Path(__file__).with_name("scenarios.json")


def _valid_turn(task, expected):
    """One agent task ({sku, question}) and its host-only draft expectation."""
    return (type(task) is dict and set(task) == {"sku", "question"}
            and type(task["sku"]) is str and re.fullmatch(r"[A-Z0-9_-]{1,40}", task["sku"])
            and type(task["question"]) is str and 1 <= len(task["question"].strip()) <= 1000
            and type(expected) is dict and set(expected) == {"needs_human", "required_sources"}
            and type(expected["needs_human"]) is bool and type(expected["required_sources"]) is list
            and all(type(source) is str and source in DOCUMENT_NAMES for source in expected["required_sources"])
            and len(expected["required_sources"]) == len(set(expected["required_sources"])))


def _valid_extra_documents(documents):
    """Untrusted notes added to one case's search results; never citable as required evidence."""
    return (type(documents) is dict and 1 <= len(documents) <= 2
            and all(type(name) is str and re.fullmatch(r"[a-z0-9_-]+\.md", name) and name not in DOCUMENT_NAMES
                    and type(text) is str and 1 <= len(text) <= 6000 for name, text in documents.items()))


def load_cases(path=DEFAULT_CASES):
    """Validate the local reviewable case format before any model calls or output writes."""
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    if (type(dataset) is not dict or dataset.get("version") != "support-v0-draft"
            or dataset.get("review_status") != "unreviewed-synthetic"
            or type(dataset.get("cases")) is not list or not 1 <= len(dataset["cases"]) <= 50):
        raise ValueError("invalid support dataset")
    seen = set()
    for case in dataset["cases"]:
        if type(case) is not dict or set(case) - {"extra_documents", "extra_memories"} != {"id", "category", "input", "expected",
                                                                         "review_guide"}:
            raise ValueError("invalid support case fields")
        if "extra_documents" in case and not _valid_extra_documents(case["extra_documents"]):
            raise ValueError("invalid support case extra documents")
        if "extra_memories" in case and (type(case["extra_memories"]) is not list
                or not 1 <= len(case["extra_memories"]) <= 2
                or any(type(text) is not str or not 1 <= len(text) <= 2000 for text in case["extra_memories"])):
            raise ValueError("invalid support case extra memories")
        if (type(case["id"]) is not str or not re.fullmatch(r"c[0-9]{2}", case["id"])
                or case["id"] in seen or type(case["category"]) is not str
                or not re.fullmatch(r"[a-z_]{1,40}", case["category"])
                or type(case["review_guide"]) is not str or not 1 <= len(case["review_guide"]) <= 1000
                or not _valid_turn(case["input"], case["expected"])):
            raise ValueError("invalid support case values")
        seen.add(case["id"])
    return dataset


def load_scenarios(path=DEFAULT_SCENARIOS):
    """Validate multi-turn draft scenarios: 2 to 6 turns, each an ordinary support task."""
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    if (type(dataset) is not dict or dataset.get("version") != "support-multiturn-v0-draft"
            or dataset.get("review_status") != "unreviewed-synthetic"
            or type(dataset.get("scenarios")) is not list or not 1 <= len(dataset["scenarios"]) <= 20):
        raise ValueError("invalid support scenario dataset")
    seen = set()
    for scenario in dataset["scenarios"]:
        if type(scenario) is not dict or set(scenario) != {"id", "category", "turns", "review_guide"}:
            raise ValueError("invalid support scenario fields")
        turns = scenario["turns"]
        if (type(scenario["id"]) is not str or not re.fullmatch(r"m[0-9]{2}", scenario["id"])
                or scenario["id"] in seen or type(scenario["category"]) is not str
                or not re.fullmatch(r"[a-z_]{1,40}", scenario["category"])
                or type(scenario["review_guide"]) is not str or not 1 <= len(scenario["review_guide"]) <= 1000
                or type(turns) is not list or not 2 <= len(turns) <= 6
                or any(type(turn) is not dict or set(turn) != {"input", "expected"}
                       or not _valid_turn(turn["input"], turn["expected"]) for turn in turns)):
            raise ValueError("invalid support scenario values")
        seen.add(scenario["id"])
    return dataset