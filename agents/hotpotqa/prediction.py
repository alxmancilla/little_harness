"""Label-free final response contract shared by applications and the host evaluator."""

import json


def parse_prediction(text):
    """Strict machine-readable final answer; never heuristically extract prose."""
    value = json.loads(text)
    if (not isinstance(value, dict) or set(value) != {"answer", "supporting_facts", "abstain"}
            or not isinstance(value["answer"], str) or len(value["answer"]) > 500
            or type(value["abstain"]) is not bool
            or not isinstance(value["supporting_facts"], list) or len(value["supporting_facts"]) > 100):
        raise ValueError("invalid final answer schema")
    for fact in value["supporting_facts"]:
        if (not isinstance(fact, list) or len(fact) != 2 or not isinstance(fact[0], str)
                or not fact[0] or type(fact[1]) is not int or fact[1] < 0):
            raise ValueError("invalid supporting-fact schema")
    if value["abstain"]:
        if value["answer"] or value["supporting_facts"]:
            raise ValueError("abstention must have an empty answer and evidence list")
    elif not value["answer"].strip():
        raise ValueError("non-abstaining answer must not be empty")
    return value