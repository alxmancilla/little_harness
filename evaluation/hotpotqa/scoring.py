"""Stdlib HotpotQA metrics, following hotpot_evaluate_v1.py (Apache-2.0).

Reference: https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
Answer normalization, categorical-answer handling, set-based sentence evidence,
and joint precision/recall match that evaluator. Scores are fractions, not percent.
"""

import re
import string
from collections import Counter

from agents.hotpotqa.prediction import parse_prediction as parse_prediction


def normalize_answer(text):
    text = "".join(character for character in text.lower() if character not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def overlap_scores(correct, predicted, expected, exact):
    precision = correct / predicted if predicted else 0.0
    recall = correct / expected if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"em": float(exact), "f1": f1, "prec": precision, "recall": recall}


def answer_scores(prediction, gold):
    prediction, gold = normalize_answer(prediction), normalize_answer(gold)
    predicted, expected = prediction.split(), gold.split()
    common = sum((Counter(predicted) & Counter(expected)).values())
    if prediction != gold and ({prediction, gold} & {"yes", "no", "noanswer"}):
        common = 0
    return overlap_scores(common, len(predicted), len(expected), prediction == gold)


def supporting_scores(prediction, gold):
    predicted, expected = set(map(tuple, prediction)), set(map(tuple, gold))
    return overlap_scores(len(predicted & expected), len(predicted), len(expected), predicted == expected)


def score_prediction(prediction, record):
    """Gold is used here in the host only, never in the agent's tools or messages."""
    result = answer_scores(prediction["answer"], record["answer"])
    candidates = {title: len(sentences) for title, sentences in record["context"]}
    invalid_gold = any(title not in candidates or not 0 <= sentence < candidates[title]
                       for title, sentence in record["supporting_facts"])
    if invalid_gold:
        # Preserve source labels, but do not score an impossible sentence reference.
        result.update({f"{prefix}{name}": None for prefix in ("sp_", "joint_")
                       for name in ("em", "f1", "prec", "recall")})
        return result
    evidence = supporting_scores(prediction["supporting_facts"], record["supporting_facts"])
    result.update({"sp_" + name: value for name, value in evidence.items()})
    precision, recall = result["prec"] * evidence["prec"], result["recall"] * evidence["recall"]
    result.update({"joint_em": result["em"] * evidence["em"], "joint_prec": precision,
                   "joint_recall": recall,
                   "joint_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0})
    return result