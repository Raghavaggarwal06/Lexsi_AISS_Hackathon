"""The contrast and test data released with the paper (IBM/activation-steering docs/demo-data, Apache-2.0).

  alpaca.json             train 700 / test 500 Alpaca instructions
  behavior_refusal.json   100 refusal prefixes ("non_compliant_responses") + 100 compliance prefixes
  condition_harmful.json  train 4,050 (harmful = machine-generated Sorry-Bench-style, harmless = Alpaca) /
                          test 450 (harmful = Sorry-Bench 2024-06, 10 per each of 45 categories)
  condition_multiple.json train 700 / test 500 Alpaca prompts ("base") + 5 paraphrases each: sexual_content,
                          legal_opinion, hate_speech, crime_planning, health_consultation
The files are read from the cloned release (refs/activation-steering), not copied into this repo.
"""
import json
import os
from pathlib import Path

DATA = Path(os.environ.get("CAST_DATA", Path(__file__).resolve().parents[1] / "refs/activation-steering/docs/demo-data"))
CATEGORIES = ["sexual_content", "legal_opinion", "hate_speech", "crime_planning", "health_consultation"]


def _load(name):
    return json.loads((DATA / name).read_text())


def behavior_refusal():
    """Paper Sec 3.2 / quickstart: first 100 Alpaca train instructions x 100 (refusal, compliance) prefix pairs."""
    a, r = _load("alpaca.json"), _load("behavior_refusal.json")
    qs = [x["question"] for x in a["train"][:100]]
    return qs, list(zip(r["non_compliant_responses"][:100], r["compliant_responses"][:100]))


def condition_harmful():
    d = _load("condition_harmful.json")
    return [x["harmful"] for x in d["train"]], [x["harmless"] for x in d["train"]]


def sec4_test():
    """Paper Sec 4: 450 unseen Sorry-Bench (harmful) + 500 unseen Alpaca (harmless)."""
    h = [x["harmful"] for x in _load("condition_harmful.json")["test"]]
    a = [x["question"] for x in _load("alpaca.json")["test"]]
    return h, a


def multiple(split):
    """{category or 'base': [prompts]} for Sec 5."""
    d = _load("condition_multiple.json")[split]
    return {k: [x[k] for x in d] for k in ["base"] + CATEGORIES}
