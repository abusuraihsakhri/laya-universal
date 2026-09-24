import pytest

from laya_universal.router import (
    Router,
    match_typed_decisions_workflow,
    normalise_name,
)

Q = {
    "department": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "refunds", "tech": "bugs"},
    }
}


def test_explicit_model_wins_over_detection():
    d = Router().route("发票重复扣款", Q, model="english")
    assert d["model"] == "english"
    assert "explicit" in d["reason"]


def test_explicit_task():
    d = Router().route("hello", Q, task="typed_decisions")
    assert d["model"] == "typed-decisions"


def test_explicit_lang():
    d = Router().route("hello", Q, lang="de")
    assert d["model"] == "multilingual"


def test_script_detection_routing():
    assert Router().route("发票重复扣款", Q)["model"] == "multilingual"
    assert Router().route("Please refund my invoice", Q)["model"] == "english"


def test_unknown_script_uses_default():
    assert Router().route("1234", Q)["model"] == "english"
    assert Router(default="multilingual").route("1234", Q)["model"] == "multilingual"


def test_workflow_detection_is_opt_in():
    qs = {k: {"type": "noul", "instructions": k}
          for k in {"action", "needs_review", "outcome", "risk", "urgency"}}
    assert Router().route("hello", qs)["model"] != "typed-decisions"
    dec = Router(auto_task_detection=True).route("hello", qs)
    assert dec["model"] == "typed-decisions"
    assert isinstance(dec["repo"], str)
    assert "typed-decisions" in dec["repo"]


def test_normalise_name_aliases():
    assert normalise_name("en") == "english"
    assert normalise_name("ML") == "multilingual"
    with pytest.raises(ValueError):
        normalise_name("bogus")


def test_route_rejects_unknown_model():
    with pytest.raises(ValueError):
        Router().route("x", Q, model="bogus")


def test_workflow_match_requires_exact_id_set():
    sig = {"urgency", "action", "needs_review", "outcome", "risk"}
    assert match_typed_decisions_workflow(sig) == "agent_trace_observability"
    assert match_typed_decisions_workflow({"urgency"}) is None
