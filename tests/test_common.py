import numpy as np
import pytest

from laya_universal.common import (
    TEMP_MAX,
    TEMP_MIN,
    build_prefix,
    build_sequence,
    clamp_temperature,
    collate_items,
    confidence_from_probs,
    render_options,
    serialize_state,
    temp_bucket,
)


class FakeTok:
    cls_token_id = 0
    sep_token_id = 1
    pad_token_id = 2
    mask_token_id = 3
    mask_token = "MASK"

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=False):
        ids = [self.vocab.setdefault(w, 10 + len(self.vocab)) for w in text.split()]
        return {"input_ids": ids}


def q_choice():
    return {"t": "choice", "ins": "Which team?", "crit": {"billing": "refunds", "tech": "bugs"}}


def test_serialize_state():
    assert serialize_state("hi") == "hi"
    assert serialize_state({"a": 1}) == '{"a": 1}'


def test_render_options():
    assert render_options(q_choice()) == ["billing: refunds", "tech: bugs"]
    assert render_options({"t": "choice", "ins": "", "crit": {"a": None, "b": ""}}) == ["a", "b"]
    assert render_options({"t": "choice", "ins": "", "crit": {"a": {"k": 1}}})[0].startswith("a: ")
    assert render_options({"t": "score", "ins": "", "crit": ["low", "high"]}) == [
        "level 0: low",
        "level 1: high",
    ]
    assert render_options({"t": "noul", "ins": "", "crit": {}}) == [
        "false: no, the statement does not hold",
        "true: yes, the statement holds",
    ]
    assert render_options({"t": "noul", "ins": "", "crit": {"false": "nope", "true": "yep"}}) == [
        "false: nope",
        "true: yep",
    ]


def test_build_prefix_structure():
    tok = FakeTok()
    ids, markers = build_prefix(tok, q_choice(), head_max_len=64)
    assert ids[0] == tok.cls_token_id
    assert tok.sep_token_id in ids
    assert all(ids[m] == tok.mask_token_id for m in markers)
    assert markers == sorted(markers)
    assert len(markers) == 2


def test_build_sequence_truncates_state_and_keeps_markers():
    tok = FakeTok()
    ids, markers = build_sequence(tok, " ".join(["word"] * 500), q_choice(), 64, 32)
    assert len(ids) <= 64
    assert ids[-1] == tok.sep_token_id
    assert all(m < len(ids) for m in markers)
    assert len(markers) == 2


def test_build_sequence_many_options_fit_token_budget():
    tok = FakeTok()
    crit = {f"opt{i}": None for i in range(30)}
    ids, markers = build_sequence(
        tok, "some state", {"t": "choice", "ins": "pick", "crit": crit}, 256, 64
    )
    assert len(ids) <= 256
    assert len(markers) == 30
    assert all(ids[m] == tok.mask_token_id for m in markers)


def test_collate_items():
    items = [
        {"ids": [0, 5, 1], "markers": [2, 4], "qtype": 0},
        {"ids": [0, 7, 7, 1], "markers": [2], "qtype": 1},
    ]
    b = collate_items(items, pad_id=2)
    assert b["input_ids"].shape == (2, 4)
    assert b["input_ids"].tolist() == [[0, 5, 1, 2], [0, 7, 7, 1]]
    assert b["attention_mask"].tolist() == [
        [True, True, True, False],
        [True, True, True, True],
    ]
    assert b["marker_pos"].shape == (2, 2)  # padded to at least two marker slots
    assert b["marker_mask"].tolist() == [[True, True], [True, False]]
    assert b["qtype"].tolist() == [0, 1]


def test_collate_pad_to_multiple_and_max_length():
    items = [{"ids": list(range(7)), "markers": [3], "qtype": 2}]
    b = collate_items(items, pad_id=0, pad_to_multiple=16, max_length=64)
    assert b["input_ids"].shape[1] == 16
    b = collate_items(items, pad_id=0, pad_to_multiple=16, max_length=8)
    assert b["input_ids"].shape[1] == 8


def test_collate_empty_raises():
    with pytest.raises(ValueError):
        collate_items([], pad_id=0)


def test_clamp_temperature():
    assert clamp_temperature(0.01) == TEMP_MIN
    assert clamp_temperature(99.0) == TEMP_MAX
    assert clamp_temperature(float("nan")) == 1.0
    assert clamp_temperature("nope") == 1.0
    assert clamp_temperature(2.0) == 2.0


def test_temp_bucket():
    assert temp_bucket(0, 2) == "choice:2"
    assert temp_bucket(1, 5) == "score:3-5"
    assert temp_bucket(2, 10) == "noul:6-10"
    assert temp_bucket(0, 11) == "choice:11+"


def test_confidence_from_probs():
    assert confidence_from_probs(np.array([1.0]), 1) == 1.0
    assert confidence_from_probs(np.array([0.5, 0.5]), 2) == pytest.approx(0.0, abs=1e-6)
    assert confidence_from_probs(np.array([0.0, 1.0]), 2) == pytest.approx(1.0)
