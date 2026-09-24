import json

import numpy as np
import pytest

import laya_universal as laya
from laya_universal.agent import Agent, resolve_model
from laya_universal.backends import ModelHandle, register_backend
from laya_universal.common import confidence_from_probs


class FakeBackend:
    """Deterministic backend: logits = [0, 1, ..., count-1] per row, act = zeros."""

    def __init__(self):
        self.calls = 0

    @property
    def name(self):
        return "fake"

    def available(self):
        return True

    def load_model(self, model_path, dtype="float16", device=None, compile=False):
        return ModelHandle("fake", str(model_path), {"supports_embed": False})

    def forward(self, handle, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        self.calls += 1
        n, count = input_ids.shape[0], marker_pos.shape[1]
        logits = np.tile(np.arange(count, dtype=np.float32), (n, 1))
        return logits, np.zeros((n, 2), dtype=np.float32)

    def embed_mean_pool(self, handle, input_ids, attention_mask):
        raise NotImplementedError

    @property
    def has_embed(self):
        return False


@pytest.fixture(scope="module")
def backend():
    b = FakeBackend()
    register_backend("fake", b)
    yield b
    from laya_universal.backends import _BACKENDS

    _BACKENDS.pop("fake", None)


def _write_tokenizer(directory):
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[CLS]": 0, "[SEP]": 1, "[PAD]": 2, "MASKTOKEN": 3, "[UNK]": 4}
    for w in (
        "question choice score noul billing technical sales refund urgent "
        "customer the a of to is which team".split()
    ):
        vocab.setdefault(w, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    directory.mkdir(parents=True, exist_ok=True)
    tok.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "pad_token": "[PAD]",
                "mask_token": "MASKTOKEN",
            }
        )
    )


def _write_config(root, temperature=None, extra_agent=None):
    agent_cfg = {
        "encoder": "modernbert",
        "head_layers": 2,
        "max_len": 64,
        "head_max_len": 32,
        "temperature": temperature or [1.0, 1.0, 1.0],
        "temperature_by_options": {},
        "act_costs": {"a": 1.0},
    }
    if extra_agent:
        agent_cfg.update(extra_agent)
    (root / "rl_agent_config.json").write_text(json.dumps(agent_cfg))
    (root / "encoder").mkdir(exist_ok=True)
    (root / "encoder" / "config.json").write_text(
        json.dumps({"model_type": "modernbert", "max_position_embeddings": 128})
    )


@pytest.fixture()
def model_dir(tmp_path):
    root = tmp_path / "ckpt"
    _write_tokenizer(root / "tokenizer")
    _write_config(root)
    return root


QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": ["billing", "technical", "sales"],
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["not urgent", "soon", "critical"],
    },
    "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
}


def _make_agent(model_dir, **kw):
    return Agent(str(model_dir), backend="fake", **kw)


def test_predict_end_to_end(model_dir, backend):
    agent = _make_agent(model_dir)
    assert agent.backend_name == "fake"
    result = agent.predict("I was billed twice. Please refund the duplicate.", QUESTIONS)
    assert result["model"] == "laya-rl-agent"
    assert result["usage"]["output_tokens"] == 0
    assert result["usage"]["input_tokens"] > 0
    assert backend.calls == 1

    dept = result["answers"]["department"]
    assert dept["type"] == "choice"
    assert dept["choice"] == "sales"  # logits [0,1,2] -> argmax is the last label
    assert sum(dept["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    p = np.exp([0.0, 1.0, 2.0])
    p /= p.sum()
    assert dept["confidence"] == round(confidence_from_probs(p, 3), 4)

    urg = result["answers"]["urgency"]
    assert urg["score"] == pytest.approx(1.5752, abs=1e-4)
    probs = np.array([urg["probabilities"][str(i)] for i in range(3)])
    assert probs == pytest.approx([0.0900, 0.2447, 0.6652], abs=1e-3)

    ref = result["answers"]["refund"]
    assert ref["noul"] == pytest.approx(0.7311, abs=1e-4)
    assert ref["confidence"] == pytest.approx(0.7311, abs=1e-4)
    assert ref["action"]["act_probability"] == pytest.approx(0.5)


def test_batch_chunking_gives_identical_answers(model_dir, backend):
    one = _make_agent(model_dir, batch_size=1)
    many = _make_agent(model_dir, batch_size=16)
    state = "customer refund billing urgent"
    before = backend.calls
    r1 = one.predict(state, QUESTIONS)
    chunked_calls = backend.calls - before
    r2 = many.predict(state, QUESTIONS)
    assert chunked_calls == 3  # one forward per question
    assert backend.calls - before - chunked_calls == 1  # single forward for all three
    assert r1["answers"] == r2["answers"]


def test_empty_questions_skip_the_model(model_dir, backend):
    agent = _make_agent(model_dir)
    before = backend.calls
    result = agent.predict("anything", {})
    assert result["answers"] == {}
    assert result["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert backend.calls == before


def test_temperature_clamping_warns(tmp_path, backend):
    root = tmp_path / "hot"
    _write_tokenizer(root / "tokenizer")
    _write_config(
        root,
        temperature=[0.1, 1.0, 1.0],
        extra_agent={"temperature_by_options": {"choice:11+": 0.1006}},
    )
    with pytest.warns(RuntimeWarning, match="clamp"):
        agent = Agent(str(root), backend="fake")
    assert agent.temperature == [0.5, 1.0, 1.0]
    assert agent.temperature_by_options == {"choice:11+": 0.5}
    assert agent.temperature_raw == [0.1, 1.0, 1.0]


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "weird", "instructions": "x"},
        {"type": "choice", "instructions": "x", "criteria": ["a", "a"]},
        {"type": "choice", "instructions": "x", "criteria": [1, 2]},
        {"type": "choice", "instructions": "x", "criteria": {}},
        {"type": "score", "instructions": "x", "criteria": "notalist"},
        {"type": "noul", "instructions": "x", "criteria": "notadict"},
        {"type": "choice", "criteria": "missing instructions"},
        "not a dict",
    ],
)
def test_to_internal_rejects_bad_questions(bad):
    with pytest.raises(ValueError):
        Agent._to_internal(bad)


def test_all_presets_are_valid_questions():
    for preset in (
        laya.triage_questions(),
        laya.email_questions(),
        laya.guard_questions(),
        laya.moderation_questions(),
        laya.router_questions(),
    ):
        for q in preset.values():
            Agent._to_internal(q)


def test_resolve_model_rejects_missing_local_paths():
    with pytest.raises(FileNotFoundError):
        resolve_model("C:\\no\\such\\directory")
    with pytest.raises(FileNotFoundError):
        resolve_model(".\\nope")
    with pytest.raises(FileNotFoundError):
        resolve_model("./nope")
    # POSIX parent directory escape
    with pytest.raises(ValueError, match="subfolder"):
        resolve_model("some/repo", subfolder="../escape")
    # Windows backslash parent directory escape
    with pytest.raises(ValueError, match="subfolder"):
        resolve_model("some/repo", subfolder="..\\escape")
    with pytest.raises(ValueError, match="subfolder"):
        resolve_model("some/repo", subfolder="..\\..\\secret")
    # Windows drive letter escape
    with pytest.raises(ValueError, match="subfolder"):
        resolve_model("some/repo", subfolder="C:\\Windows")
    # Unix root escape
    with pytest.raises(ValueError, match="subfolder"):
        resolve_model("some/repo", subfolder="/etc/passwd")
    # Windows UNC network share escape
    with pytest.raises(ValueError, match="subfolder"):
        resolve_model("some/repo", subfolder="\\\\evil-server\\share")


def test_resolve_model_blocks_traversal_on_existing_dir(model_dir):
    with pytest.raises(ValueError, match="subfolder"):
        resolve_model(str(model_dir), subfolder="..\\escape")
    with pytest.raises(ValueError, match="subfolder"):
        resolve_model(str(model_dir), subfolder="../escape")


def test_agent_validates_kwargs(model_dir):
    with pytest.raises(ValueError, match="dtype"):
        Agent(str(model_dir), backend="fake", dtype="float99")
    with pytest.raises(ValueError, match="device"):
        Agent(str(model_dir), backend="fake", device="tpu")
    with pytest.raises(ValueError, match="batch_size"):
        Agent(str(model_dir), backend="fake", batch_size=0)


def test_router_uses_attached_agent(model_dir, backend):
    agent = _make_agent(model_dir)
    router = laya.Router()
    router.attach("english", agent)
    router.attach("multilingual", agent)
    result = router.predict("I was billed twice", {"department": QUESTIONS["department"]})
    assert result["routing"]["model"] == "english"
    assert result["answers"]["department"]["choice"] == "sales"


def test_embed_fn_from_agent_guard(model_dir, backend):
    from laya_universal.shortlist import embed_fn_from_agent

    with pytest.raises(RuntimeError, match="embed_fn"):
        embed_fn_from_agent(_make_agent(model_dir))


def test_predict_shortlist_passthrough(model_dir, backend):
    from laya_universal.shortlist import predict_shortlist

    def embed_fn(texts):
        raise AssertionError("must not be called when k >= label count")

    res = predict_shortlist(_make_agent(model_dir), "customer refund", QUESTIONS, embed_fn, k=99)
    assert res["shortlist"]["department"]["passthrough"] is True
    assert res["answers"]["department"]["choice"] == "sales"
