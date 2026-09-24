# Derived from Laya (Apache-2.0); see NOTICE. Modified for laya-universal.
"""Opt-in embedding shortlist for high-cardinality choice questions.

Backend-agnostic: uses the agent's backend for mean-pooling when available,
or a caller-supplied embed_fn.
"""

import json
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from .common import render_options, serialize_state

DEFAULT_SHORTLIST_K = 20


def shortlist_choice(state, criteria, embed_fn, k=DEFAULT_SHORTLIST_K, *, instructions=None):
    labels, _scores, _passthrough, _n = _rank(state, criteria, embed_fn, k, instructions)
    return labels


def predict_shortlist(
    agent: Any,
    state: Any,
    questions: Dict[str, Dict[str, Any]],
    embed_fn: Callable[[Sequence[str]], Any],
    k: int = DEFAULT_SHORTLIST_K,
    **predict_kwargs: Any,
) -> Dict[str, Any]:
    if not isinstance(questions, dict):
        raise TypeError("questions must be a dict of question id -> definition")
    checked = _check_k(k)
    reduced: Dict[str, Any] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    for qid, qdef in questions.items():
        if not isinstance(qdef, dict) or qdef.get("type") != "choice":
            reduced[qid] = qdef
            continue
        if "criteria" not in qdef:
            raise ValueError("question %r is a choice but has no criteria" % (qid,))
        labels, scores, passthrough, n = _rank(
            state, qdef["criteria"], embed_fn, checked, qdef.get("instructions")
        )
        meta[qid] = {"labels": list(labels), "scores": scores, "k": checked, "n": n, "passthrough": passthrough}
        if passthrough:
            reduced[qid] = qdef
            continue
        updated = dict(qdef)
        updated["criteria"] = _subset_criteria(qdef["criteria"], labels)
        reduced[qid] = updated

    result = _call_predict(agent, state, reduced, **predict_kwargs)
    if not isinstance(result, dict):
        raise TypeError("predict/system_one must return a dict")
    out = dict(result)
    out["shortlist"] = meta
    return out


def embed_fn_from_agent(agent: Any, max_length: int = 512, batch_size: int = 32) -> Callable[[Sequence[str]], np.ndarray]:
    """Mean-pool the checkpoint encoder already loaded on agent.

    Works with any backend that supports embed_mean_pool.
    """
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    tok = agent.tok
    backend = agent._backend
    handle = agent._handle

    meta = getattr(handle, "metadata", None) or {}
    supports = bool(getattr(backend, "has_embed", False)) or bool(meta.get("supports_embed"))
    if not supports:
        raise RuntimeError(
            f"Backend {backend.name!r} cannot embed with this checkpoint "
            "(no encoder.onnx next to the ONNX model). Pass a dedicated embed_fn "
            "to predict_shortlist, or re-export with laya-universal convert."
        )

    def embed_fn(texts: Sequence[str]) -> np.ndarray:
        rows = ["" if text is None else str(text) for text in texts]
        if not rows:
            return np.zeros((0, 1), dtype=np.float32)
        parts: List[np.ndarray] = []
        for start in range(0, len(rows), batch_size):
            chunk = rows[start: start + batch_size]
            encoded = [
                tok.backend.encode(text, add_special_tokens=True).ids[:max_length] for text in chunk
            ]
            length = max(1, max(len(ids) for ids in encoded))
            input_ids = np.full((len(encoded), length), tok.pad_token_id, dtype=np.int32)
            attention_mask = np.zeros((len(encoded), length), dtype=np.bool_)
            for i, ids in enumerate(encoded):
                input_ids[i, :len(ids)] = ids
                attention_mask[i, :len(ids)] = True
            pooled = backend.embed_mean_pool(handle, input_ids, attention_mask)
            parts.append(pooled)
        return np.concatenate(parts, axis=0)

    return embed_fn


def _rank(state, criteria, embed_fn, k, instructions):
    checked = _check_k(k)
    items = _criteria_items(criteria)
    n = len(items)
    keys = [key for key, _value in items]
    if checked >= n:
        return list(keys), None, True, n
    query = _query_text(state, instructions)
    matrix = _embeddings(embed_fn, [query] + _option_texts(items))
    sims = _cosine(matrix[0], matrix[1:])
    order = np.argsort(-sims, kind="mergesort")[:checked]
    labels = [keys[int(i)] for i in order]
    scores = [float(sims[int(i)]) for i in order]
    return labels, scores, False, n


def _check_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    return k


def _criteria_items(criteria):
    if isinstance(criteria, dict):
        items = list(criteria.items())
    elif isinstance(criteria, list):
        items = [(item, None) for item in criteria]
    else:
        raise TypeError("choice criteria must be a dict or list")
    if not items:
        raise ValueError("choice criteria must contain at least one option")
    seen = set()
    for key, _value in items:
        if key in seen:
            raise ValueError("choice label %r is duplicated" % (key,))
        seen.add(key)
    return items


def _option_texts(items) -> List[str]:
    crit = {key: value for key, value in items}
    rendered = render_options({"t": "choice", "ins": "", "crit": crit})
    texts = [piece if isinstance(piece, str) else str(piece) for piece in rendered]
    if len(texts) != len(items):
        raise ValueError("could not render every choice option")
    return texts


def _query_text(state, instructions) -> str:
    body = serialize_state(state)
    if instructions is None or instructions == "":
        return body
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False)
    return "%s\n%s" % (instructions, body)


def _subset_criteria(criteria, labels):
    if isinstance(criteria, dict):
        return {label: criteria[label] for label in labels}
    return list(labels)


def _embeddings(embed_fn, texts: Sequence[str]) -> np.ndarray:
    if not callable(embed_fn):
        raise TypeError("embed_fn must be callable")
    raw = embed_fn(list(texts))
    if hasattr(raw, "detach"):
        raw = raw.detach().float().cpu().numpy()
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != len(texts) or arr.shape[1] < 1:
        raise ValueError("embed_fn must return array of shape (%d, dim), got %s" % (len(texts), tuple(arr.shape)))
    return np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)


def _cosine(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    qn = float(np.linalg.norm(query))
    dn = np.linalg.norm(docs, axis=1)
    sims = np.zeros(docs.shape[0], dtype=np.float64)
    if qn == 0.0:
        return sims
    denom = dn * qn
    ok = denom > 0.0
    if np.any(ok):
        sims[ok] = docs[ok] @ query / denom[ok]
    return sims


def _call_predict(agent, state, questions, **predict_kwargs):
    fn = getattr(agent, "predict", None)
    if fn is None:
        fn = getattr(agent, "system_one", None)
    if fn is None:
        raise TypeError("agent must provide predict or system_one")
    return fn(state, questions, **predict_kwargs)
