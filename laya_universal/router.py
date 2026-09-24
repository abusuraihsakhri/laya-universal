# Derived from Laya (Apache-2.0); see NOTICE. Modified for laya-universal.
"""Route a request to the Laya checkpoint best suited to it."""

import os
import threading
from typing import Any, Dict, List, Optional, Union

from .lang import analyse

BUNDLE_REPO = "convaiinnovations/laya"
DEFAULT_MODELS = {
    "english": (BUNDLE_REPO, None),
    "multilingual": (BUNDLE_REPO, "multilingual"),
    "typed-decisions": (BUNDLE_REPO, "typed-decisions"),
}

STANDALONE_MODELS = {
    "english": "convaiinnovations/laya",
    "multilingual": "convaiinnovations/laya-multilingual",
    "typed-decisions": "convaiinnovations/laya-typed-decisions",
}

_ALIASES = {
    "en": "english", "laya": "english", "default": "english",
    "multi": "multilingual", "ml": "multilingual", "laya-multilingual": "multilingual",
    "typed": "typed-decisions", "typed_decisions": "typed-decisions",
    "laya-typed-decisions": "typed-decisions", "decisions": "typed-decisions",
}

_TYPED_DECISION_WORKFLOWS = {
    "agent_trace_observability": {"action", "needs_review", "outcome", "risk", "urgency"},
    "customer_service": {"action", "category", "churn_risk", "needs_human", "urgency"},
    "invoice_processing": {"discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"},
    "security_incidents": {"credential_compromise", "disposition", "severity", "true_positive", "urgency"},
}


def _repo_str(spec):
    repo, sub = _split(spec)
    return "%s/%s" % (repo, sub) if sub else repo


def _split(spec):
    if isinstance(spec, (tuple, list)):
        repo, sub = (list(spec) + [None])[:2]
        return repo, sub
    return spec, None


def normalise_name(name: str) -> str:
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in DEFAULT_MODELS:
        raise ValueError(
            "unknown model %r; choose one of %s (or an alias: %s)"
            % (name, sorted(DEFAULT_MODELS), sorted(_ALIASES))
        )
    return key


def match_typed_decisions_workflow(questions):
    ids = set(questions or {})
    for wf, sig in _TYPED_DECISION_WORKFLOWS.items():
        if ids == sig:
            return wf
    return None


class RouteDecision(dict):
    @property
    def model(self) -> str:
        return self["model"]

    @property
    def reason(self) -> str:
        return self["reason"]

    def __repr__(self):
        return "RouteDecision(model=%r, reason=%r)" % (self["model"], self["reason"])


class Router:
    """Lazily loads Laya checkpoints and sends each request to the right one."""

    def __init__(
        self,
        models: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
        token: Optional[str] = None,
        max_loaded: int = 1,
        default: str = "english",
        auto_task_detection: bool = False,
        standalone_repos: bool = False,
        preload: bool = False,
        dtype: str = "float16",
        backend: Optional[str] = None,
    ):
        self.models = dict(STANDALONE_MODELS if standalone_repos else DEFAULT_MODELS)
        if models:
            self.models.update({normalise_name(k): v for k, v in models.items()})
        self.dtype = dtype
        self.device = device
        self.token = token or os.environ.get("HF_TOKEN")
        self.max_loaded = max(1, int(max_loaded))
        self.default = normalise_name(default)
        self.auto_task_detection = bool(auto_task_detection)
        self.backend = backend
        self._agents: Dict[str, Any] = {}
        self._order: List[str] = []
        self._lock = threading.RLock()
        if preload:
            self.preload()

    def load(self, name: str):
        key = normalise_name(name)
        with self._lock:
            if key in self._agents:
                self._touch(key)
                return self._agents[key]
            from .agent import Agent
            repo, sub = _split(self.models[key])
            agent = Agent(
                repo, device=self.device, token=self.token, subfolder=sub,
                dtype=self.dtype, backend=self.backend,
            )
            self._agents[key] = agent
            self._order.append(key)
            self._evict()
            return agent

    def _touch(self, key: str):
        with self._lock:
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)

    def _evict(self):
        with self._lock:
            while len(self._order) > self.max_loaded:
                victim = self._order.pop(0)
                self._agents.pop(victim, None)
            if len(self._order) < len(self._agents):
                for k in list(self._agents):
                    if k not in self._order:
                        self._agents.pop(k, None)

    def attach(self, name: str, agent: Any):
        key = normalise_name(name)
        with self._lock:
            self._agents[key] = agent
            self._touch(key)
            self.max_loaded = max(self.max_loaded, len(self._agents))
        return agent

    def preload(self, names: Optional[List[str]] = None):
        names = [normalise_name(n) for n in (names or list(self.models))]
        with self._lock:
            self.max_loaded = max(self.max_loaded, len(names), len(self._agents))
            for n in names:
                if n not in self._agents:
                    self.load(n)
        return self

    def unload(self, name: Optional[str] = None):
        with self._lock:
            if name is None:
                self._agents.clear()
                self._order.clear()
            else:
                key = normalise_name(name)
                self._agents.pop(key, None)
                if key in self._order:
                    self._order.remove(key)

    @property
    def loaded(self) -> List[str]:
        with self._lock:
            return list(self._order)

    def route(
        self,
        state: Union[str, dict, list, None],
        questions: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> RouteDecision:
        if model is not None:
            key = normalise_name(model)
            return RouteDecision(model=key, repo=_repo_str(self.models[key]),
                                 reason="explicit model=%r" % model, detection=None, workflow=None)
        if task is not None:
            key = normalise_name("typed-decisions" if str(task).lower().replace("-", "_") == "typed_decisions" else task)
            return RouteDecision(model=key, repo=_repo_str(self.models[key]),
                                 reason="explicit task=%r" % task, detection=None, workflow=None)
        workflow = match_typed_decisions_workflow(questions or {})
        if workflow and self.auto_task_detection:
            return RouteDecision(model="typed-decisions", repo=_repo_str(self.models["typed-decisions"]),
                                 reason="question ids match the %r typed-decisions workflow" % workflow,
                                 detection=None, workflow=workflow)
        if lang is not None:
            key = "english" if str(lang).lower().split("-")[0] in ("en", "eng", "english") else "multilingual"
            return RouteDecision(model=key, repo=_repo_str(self.models[key]),
                                 reason="explicit lang=%r" % lang, detection=None, workflow=workflow)
        det = analyse(state)
        if det["script"] == "unknown":
            key = self.default
            reason = "no letters detected; using default (%s)" % key
        elif det["script"] != "latin":
            key = "multilingual"
            reason = "non-Latin script (%s, %.0f%% of letters)" % (det["script"], 100 * float(det["non_latin_fraction"]))
        elif not det["is_english"]:
            key = "multilingual"
            if det["language"]:
                reason = "Latin script but language looks like %r" % det["language"]
            else:
                reason = "Latin script, unidentified language (%.0f%% non-English letters)" % (100 * float(det["diacritic_rate"]))
        else:
            key = "english"
            reason = "English Latin text"
        return RouteDecision(model=key, repo=_repo_str(self.models[key]), reason=reason,
                             detection=det, workflow=workflow)

    def predict(self, state, questions, model=None, task=None, lang=None):
        decision = self.route(state, questions, model=model, task=task, lang=lang)
        agent = self.load(decision["model"])
        result = agent.system_one(state, questions)
        result["routing"] = dict(decision)
        return result

    system_one = predict

    def __repr__(self):
        return "Router(loaded=%s, max_loaded=%d, default=%r)" % (self.loaded, self.max_loaded, self.default)
