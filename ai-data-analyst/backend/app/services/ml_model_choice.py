"""Pick which of the owner's saved models a chat request means ("score this with my churn model").

Code decides, never the language model: the planner may suggest a name, but it is bound here
against the owner's own registry and the columns of the file being scored. A model that lacks
half or more of its inputs in this file is not offered, because it would score mostly guesses.
When more than one model fits, the user is asked; nothing is picked at random.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from app.ml.model_registry import get_model_registry
from app.ml.scoring import MAX_MISSING_INPUT_SHARE

_GENERIC = {
    "model", "models", "the", "this", "that", "these", "with", "using", "use", "score", "scores", "scoring",
    "file", "data", "dataset", "new", "rows", "row", "predict", "prediction", "predictions", "run", "apply",
    "saved", "trained", "existing", "latest", "every", "all", "for", "and", "my", "our", "on", "to",
}


@dataclass
class ModelChoice:
    name: str
    version: int
    target: str
    task: str
    input_columns: list[str]
    missing: list[str]
    tags: dict[str, Any]
    fingerprint: str
    stage: str = ""
    created_at: str = ""

    @property
    def usable(self) -> bool:
        return not self.input_columns or len(self.missing) < MAX_MISSING_INPUT_SHARE * len(self.input_columns)

    @property
    def label(self) -> str:
        return f"{self.name} (predicts {self.target}, v{self.version})"


@dataclass
class ChoiceOutcome:
    choice: Optional[ModelChoice] = None
    options: list[ModelChoice] = field(default_factory=list)
    reason: str = ""


def _tokens(text: Any) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", str(text or "").casefold()) if len(t) >= 3 and t not in _GENERIC}


def list_scoring_models(owner_id: Any, columns: list[str]) -> list[ModelChoice]:
    """The owner's models with the version that would score (production if set, else the latest)."""
    registry = get_model_registry(owner_id)
    present = set(columns)
    out: list[ModelChoice] = []
    for info in registry.list_models():
        versions = registry.get_model_versions(info["name"])
        if not versions:
            continue
        wanted = info.get("production_version") or info.get("latest_version")
        v = next((x for x in versions if x["version"] == wanted), versions[-1])
        params = v.get("params") or {}
        inputs = [str(c) for c in params.get("input_columns") or []]
        target = str(params.get("target_column") or "")
        out.append(ModelChoice(
            name=info["name"], version=int(v["version"]), target=target, task=str(params.get("task") or ""),
            input_columns=inputs, missing=[c for c in inputs if c not in present], tags=dict(info.get("tags") or {}),
            fingerprint=f"{v.get('model_id') or info.get('model_id')}:{v['version']}:{v.get('created_at')}",
            stage=str(v.get("stage") or ""), created_at=str(v.get("created_at") or ""),
        ))
    out.sort(key=lambda m: m.created_at, reverse=True)  # newest first
    return out


def choose_model(owner_id: Any, *, columns: list[str], requested: Optional[str] = None,
                 message: str = "") -> ChoiceOutcome:
    models = list_scoring_models(owner_id, columns)
    if not models:
        return ChoiceOutcome(reason="You have no saved models yet. Train one first, for example "
                                    "'predict <the outcome column>' on the data that has the outcome.")
    text = f"{requested or ''} {message}".casefold()
    if requested:
        exact = [m for m in models if m.name.casefold() == str(requested).strip().casefold()]
        if exact:
            return _checked(exact[0])
    named = [m for m in models if m.name.casefold() in text]
    if len(named) == 1:
        return _checked(named[0])

    usable = [m for m in models if m.usable]
    if not usable:
        closest = min(models, key=lambda m: len(m.missing))
        return ChoiceOutcome(reason=(
            f"None of your models can score this file. The closest, {closest.name} (predicts {closest.target}), "
            f"needs columns this file lacks: {', '.join(closest.missing[:8])}."))
    words = _tokens(text)
    by_target = [m for m in usable if _tokens(m.target) & words]
    pool = by_target or usable
    if len(pool) == 1:
        return ChoiceOutcome(choice=pool[0])
    return ChoiceOutcome(options=pool[:8], reason="Several of your saved models can score this file. Which one should I use?")


def _checked(model: ModelChoice) -> ChoiceOutcome:
    if model.usable:
        return ChoiceOutcome(choice=model)
    return ChoiceOutcome(reason=(
        f"{model.name} cannot score this file: it lacks {len(model.missing)} of the {len(model.input_columns)} "
        f"columns the model uses ({', '.join(model.missing[:8])})."))


def model_by_name(owner_id: Any, name: str, columns: list[str]) -> Optional[ModelChoice]:
    return next((m for m in list_scoring_models(owner_id, columns) if m.name == name), None)
