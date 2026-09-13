"""Model registry: version, store, and compare models.

As you iterate on features/models, you'll want to know *which* model is
deployed, what it was trained on, and how it compared to predecessors.
This is a tiny, file-backed registry (no DB required) to make that explicit
and auditable.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ModelRecord:
    """Metadata about a trained model."""

    name: str
    version: int
    created: float = field(default_factory=time.time)
    artifact_path: str = ""
    metrics: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class ModelRegistry:
    """A simple JSON-backed registry of model versions."""

    def __init__(self, root: str | Path = "models") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._file = self.root / "registry.json"
        self._records: dict[str, list[ModelRecord]] = self._load()

    def _load(self) -> dict[str, list[ModelRecord]]:
        if not self._file.exists():
            return {}
        with open(self._file) as fh:
            raw = json.load(fh)
        return {
            name: [ModelRecord(**r) for r in recs]
            for name, recs in raw.items()
        }

    def _save(self) -> None:
        payload = {
            name: [r.as_dict() for r in recs]
            for name, recs in self._records.items()
        }
        with open(self._file, "w") as fh:
            json.dump(payload, fh, indent=2)

    def register(
        self,
        name: str,
        artifact_path: str,
        metrics: dict | None = None,
        params: dict | None = None,
        notes: str = "",
    ) -> ModelRecord:
        """Add a new version of a model, auto-incrementing its version."""
        recs = self._records.setdefault(name, [])
        version = max((r.version for r in recs), default=0) + 1
        record = ModelRecord(
            name=name,
            version=version,
            artifact_path=artifact_path,
            metrics=metrics or {},
            params=params or {},
            notes=notes,
        )
        recs.append(record)
        self._save()
        return record

    def latest(self, name: str) -> ModelRecord | None:
        recs = self._records.get(name, [])
        return max(recs, key=lambda r: r.version) if recs else None

    def history(self, name: str) -> list[ModelRecord]:
        return self._records.get(name, [])

    def list_names(self) -> list[str]:
        return list(self._records.keys())
