"""Versioned, bounded persistent strategy state."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class StrategyEntry:
    rule_id: str
    signature: str
    mode: str
    evidence_episode_ids: List[str]
    evidence_scores: List[float]
    confidence: float
    status: str = "active"
    supersedes: Optional[str] = None
    hypothesis: str = ""
    scope: Dict[str, str] = field(default_factory=dict)
    created_version: int = 0
    last_updated_version: int = 0
    counterevidence_episode_ids: List[str] = field(default_factory=list)
    counterevidence_scores: List[float] = field(default_factory=list)
    application_count: int = 0
    successful_application_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "StrategyEntry":
        return cls(**value)


@dataclass
class StrategyStore:
    max_entries: int = 12
    version: int = 0
    parent_hash: str = "genesis"
    entries: List[StrategyEntry] = field(default_factory=list)
    pending: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_entries": self.max_entries,
            "version": self.version,
            "parent_hash": self.parent_hash,
            "entries": [entry.to_dict() for entry in self.entries],
            "pending": self.pending,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "StrategyStore":
        return cls(
            max_entries=value.get("max_entries", 12),
            version=value.get("version", 0),
            parent_hash=value.get("parent_hash", "genesis"),
            entries=[StrategyEntry.from_dict(item) for item in value.get("entries", [])],
            pending=dict(value.get("pending", {})),
        )

    def content_hash(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def active_entries(self) -> List[StrategyEntry]:
        return [entry for entry in self.entries if entry.status == "active"]

    def retrieve(self, signature: str) -> Optional[StrategyEntry]:
        exact = [entry for entry in self.active_entries() if entry.signature == signature]
        if not exact:
            return None
        return max(exact, key=lambda entry: (entry.confidence, len(entry.evidence_episode_ids), entry.rule_id))

    def retrieve_with_match(self, signature: str) -> Tuple[Optional[StrategyEntry], str]:
        """Retrieve an exact rule first, then a conservative same-regime transfer."""
        exact = self.retrieve(signature)
        if exact is not None:
            return exact, "exact"
        regime = signature.split(":", 1)[0]
        candidates = [
            entry
            for entry in self.active_entries()
            if entry.signature.split(":", 1)[0] == regime
        ]
        if not candidates:
            return None, "miss"
        selected = max(
            candidates,
            key=lambda entry: (entry.confidence, len(entry.evidence_episode_ids), entry.rule_id),
        )
        return selected, "scope_transfer"

    @staticmethod
    def _confidence(scores: List[float], counter_scores: Optional[List[float]] = None) -> float:
        if not scores:
            return 0.0
        mean_score = sum(scores) / len(scores)
        evidence_bonus = min(0.25, 0.05 * len(scores))
        counter_penalty = min(0.40, 0.08 * len(counter_scores or []))
        return max(0.0, min(1.0, 0.50 + 6.0 * mean_score + evidence_bonus - counter_penalty))

    @staticmethod
    def _scope(signature: str) -> Dict[str, str]:
        regime, execution = (signature.split(":", 1) + ["unknown"])[:2]
        return {"regime": regime, "execution": execution}

    def record_application_outcome(self, rule_id: str, episode_id: str, score: float) -> bool:
        """Attach feedback to the artifact that actually influenced the decision."""
        for entry in self.entries:
            if entry.rule_id != rule_id or entry.status != "active":
                continue
            previous_hash = self.content_hash()
            entry.application_count += 1
            if score >= 0.0:
                entry.successful_application_count += 1
            else:
                entry.counterevidence_episode_ids.append(episode_id)
                entry.counterevidence_scores.append(float(score))
            entry.confidence = self._confidence(
                entry.evidence_scores,
                entry.counterevidence_scores,
            )
            self.version += 1
            entry.last_updated_version = self.version
            self.parent_hash = previous_hash
            return True
        return False

    def propose(
        self,
        signature: str,
        mode: str,
        episode_id: str,
        score: float,
        gate_evidence: int = 1,
        hypothesis: Optional[str] = None,
        scope: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, Optional[StrategyEntry]]:
        key = f"{signature}|{mode}"
        pending = self.pending.setdefault(key, {"episodes": [], "scores": []})
        pending["episodes"].append(episode_id)
        pending["scores"].append(float(score))
        if len(pending["episodes"]) < gate_evidence:
            return False, None
        previous_hash = self.content_hash()
        existing = self.retrieve(signature)
        evidence_ids = list(pending["episodes"])
        evidence_scores = list(pending["scores"])
        del self.pending[key]
        if existing is not None and existing.mode == mode:
            existing.evidence_episode_ids.extend(evidence_ids)
            existing.evidence_scores.extend(evidence_scores)
            if hypothesis:
                existing.hypothesis = hypothesis
            if scope is not None:
                existing.scope = dict(scope)
            existing.confidence = self._confidence(
                existing.evidence_scores,
                existing.counterevidence_scores,
            )
            existing.last_updated_version = self.version + 1
            committed = existing
        else:
            supersedes = existing.rule_id if existing is not None else None
            if existing is not None:
                existing.status = "invalidated"
            rule_id = hashlib.sha1(
                f"{self.version + 1}|{signature}|{mode}|{episode_id}".encode("utf-8")
            ).hexdigest()[:12]
            committed = StrategyEntry(
                rule_id=rule_id,
                signature=signature,
                mode=mode,
                evidence_episode_ids=evidence_ids,
                evidence_scores=evidence_scores,
                confidence=self._confidence(evidence_scores),
                status="active",
                supersedes=supersedes,
                hypothesis=hypothesis or f"Use {mode} when observable scope is {signature}.",
                scope=dict(scope) if scope is not None else self._scope(signature),
                created_version=self.version + 1,
                last_updated_version=self.version + 1,
            )
            self.entries.append(committed)
        self._enforce_capacity()
        self.version += 1
        self.parent_hash = previous_hash
        return True, committed

    def _enforce_capacity(self) -> None:
        active = self.active_entries()
        if len(active) <= self.max_entries:
            return
        active.sort(key=lambda entry: (entry.confidence, len(entry.evidence_episode_ids), entry.rule_id))
        for entry in active[: len(active) - self.max_entries]:
            entry.status = "invalidated"

    def remove_rule(self, rule_id: str) -> bool:
        for entry in self.entries:
            if entry.rule_id == rule_id and entry.status == "active":
                previous_hash = self.content_hash()
                entry.status = "invalidated"
                self.version += 1
                self.parent_hash = previous_hash
                return True
        return False
