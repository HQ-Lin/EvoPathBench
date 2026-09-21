"""Crash-safe trajectory persistence for long-running remote evaluations."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from .models import EvaluationRecord


STORE_SCHEMA_VERSION = "evopath-trajectory-store-v1"
KEY_FIELDS = ("condition", "campaign_id", "stream_id")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def trajectory_key(condition: str, campaign_id: int, stream_id: str) -> Dict[str, Any]:
    return {
        "condition": str(condition),
        "campaign_id": int(campaign_id),
        "stream_id": str(stream_id),
    }


def trajectory_id(key: Mapping[str, Any]) -> str:
    normalized = {field: key[field] for field in KEY_FIELDS}
    label = re.sub(
        r"[^a-zA-Z0-9._-]+",
        "-",
        f"{normalized['condition']}__c{normalized['campaign_id']}__{normalized['stream_id']}",
    ).strip("-._")[:120]
    return f"{label}__{_sha256_json(normalized)[:16]}"


def _checked_text(value: Any, api_key_env: str) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    secret = os.getenv(api_key_env, "")
    if secret and secret in text:
        raise RuntimeError("secret-leak guard rejected a trajectory artifact")
    return text


def _atomic_write_json(path: Path, value: Any, api_key_env: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(_checked_text(value, api_key_env))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


class TrajectoryStore:
    """Persist one complete condition/campaign/stream path per atomic shard."""

    def __init__(
        self,
        run_dir: Path,
        expected_keys: Sequence[Mapping[str, Any]],
        api_key_env: str = "EVOPATHBENCH_API_KEY",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.api_key_env = api_key_env
        self.shards_dir = self.run_dir / "checkpoints" / "trajectories"
        self.progress_dir = self.run_dir / "progress"
        self.ledger_path = self.progress_dir / "trajectory_ledger.jsonl"
        self.progress_path = self.progress_dir / "trajectory_progress.json"
        self.quarantine_dir = self.run_dir / "quarantine" / "trajectories"
        self.expected_keys = [
            {field: key[field] for field in KEY_FIELDS} for key in expected_keys
        ]
        self.expected_ids = [trajectory_id(key) for key in self.expected_keys]
        if len(self.expected_ids) != len(set(self.expected_ids)):
            raise ValueError("trajectory protocol contains duplicate keys")
        self.expected_by_id = dict(zip(self.expected_ids, self.expected_keys))
        self._lock = threading.RLock()
        self._attempts: Dict[str, int] = {}
        self._failed_attempts = 0
        self._active: Dict[str, Dict[str, Any]] = {}
        self._quarantined: List[str] = []
        self._load_attempts()
        self._completed = self._load_valid_shards()
        self._write_progress()

    def _load_attempts(self) -> None:
        if not self.ledger_path.exists():
            return
        with self.ledger_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "STARTED" and event.get("trajectory_id"):
                    item_id = str(event["trajectory_id"])
                    self._attempts[item_id] = self._attempts.get(item_id, 0) + 1
                elif event.get("event") == "FAILED":
                    self._failed_attempts += 1

    def _append_event(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self.progress_dir.mkdir(parents=True, exist_ok=True)
            self.progress_dir.chmod(0o700)
            payload = dict(event)
            payload.setdefault("schema_version", STORE_SCHEMA_VERSION)
            payload.setdefault("timestamp_utc", _utc_now())
            line = _canonical_json(payload) + "\n"
            secret = os.getenv(self.api_key_env, "")
            if secret and secret in line:
                raise RuntimeError("secret-leak guard rejected a trajectory ledger event")
            with self.ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            self.ledger_path.chmod(0o600)

    def _quarantine(self, path: Path, reason: str) -> None:
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        target = self.quarantine_dir / f"{path.name}.{int(datetime.now().timestamp())}.corrupt"
        os.replace(path, target)
        self._quarantined.append(str(target.relative_to(self.run_dir)))
        self._append_event(
            {
                "event": "QUARANTINED",
                "source": str(path.relative_to(self.run_dir)),
                "target": str(target.relative_to(self.run_dir)),
                "reason": reason[:1000],
            }
        )

    def _load_valid_shards(self) -> Dict[str, Dict[str, Any]]:
        completed: Dict[str, Dict[str, Any]] = {}
        if not self.shards_dir.exists():
            return completed
        for path in sorted(self.shards_dir.glob("*.json")):
            try:
                shard = json.loads(path.read_text(encoding="utf-8"))
                if shard.get("schema_version") != STORE_SCHEMA_VERSION:
                    raise ValueError("unsupported trajectory shard schema")
                item_id = str(shard["trajectory_id"])
                if item_id not in self.expected_by_id:
                    raise ValueError("trajectory is not part of the frozen protocol")
                if shard.get("trajectory_key") != self.expected_by_id[item_id]:
                    raise ValueError("trajectory key does not match its stable id")
                records = shard.get("records", {})
                counts = shard.get("record_counts", {})
                for name in ("evaluation_records", "training_records", "state_records"):
                    if not isinstance(records.get(name), list) or counts.get(name) != len(
                        records[name]
                    ):
                        raise ValueError("trajectory record count mismatch")
                checksum = shard.get("payload_sha256")
                payload = dict(shard)
                payload.pop("payload_sha256", None)
                if checksum != _sha256_json(payload):
                    raise ValueError("trajectory shard checksum mismatch")
                completed[item_id] = shard
            except Exception as exc:
                self._quarantine(path, f"{type(exc).__name__}: {exc}")
        return completed

    @property
    def completed_ids(self) -> Set[str]:
        with self._lock:
            return set(self._completed)

    def begin(self, key: Mapping[str, Any]) -> str:
        with self._lock:
            item_id = trajectory_id(key)
            if item_id not in self.expected_by_id:
                raise ValueError("cannot start a trajectory outside the frozen protocol")
            if item_id in self._completed:
                raise ValueError("cannot restart an already completed trajectory")
            if item_id in self._active:
                raise ValueError("trajectory is already active")
            self._attempts[item_id] = self._attempts.get(item_id, 0) + 1
            active = {
                "trajectory_id": item_id,
                "trajectory_key": self.expected_by_id[item_id],
                "attempt": self._attempts[item_id],
                "started_at_utc": _utc_now(),
            }
            self._active[item_id] = active
            self._append_event({"event": "STARTED", **active})
            self._write_progress()
            return item_id

    def complete(self, key: Mapping[str, Any], result: Dict[str, Any], usage: Dict[str, Any]) -> str:
        item_id = trajectory_id(key)
        with self._lock:
            if item_id in self._completed:
                raise ValueError("trajectory completion would overwrite an existing shard")
        expected_condition = str(key["condition"])
        accepted_conditions = {expected_condition, f"{expected_condition}_state_off"}
        expected_campaign = int(key["campaign_id"])
        expected_stream = str(key["stream_id"])
        for row in result["evaluation_records"]:
            if (
                row.condition not in accepted_conditions
                or row.campaign_id != expected_campaign
                or row.stream_id != expected_stream
            ):
                raise ValueError("evaluation record escaped its trajectory boundary")
        for collection_name in ("training_records", "state_records"):
            for row in result[collection_name]:
                if (
                    row.get("condition") not in accepted_conditions
                    or row.get("campaign_id") != expected_campaign
                    or row.get("stream_id") != expected_stream
                ):
                    raise ValueError(
                        f"{collection_name} record escaped its trajectory boundary"
                    )
        records = {
            "evaluation_records": [row.to_dict() for row in result["evaluation_records"]],
            "training_records": list(result["training_records"]),
            "state_records": list(result["state_records"]),
        }
        shard: Dict[str, Any] = {
            "schema_version": STORE_SCHEMA_VERSION,
            "trajectory_id": item_id,
            "trajectory_key": self.expected_by_id[item_id],
            "completed_at_utc": _utc_now(),
            "attempt": self._attempts.get(item_id, 1),
            "record_counts": {name: len(values) for name, values in records.items()},
            "usage": usage,
            "records": records,
        }
        shard["payload_sha256"] = _sha256_json(shard)
        path = self.shards_dir / f"{item_id}.json"
        with self._lock:
            if item_id in self._completed:
                raise ValueError("trajectory completion would overwrite an existing shard")
            _atomic_write_json(path, shard, self.api_key_env)
            self._completed[item_id] = shard
            self._active.pop(item_id, None)
            self._append_event(
                {
                    "event": "COMPLETED",
                    "trajectory_id": item_id,
                    "trajectory_key": self.expected_by_id[item_id],
                    "attempt": self._attempts.get(item_id, 1),
                    "shard": str(path.relative_to(self.run_dir)),
                    "record_counts": shard["record_counts"],
                    "usage": usage,
                }
            )
            self._write_progress()
            return item_id

    def fail(self, key: Mapping[str, Any], exc: BaseException) -> None:
        with self._lock:
            item_id = trajectory_id(key)
            self._active.pop(item_id, None)
            self._failed_attempts += 1
            self._append_event(
                {
                    "event": "FAILED",
                    "trajectory_id": item_id,
                    "trajectory_key": self.expected_by_id.get(item_id, dict(key)),
                    "attempt": self._attempts.get(item_id, 1),
                    "failure": {"type": type(exc).__name__, "message": str(exc)[:1000]},
                }
            )
            self._write_progress()

    def _usage_totals(self) -> Dict[str, int]:
        totals = {
            "logical_calls": 0,
            "failed_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }
        for shard in self._completed.values():
            usage = shard.get("usage", {})
            for field in totals:
                totals[field] += int(usage.get(field, 0) or 0)
        return totals

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self.expected_ids)
            completed = len(self._completed)
            active = [self._active[item_id] for item_id in sorted(self._active)]
            return {
                "schema_version": STORE_SCHEMA_VERSION,
                "updated_at_utc": _utc_now(),
                "total_trajectories": total,
                "completed_trajectories": completed,
                "remaining_trajectories": total - completed,
                "completion_fraction": completed / total if total else 1.0,
                "active": active[0] if len(active) == 1 else None,
                "active_trajectories": active,
                "completed_trajectory_ids": [
                    item_id for item_id in self.expected_ids if item_id in self._completed
                ],
                "attempts": dict(sorted(self._attempts.items())),
                "failed_trajectory_attempts": self._failed_attempts,
                "quarantined_shards": list(self._quarantined),
                "persisted_usage": self._usage_totals(),
            }

    def _write_progress(self) -> None:
        with self._lock:
            _atomic_write_json(self.progress_path, self.snapshot(), self.api_key_env)

    def materialize(self, config: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            missing = [item_id for item_id in self.expected_ids if item_id not in self._completed]
            if missing:
                raise RuntimeError(f"cannot materialize an incomplete run; missing {len(missing)} trajectories")
            evaluation: List[EvaluationRecord] = []
            training: List[Dict[str, Any]] = []
            states: List[Dict[str, Any]] = []
            for item_id in self.expected_ids:
                records = self._completed[item_id]["records"]
                evaluation.extend(EvaluationRecord(**row) for row in records["evaluation_records"])
                training.extend(records["training_records"])
                states.extend(records["state_records"])
            return {
                "config": config,
                "evaluation_records": evaluation,
                "training_records": training,
                "state_records": states,
            }
