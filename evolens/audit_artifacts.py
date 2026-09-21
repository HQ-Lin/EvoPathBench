"""Audit-oriented artifact layout for remote-agent experiments."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import __version__


AUDIT_SCHEMA_VERSION = "evolens-remote-audit-v1"
TOKEN_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _slug(value: str, limit: int = 64) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return (normalized or "unnamed")[:limit]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((project_root / "evolens").glob("*.py")) + [project_root / "pyproject.toml"]
    for path in paths:
        if not path.exists():
            continue
        digest.update(str(path.relative_to(project_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _checked_text(text: str, api_key_env: str) -> str:
    secret = os.getenv(api_key_env, "")
    if secret and secret in text:
        raise RuntimeError("secret-leak guard rejected an audit artifact")
    return text


def write_json(path: Path, value: Any, api_key_env: str = "EVOPATHBENCH_API_KEY") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    serialized = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(_checked_text(serialized, api_key_env), encoding="utf-8")
    path.chmod(0o600)


def write_jsonl(
    path: Path,
    values: Iterable[Dict[str, Any]],
    api_key_env: str = "EVOPATHBENCH_API_KEY",
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            line = _checked_text(_canonical_json(value), api_key_env)
            handle.write(line + "\n")
            count += 1
    path.chmod(0o600)
    return count


def create_audit_run(
    output_root: Path,
    run_kind: str,
    model: str,
    conditions: Sequence[str],
    descriptor: Dict[str, Any],
    dataset_manifest: Optional[Dict[str, Any]],
    run_name: Optional[str] = None,
    now: Optional[datetime] = None,
    project_root: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Create a unique, self-describing directory and an initial RUNNING manifest."""
    created = (now or _utc_now()).astimezone(timezone.utc)
    timestamp = created.strftime("%Y%m%dT%H%M%SZ")
    fingerprint_payload = {
        "run_kind": run_kind,
        "model": model,
        "conditions": list(conditions),
        "descriptor": descriptor,
        "dataset_sha256": (dataset_manifest or {}).get("sha256", {}),
    }
    fingerprint = _sha256_bytes(_canonical_json(fingerprint_payload).encode("utf-8"))[:10]
    condition_slug = _slug("+".join(conditions), 40)
    pieces = [timestamp, _slug(run_kind, 16), _slug(model, 40), condition_slug]
    if run_name:
        pieces.append(_slug(run_name, 40))
    pieces.append(fingerprint)
    run_id = "__".join(pieces)
    output_root.mkdir(parents=True, exist_ok=True)
    output_root.chmod(0o700)
    run_dir = output_root / run_id
    suffix = 2
    while run_dir.exists():
        run_dir = output_root / f"{run_id}__{suffix:02d}"
        suffix += 1
    run_dir.mkdir(mode=0o700)
    project = project_root or Path(__file__).resolve().parent.parent
    manifest = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "run_kind": run_kind,
        "status": "RUNNING",
        "created_at_utc": created.isoformat(),
        "completed_at_utc": None,
        "provider": "openai_compatible",
        "model": model,
        "thinking": {"enabled": False, "thinking_budget": None},
        "api_key_env": "EVOPATHBENCH_API_KEY",
        "conditions": list(conditions),
        "descriptor": descriptor,
        "dataset": dataset_manifest,
        "code": {
            "benchmark_version": __version__,
            "source_tree_sha256": source_tree_sha256(project),
        },
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": sys.platform,
        },
        "result": None,
        "failure": None,
    }
    write_json(run_dir / "audit_manifest.json", manifest)
    return run_dir, manifest


def _ledger_context(source: str, record: Any) -> Dict[str, Any]:
    if source == "evaluation":
        return {
            "source": source,
            "condition": record.condition,
            "campaign_id": record.campaign_id,
            "stream_id": record.stream_id,
            "checkpoint_id": record.checkpoint_id,
            "checkpoint_order": record.checkpoint_order,
            "probe_episode_id": record.probe_episode_id,
            "repeat_id": record.repeat_id,
            "execution_seed": record.execution_seed,
            "state_hash": record.state_hash,
        }
    return {
        "source": source,
        "condition": record.get("condition"),
        "campaign_id": record.get("campaign_id"),
        "stream_id": record.get("stream_id"),
        "event_index": record.get("event_index"),
        "episode_id": record.get("episode_id"),
        "execution_seed": record.get("execution_seed"),
        "state_hash_before": record.get("state_hash_before"),
        "state_hash_after": record.get("state_hash_after"),
    }


def _zero_tokens() -> Dict[str, int]:
    return {field: 0 for field in TOKEN_FIELDS}


def _event_tokens(event: Dict[str, Any]) -> Dict[str, int]:
    usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    reasoning_tokens = int(usage.get("reasoning_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", 0) or 0) or input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }


def _sum_event_tokens(events: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    total = _zero_tokens()
    for event in events:
        usage = _event_tokens(event)
        for field in TOKEN_FIELDS:
            total[field] += usage[field]
    return total


def _token_attribution(event: Dict[str, Any]) -> str:
    context = event.get("audit_context")
    if isinstance(context, dict) and context.get("token_attribution"):
        return str(context["token_attribution"])
    if event.get("phase") == "reflection":
        return "self_evolution_reflection"
    return "base_decision"


def model_call_ledger(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source, records in (
        ("training", result.get("training_records", [])),
        ("evaluation", result.get("evaluation_records", [])),
    ):
        for record_index, record in enumerate(records):
            trace = record.agent_trace if source == "evaluation" else record.get("agent_trace", [])
            context = _ledger_context(source, record)
            for trace_index, event in enumerate(trace):
                if event.get("event") != "model_call":
                    continue
                rows.append(
                    {
                        "ledger_schema_version": 3,
                        "record_index": record_index,
                        "trace_index": trace_index,
                        **context,
                        **{key: value for key, value in event.items() if key != "event"},
                        "token_attribution": _token_attribution(event),
                        **_event_tokens(event),
                    }
                )
    rows.sort(
        key=lambda item: (
            0 if item.get("source") == "training" else 1,
            int(item.get("record_index", 0)),
            int(item.get("trace_index", 0)),
        )
    )
    return rows


def state_transition_ledger(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in result.get("training_records", []):
        model_events = [
            event for event in record.get("agent_trace", []) if event.get("event") == "model_call"
        ]
        decision_events = [event for event in model_events if event.get("phase") == "decision"]
        reflection_events = [
            event
            for event in model_events
            if _token_attribution(event)
            in {"self_evolution_reflection", "self_evolution_skill_optimizer"}
        ]
        memory_events = [
            event
            for event in decision_events
            if _token_attribution(event) == "memory_conditioned_decision"
        ]
        context_events = [
            event
            for event in decision_events
            if _token_attribution(event) == "context_conditioned_decision"
        ]
        skill_events = [
            event
            for event in decision_events
            if _token_attribution(event) == "skill_conditioned_decision"
        ]
        validation_events = [
            event
            for event in decision_events
            if _token_attribution(event) == "skill_validation_decision"
        ]
        direct_evolution = _sum_event_tokens(reflection_events)
        memory_conditioned = _sum_event_tokens(memory_events)
        context_conditioned = _sum_event_tokens(context_events)
        skill_conditioned = _sum_event_tokens(skill_events)
        candidate_validation = _sum_event_tokens(validation_events)
        gross_evolution = {
            field: (
                direct_evolution[field]
                + memory_conditioned[field]
                + context_conditioned[field]
                + skill_conditioned[field]
                + candidate_validation[field]
            )
            for field in TOKEN_FIELDS
        }
        mechanism_events = [
            event
            for event in record.get("agent_trace", [])
            if event.get("event") in {"retrieve", "apply", "attribute", "write", "invalidate"}
        ]
        rows.append(
            {
                "ledger_schema_version": 2,
                "condition": record.get("condition"),
                "campaign_id": record.get("campaign_id"),
                "stream_id": record.get("stream_id"),
                "event_index": record.get("event_index"),
                "episode_id": record.get("episode_id"),
                "execution_seed": record.get("execution_seed"),
                "score": record.get("score"),
                "return_pct": record.get("return_pct"),
                "state_hash_before": record.get("state_hash_before"),
                "state_hash_after": record.get("state_hash_after"),
                "state_changed": record.get("state_hash_before") != record.get("state_hash_after"),
                "token_cost": {
                    "all_model_calls": _sum_event_tokens(model_events),
                    "decision": _sum_event_tokens(decision_events),
                    "direct_self_evolution_update": direct_evolution,
                    "direct_self_evolution_reflection": direct_evolution,
                    "memory_conditioned_decision": memory_conditioned,
                    "context_conditioned_decision": context_conditioned,
                    "skill_conditioned_decision": skill_conditioned,
                    "candidate_validation_decision": candidate_validation,
                    "gross_self_evolution_path": gross_evolution,
                    "semantics": {
                        "direct": "Tokens billed for reflection or skill-optimizer update calls.",
                        "memory_conditioned": (
                            "Full tokens for decisions whose prompt contained structured persistent artifacts; "
                            "this is not a marginal token estimate."
                        ),
                        "context_conditioned": (
                            "Full tokens for decisions whose prompt contained the bounded sliding episode window."
                        ),
                        "skill_conditioned": (
                            "Full tokens for deployment decisions whose prompt contained a persistent skill document."
                        ),
                        "candidate_validation": (
                            "Tokens for paired internal rollouts used only to accept or reject skill candidates."
                        ),
                        "gross": (
                            "Direct updates plus memory-, context-, skill-, and validation-conditioned decision tokens."
                        ),
                        "reasoning_tokens": "Subset of output_tokens when reported by the provider.",
                    },
                },
                "mechanism_events": mechanism_events,
            }
        )
    return rows


def _audit_readme(manifest: Dict[str, Any]) -> str:
    raw_saved = bool(
        manifest.get("descriptor", {}).get("remote_settings", {}).get("save_raw_responses")
    )
    raw_statement = (
        "Raw responses were explicitly retained; directory and files use restrictive local permissions."
        if raw_saved
        else "Raw prompts and raw responses are not stored; parsed outputs and prompt/response hashes remain auditable."
    )
    return "\n".join(
        [
            f"# EvoPathBench Remote Experiment Audit — {manifest['run_id']}",
            "",
            f"- Status: `{manifest['status']}`",
            f"- Provider/model: `openai_compatible/{manifest['model']}`",
            "- Thinking mode: `disabled` (fixed by the benchmark CLI)",
            f"- Created UTC: `{manifest['created_at_utc']}`",
            f"- Completed UTC: `{manifest.get('completed_at_utc')}`",
            "",
            "## Layout",
            "",
            "| Path | Audit purpose |",
            "|---|---|",
            "| `audit_manifest.json` | Run identity, protocol, dataset/code hashes and status |",
            "| `run_config.json` | Exact benchmark and remote-agent configuration |",
            "| `summary.json` | Aggregate longitudinal, risk and mechanism metrics |",
            "| `records/evaluation_records.jsonl` | Frozen-checkpoint probe outcomes |",
            "| `records/training_records.jsonl` | Ordered learning episodes and state transitions |",
            "| `records/state_records.jsonl` | Frozen persistent-state snapshots |",
            "| `records/smoke_result.json` | Single-episode smoke outcome, when run kind is smoke |",
            "| `records/final_state.json` | Final smoke state, when run kind is smoke |",
            "| `ledgers/model_calls.jsonl` | One row per logical remote call |",
            "| `ledgers/state_transitions.jsonl` | One row per learning event |",
            "| `integrity.json` | Size, line count and SHA-256 for audit files |",
            "| `SHA256SUMS` | Machine-verifiable checksums |",
            "",
            "Evaluation-only or smoke-only rows above may be absent according to `run_kind`.",
            raw_statement,
            "The API-key value is never an artifact; only the environment-variable name is recorded.",
            "",
            "## Token accounting",
            "",
            "Every row in `ledgers/model_calls.jsonl` exposes `input_tokens`, `output_tokens`, "
            "`reasoning_tokens`, `total_tokens`, and `token_attribution` as top-level fields.",
            "Every row in `ledgers/state_transitions.jsonl` contains `token_cost` with:",
            "",
            "- `direct_self_evolution_update`: reflection and skill-optimizer update calls;",
            "- `direct_self_evolution_reflection`: backward-compatible alias of the preceding field;",
            "- `memory_conditioned_decision`: full calls whose prompt contained structured persistent artifacts;",
            "- `context_conditioned_decision`: full calls whose prompt contained the sliding episode window;",
            "- `skill_conditioned_decision`: deployment calls whose prompt contained a skill document;",
            "- `candidate_validation_decision`: internal paired rollout calls used for skill acceptance;",
            "- `gross_self_evolution_path`: the update and four conditioned-decision categories combined.",
            "",
            "In `summary.json`, paired state-on/state-off input-token deltas estimate marginal "
            "persistent-memory prompt overhead. Provider-reported reasoning tokens are a subset of output tokens.",
            "",
            "## Verify",
            "",
            "```bash",
            "shasum -a 256 -c SHA256SUMS",
            "```",
            "",
        ]
    )


def _line_count(path: Path) -> int:
    if path.suffix not in {".jsonl", ".md", ".json", ""}:
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def finalize_audit_run(
    run_dir: Path,
    manifest: Dict[str, Any],
    status: str,
    result_metadata: Optional[Dict[str, Any]] = None,
    failure: Optional[BaseException] = None,
) -> Dict[str, Any]:
    if status not in {"COMPLETE", "INVALID", "FAILED"}:
        raise ValueError(f"unsupported terminal audit status: {status}")
    manifest = dict(manifest)
    manifest["status"] = status
    manifest["completed_at_utc"] = _utc_now().isoformat()
    manifest["result"] = result_metadata
    if failure is not None:
        message = str(failure)
        secret = os.getenv(str(manifest.get("api_key_env", "EVOPATHBENCH_API_KEY")), "")
        if secret:
            message = message.replace(secret, "[REDACTED]")
        message = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[REDACTED]", message)
        manifest["failure"] = {
            "type": type(failure).__name__,
            "message": message[:2000],
        }
    write_json(run_dir / "audit_manifest.json", manifest)
    readme = _audit_readme(manifest)
    readme_path = run_dir / "AUDIT_README.md"
    readme_path.write_text(_checked_text(readme, str(manifest["api_key_env"])), encoding="utf-8")
    readme_path.chmod(0o600)

    excluded = {"integrity.json", "SHA256SUMS"}
    files = sorted(
        path for path in run_dir.rglob("*") if path.is_file() and path.name not in excluded
    )
    integrity_rows = [
        {
            "path": str(path.relative_to(run_dir)),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "lines": _line_count(path),
        }
        for path in files
    ]
    integrity = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "generated_at_utc": _utc_now().isoformat(),
        "files": integrity_rows,
    }
    integrity_path = run_dir / "integrity.json"
    write_json(integrity_path, integrity)
    checksum_files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    checksum_text = "".join(
        f"{file_sha256(path)}  {path.relative_to(run_dir)}\n" for path in checksum_files
    )
    sums_path = run_dir / "SHA256SUMS"
    sums_path.write_text(_checked_text(checksum_text, str(manifest["api_key_env"])), encoding="utf-8")
    sums_path.chmod(0o600)
    return manifest
