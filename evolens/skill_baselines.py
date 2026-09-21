"""Pure protocol helpers for the SkillOpt and SkillBoost baseline adapters.

The remote optimizer lives in :mod:`evolens.agent`.  This module keeps
candidate mutation and acceptance deterministic, unit-testable, and independent
of any model provider.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def clean_skill_document(value: Any, max_chars: int) -> str:
    """Normalize a generated skill document without changing its semantics."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", str(value or ""))
    text = "\n".join(line.rstrip() for line in text.strip().splitlines())
    return text[:max_chars].strip()


def apply_skillopt_edits(
    current: str,
    edits: Any,
    *,
    edit_budget: int,
    max_chars: int,
) -> Tuple[str, List[Dict[str, str]], List[Dict[str, str]]]:
    """Apply bounded textual edits using SkillOpt's add/replace/delete semantics.

    Invalid or non-matching edits are rejected rather than guessed.  ``target``
    must occur exactly once for replacement and deletion, which makes every
    accepted mutation reproducible from the audit ledger.
    """
    document = clean_skill_document(current, max_chars)
    applied: List[Dict[str, str]] = []
    rejected: List[Dict[str, str]] = []
    if not isinstance(edits, list):
        return document, applied, [{"reason": "edits_not_list"}]

    for raw in edits[:edit_budget]:
        if not isinstance(raw, dict):
            rejected.append({"reason": "edit_not_object"})
            continue
        operation = str(raw.get("op", "")).strip().lower()
        target = clean_skill_document(raw.get("target", ""), max_chars)
        content = clean_skill_document(raw.get("content", ""), max_chars)
        before = document
        reason = ""
        if operation in {"append", "add"}:
            if not content:
                reason = "empty_content"
            else:
                document = f"{document}\n\n{content}".strip()
        elif operation == "insert_after":
            if not target or document.count(target) != 1:
                reason = "target_must_match_once"
            elif not content:
                reason = "empty_content"
            else:
                document = document.replace(target, f"{target}\n{content}", 1)
        elif operation == "replace":
            if not target or document.count(target) != 1:
                reason = "target_must_match_once"
            elif not content:
                reason = "empty_content"
            else:
                document = document.replace(target, content, 1)
        elif operation == "delete":
            if not target or document.count(target) != 1:
                reason = "target_must_match_once"
            else:
                document = document.replace(target, "", 1)
        else:
            reason = "unsupported_operation"

        document = clean_skill_document(document, max_chars)
        record = {"op": operation, "target": target, "content": content}
        if reason or document == before:
            rejected.append({**record, "reason": reason or "no_change"})
            document = before
        else:
            applied.append(record)
    return document, applied, rejected


def _case_scores(report: Mapping[str, Any]) -> Dict[str, float]:
    rows = report.get("cases")
    if not isinstance(rows, list):
        return {}
    scores: Dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
            continue
        value = row.get("score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            scores[row["case_id"]] = float(value)
    return scores


def _slice_scores(report: Mapping[str, Any]) -> Dict[str, float]:
    rows = report.get("cases")
    if not isinstance(rows, list):
        return {}
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("slice_id"), str):
            continue
        value = row.get("score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            grouped.setdefault(row["slice_id"], []).append(float(value))
    return {key: sum(values) / len(values) for key, values in grouped.items() if values}


def select_skillboost_candidate(
    incumbent: Mapping[str, Any],
    candidates: Sequence[Tuple[str, Mapping[str, Any]]],
    *,
    max_case_regression: float,
    max_slice_regression: float = 0.0,
    min_improvement: float = 1e-12,
    regression_epsilon: float = 1e-12,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Select the best improving candidate under a paired case-regression cap."""
    baseline = _case_scores(incumbent)
    baseline_slices = _slice_scores(incumbent)
    baseline_mean = float(incumbent.get("mean_score", 0.0) or 0.0)
    assessments: List[Dict[str, Any]] = []
    for candidate_id, report in candidates:
        scores = _case_scores(report)
        candidate_slices = _slice_scores(report)
        shared = sorted(set(baseline) & set(scores))
        improvement = float(report.get("mean_score", 0.0) or 0.0) - baseline_mean
        regressions = sum(
            1 for case_id in shared if scores[case_id] < baseline[case_id] - regression_epsilon
        )
        regression_rate = regressions / len(shared) if shared else 1.0
        missing_slices = sorted(set(baseline_slices) - set(candidate_slices))
        slice_regressions = {
            slice_id: baseline_slices[slice_id] - candidate_slices[slice_id]
            for slice_id in sorted(set(baseline_slices) & set(candidate_slices))
            if candidate_slices[slice_id] < baseline_slices[slice_id]
        }
        slice_gate_passed = not missing_slices and all(
            value <= max_slice_regression for value in slice_regressions.values()
        )
        eligible = (
            bool(shared)
            and improvement > min_improvement
            and regression_rate < max_case_regression
            and slice_gate_passed
        )
        reasons: List[str] = []
        if not shared:
            reasons.append("no_paired_validation_cases")
        if improvement <= min_improvement:
            reasons.append("no_strict_improvement")
        if regression_rate >= max_case_regression:
            reasons.append("case_regression_cap_exceeded")
        if missing_slices:
            reasons.append("protected_slice_missing")
        if not slice_gate_passed and not missing_slices:
            reasons.append("slice_regression_cap_exceeded")
        assessments.append(
            {
                "candidate_id": candidate_id,
                "mean_score": float(report.get("mean_score", 0.0) or 0.0),
                "improvement": improvement,
                "paired_cases": len(shared),
                "regressed_cases": regressions,
                "case_regression_rate": regression_rate,
                "slice_regressions": slice_regressions,
                "missing_slices": missing_slices,
                "eligible": eligible,
                "reasons": reasons,
            }
        )
    eligible = [row for row in assessments if row["eligible"]]
    if not eligible:
        return None, assessments
    winner = max(
        eligible,
        key=lambda row: (row["improvement"], -row["case_regression_rate"], row["candidate_id"]),
    )
    return str(winner["candidate_id"]), assessments
