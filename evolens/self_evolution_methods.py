"""Deterministic state adapters for hierarchical self-evolving skill methods.

Remote reasoning stays in :mod:`evolens.agent`.  The helpers here
validate, bound, and render persistent state so every accepted model proposal
has a reproducible audit representation.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Tuple

from .skill_baselines import clean_skill_document


SKILL_CONDITIONS = frozenset(
    {"skillopt", "skillboost", "skillx", "trace2skill", "skillgrad"}
)
NATIVE_UPDATE_CONDITIONS = frozenset({"skillx", "trace2skill", "skillgrad"})
SKILLX_LEVELS = ("planning", "functional", "atomic")
TRACE2SKILL_OPERATIONS = frozenset(
    {
        "add_section",
        "insert_after",
        "insert_before",
        "replace_in_section",
        "append_to_section",
        "delete_section",
    }
)


def _clean_inline(value: Any, limit: int) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]", " ", str(value or ""))
    return " ".join(text.split())[:limit].strip()


def _clean_list(value: Any, *, max_items: int, item_chars: int) -> List[str]:
    if not isinstance(value, list):
        return []
    cleaned: List[str] = []
    for item in value:
        text = _clean_inline(item, item_chars)
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def empty_skillx_library() -> Dict[str, List[Dict[str, Any]]]:
    return {level: [] for level in SKILLX_LEVELS}


def sanitize_skillx_library(
    value: Any,
    *,
    max_items_per_level: int,
    max_content_chars: int,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, str]]]:
    """Validate a complete SkillX three-level library returned by consolidation."""
    library = empty_skillx_library()
    rejected: List[Dict[str, str]] = []
    if not isinstance(value, Mapping):
        return library, [{"reason": "library_not_object"}]
    for level in SKILLX_LEVELS:
        rows = value.get(level)
        if not isinstance(rows, list):
            rejected.append({"level": level, "reason": "level_not_list"})
            continue
        seen = set()
        for index, raw in enumerate(rows):
            if len(library[level]) >= max_items_per_level:
                rejected.append({"level": level, "reason": "level_capacity_exceeded"})
                break
            if not isinstance(raw, Mapping):
                rejected.append({"level": level, "reason": "skill_not_object"})
                continue
            name = _clean_inline(raw.get("name"), 96)
            content = clean_skill_document(raw.get("content"), max_content_chars)
            key = name.casefold()
            if not name or not content:
                rejected.append({"level": level, "reason": "missing_name_or_content"})
                continue
            if key in seen:
                rejected.append({"level": level, "reason": "duplicate_name", "name": name})
                continue
            seen.add(key)
            source_count = raw.get("source_count", 1)
            if isinstance(source_count, bool) or not isinstance(source_count, (int, float)):
                source_count = 1
            library[level].append(
                {
                    "name": name,
                    "content": content,
                    "activation_signals": _clean_list(
                        raw.get("activation_signals"), max_items=6, item_chars=120
                    ),
                    "tools": _clean_list(raw.get("tools"), max_items=6, item_chars=80),
                    "source_count": max(1, int(source_count)),
                }
            )
    return library, rejected


def render_skillx_library(library: Mapping[str, Any], max_chars: int) -> str:
    headings = {
        "planning": "Planning skills",
        "functional": "Functional skills",
        "atomic": "Atomic skills",
    }
    lines = ["# SkillX hierarchical skill library"]
    for level in SKILLX_LEVELS:
        lines.extend(["", f"## {headings[level]}"])
        rows = library.get(level, []) if isinstance(library, Mapping) else []
        if not rows:
            lines.append("- No retained skill yet.")
            continue
        for row in rows:
            lines.extend(["", f"### {row['name']}"])
            signals = row.get("activation_signals") or []
            tools = row.get("tools") or []
            if signals:
                lines.append("Activate when: " + "; ".join(signals))
            if tools:
                lines.append("Relevant interfaces: " + ", ".join(tools))
            lines.append(str(row["content"]))
    return clean_skill_document("\n".join(lines), max_chars)


def sanitize_trace2skill_document(value: Any, max_chars: int) -> str:
    """Validate the reduce/apply stage's complete replacement document."""
    return clean_skill_document(value, max_chars)


def sanitize_trace2skill_proposal(
    value: Any,
    *,
    max_operations: int = 8,
    max_content_chars: int = 1600,
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Validate one independent Trace2Skill map-stage patch proposal."""
    rejected: List[Dict[str, str]] = []
    if not isinstance(value, Mapping):
        return {}, [{"reason": "proposal_not_object"}]
    label = _clean_inline(value.get("analysis_label"), 24).lower()
    if label not in {"success", "failure", "mixed"}:
        label = "mixed"
    proposal: Dict[str, Any] = {
        "analysis_label": label,
        "evidence": _clean_list(value.get("evidence"), max_items=8, item_chars=300),
        "protected_behaviors": _clean_list(
            value.get("protected_behaviors"), max_items=8, item_chars=300
        ),
        "patch": [],
    }
    raw_patch = value.get("patch")
    if not isinstance(raw_patch, list):
        return proposal, [{"reason": "patch_not_list"}]
    for raw in raw_patch[:max_operations]:
        if not isinstance(raw, Mapping):
            rejected.append({"reason": "patch_operation_not_object"})
            continue
        operation = _clean_inline(raw.get("operation"), 40).lower()
        target = _clean_inline(raw.get("target"), 300)
        content = clean_skill_document(raw.get("content"), max_content_chars)
        if operation not in TRACE2SKILL_OPERATIONS:
            rejected.append({"reason": "unsupported_patch_operation"})
            continue
        if not target:
            rejected.append({"reason": "missing_patch_target"})
            continue
        if operation != "delete_section" and not content:
            rejected.append({"reason": "missing_patch_content"})
            continue
        proposal["patch"].append(
            {
                "operation": operation,
                "target": target,
                "content": content,
                "rationale": _clean_inline(raw.get("rationale"), 500),
            }
        )
    if len(raw_patch) > max_operations:
        rejected.append({"reason": "patch_operation_capacity_exceeded"})
    return proposal, rejected


def empty_skillgrad_state() -> Dict[str, Any]:
    return {
        "routing": {"name": "", "description": "", "activation_signals": []},
        "body": "",
        "references": [],
        "momentum_patterns": [],
        "last_overlay": {},
    }


def sanitize_skillgrad_momentum(
    value: Any,
    *,
    max_patterns: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, str]]]:
    """Validate SkillGrad's persistent pattern memory and per-task overlay."""
    rejected: List[Dict[str, str]] = []
    if not isinstance(value, Mapping):
        return [], {}, [{"reason": "momentum_not_object"}]
    patterns_raw = value.get("patterns")
    patterns: List[Dict[str, Any]] = []
    seen = set()
    if not isinstance(patterns_raw, list):
        rejected.append({"reason": "patterns_not_list"})
    else:
        for raw in patterns_raw:
            if len(patterns) >= max_patterns:
                rejected.append({"reason": "pattern_capacity_exceeded"})
                break
            if not isinstance(raw, Mapping):
                rejected.append({"reason": "pattern_not_object"})
                continue
            pattern_id = _clean_inline(raw.get("pattern_id"), 64)
            description = _clean_inline(raw.get("description"), 600)
            if not pattern_id or not description or pattern_id in seen:
                rejected.append({"reason": "invalid_or_duplicate_pattern"})
                continue
            seen.add(pattern_id)
            kind = _clean_inline(raw.get("kind"), 24).lower()
            if kind not in {"success", "failure", "mixed"}:
                kind = "mixed"
            appeared_in = raw.get("appeared_in", 1)
            if isinstance(appeared_in, bool) or not isinstance(appeared_in, (int, float)):
                appeared_in = 1
            patterns.append(
                {
                    "pattern_id": pattern_id,
                    "kind": kind,
                    "anchor": _clean_inline(raw.get("anchor"), 160),
                    "appeared_in": max(1, int(appeared_in)),
                    "description": description,
                    "latest_executor_action": _clean_inline(
                        raw.get("latest_executor_action"), 500
                    ),
                    "remedy_log": _clean_list(
                        raw.get("remedy_log"), max_items=6, item_chars=300
                    ),
                }
            )
    overlay_raw = value.get("overlay")
    overlay: Dict[str, Any] = {}
    if isinstance(overlay_raw, Mapping):
        overlay = {
            "signal": _clean_inline(overlay_raw.get("signal"), 300),
            "pattern": _clean_inline(overlay_raw.get("pattern"), 300),
            "anchor": _clean_inline(overlay_raw.get("anchor"), 160),
            "gap": _clean_inline(overlay_raw.get("gap"), 500),
            "proposed_change": _clean_inline(overlay_raw.get("proposed_change"), 800),
        }
    else:
        rejected.append({"reason": "overlay_not_object"})
    return patterns, overlay, rejected


def sanitize_skillgrad_package(
    value: Any,
    *,
    max_references: int,
    max_content_chars: int,
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Validate SkillGrad's L1 routing, L2 body, and L3 references."""
    state = empty_skillgrad_state()
    rejected: List[Dict[str, str]] = []
    if not isinstance(value, Mapping):
        return state, [{"reason": "package_not_object"}]
    routing = value.get("routing")
    if isinstance(routing, Mapping):
        state["routing"] = {
            "name": _clean_inline(routing.get("name"), 96),
            "description": _clean_inline(routing.get("description"), 500),
            "activation_signals": _clean_list(
                routing.get("activation_signals"), max_items=8, item_chars=120
            ),
        }
    else:
        rejected.append({"reason": "routing_not_object"})
    state["body"] = clean_skill_document(value.get("body"), max_content_chars)
    references = value.get("references")
    seen = set()
    if not isinstance(references, list):
        rejected.append({"reason": "references_not_list"})
    else:
        for raw in references:
            if len(state["references"]) >= max_references:
                rejected.append({"reason": "reference_capacity_exceeded"})
                break
            if not isinstance(raw, Mapping):
                rejected.append({"reason": "reference_not_object"})
                continue
            name = _clean_inline(raw.get("name"), 96)
            content = clean_skill_document(raw.get("content"), max_content_chars)
            key = name.casefold()
            if not name or not content or key in seen:
                rejected.append({"reason": "invalid_or_duplicate_reference"})
                continue
            seen.add(key)
            state["references"].append(
                {
                    "name": name,
                    "when_to_load": _clean_inline(raw.get("when_to_load"), 240),
                    "content": content,
                }
            )
    if not state["body"]:
        rejected.append({"reason": "empty_body"})
    return state, rejected


def render_skillgrad_package(state: Mapping[str, Any], max_chars: int) -> str:
    routing = state.get("routing", {}) if isinstance(state, Mapping) else {}
    lines = ["# SkillGrad layered skill package", "", "## L1 routing metadata"]
    if routing.get("name"):
        lines.append("Name: " + str(routing["name"]))
    if routing.get("description"):
        lines.append("Description: " + str(routing["description"]))
    signals = routing.get("activation_signals") or []
    if signals:
        lines.append("Activation signals: " + "; ".join(signals))
    lines.extend(["", "## L2 general guidance", str(state.get("body", ""))])
    for reference in state.get("references", []):
        lines.extend(["", f"## L3 reference: {reference['name']}"])
        if reference.get("when_to_load"):
            lines.append("Load when: " + str(reference["when_to_load"]))
        lines.append(str(reference["content"]))
    return clean_skill_document("\n".join(lines), max_chars)
