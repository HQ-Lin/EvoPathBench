"""Mechanism taxonomy for auditable self-evolution evaluation."""
from __future__ import annotations

from typing import Dict, Tuple

# ACQUIRE is the mechanism chain measured by EvoPathBench. Risk calibration is a
# transversal guardrail rather than another memory operation.
ACQUIRE_MECHANISMS: Tuple[str, ...] = (
    "outcome_attribution",
    "experience_compression",
    "scope_qualification",
    "evidence_consolidation",
    "memory_invocation",
    "conflict_revision",
    "capability_endurance",
)
GUARDRAIL_MECHANISMS: Tuple[str, ...] = ("risk_calibration",)
EVOLUTION_MECHANISMS: Tuple[str, ...] = ACQUIRE_MECHANISMS + GUARDRAIL_MECHANISMS

MECHANISM_DESCRIPTIONS: Dict[str, str] = {
    "outcome_attribution": "Attribute feedback to the decision or assumption that should change.",
    "experience_compression": "Distill a trajectory into a reusable strategy artifact.",
    "scope_qualification": "Infer when an artifact applies and resist surface-level shortcuts.",
    "evidence_consolidation": "Accumulate evidence and gate persistent state updates.",
    "memory_invocation": "Retrieve and apply the right artifact at decision time.",
    "conflict_revision": "Detect counterevidence, invalidate stale rules, and install scoped replacements.",
    "capability_endurance": "Preserve previously acquired capabilities through later learning.",
    "risk_calibration": "Adapt while continuing to satisfy drawdown, turnover, and action constraints.",
}

ROLE_MECHANISMS: Dict[str, Tuple[str, ...]] = {
    "learn_near": (
        "outcome_attribution",
        "experience_compression",
        "evidence_consolidation",
    ),
    "probe_near": ("memory_invocation",),
    "probe_transfer": ("scope_qualification", "memory_invocation"),
    "retention_anchor": ("capability_endurance", "memory_invocation"),
    "update": ("outcome_attribution", "conflict_revision"),
    "probe_update": ("conflict_revision", "scope_qualification", "memory_invocation"),
    "stress": ("risk_calibration", "capability_endurance"),
    "shortcut_control": ("scope_qualification",),
}

TEMPLATE_MECHANISMS: Dict[str, Tuple[str, ...]] = {
    "accumulation": (
        "outcome_attribution",
        "experience_compression",
        "evidence_consolidation",
        "memory_invocation",
        "scope_qualification",
    ),
    "interference": (
        "evidence_consolidation",
        "memory_invocation",
        "capability_endurance",
        "scope_qualification",
    ),
    "reversal": (
        "outcome_attribution",
        "conflict_revision",
        "memory_invocation",
        "capability_endurance",
        "risk_calibration",
    ),
    "chronological": (
        "outcome_attribution",
        "experience_compression",
        "evidence_consolidation",
        "memory_invocation",
        "conflict_revision",
        "capability_endurance",
        "risk_calibration",
    ),
}
