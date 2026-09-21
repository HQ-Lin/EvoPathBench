"""Longitudinal metrics for benchmark evaluation records."""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence, Tuple

from .models import EvaluationRecord


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def empirical_cvar(losses: Sequence[float], alpha: float = 0.90) -> float:
    """Upper-tail empirical CVaR with fractional boundary weighting."""
    if not losses:
        return 0.0
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    ordered = sorted((float(value) for value in losses), reverse=True)
    tail_mass = (1.0 - alpha) * len(ordered)
    if tail_mass <= 0:
        return ordered[0]
    whole = int(math.floor(tail_mass))
    fraction = tail_mass - whole
    weighted_sum = sum(ordered[:whole])
    if fraction > 1e-12 and whole < len(ordered):
        weighted_sum += fraction * ordered[whole]
    denominator = whole + fraction
    if denominator <= 0:
        return ordered[0]
    return weighted_sum / denominator


def _group_records(
    records: Iterable[EvaluationRecord],
    fields: Sequence[str],
) -> DefaultDict[Tuple[Any, ...], List[EvaluationRecord]]:
    groups: DefaultDict[Tuple[Any, ...], List[EvaluationRecord]] = defaultdict(list)
    for record in records:
        groups[tuple(getattr(record, field) for field in fields)].append(record)
    return groups


def _primary_summary(records: Sequence[EvaluationRecord]) -> List[Dict[str, Any]]:
    groups = _group_records(records, ("track_kind", "layer", "stream_template", "condition", "checkpoint_id", "checkpoint_order"))
    rows: List[Dict[str, Any]] = []
    for (track_kind, layer, stream_template, condition, checkpoint_id, checkpoint_order), values in sorted(groups.items()):
        rows.append(
            {
                "track_kind": track_kind, "layer": layer, "stream_template": stream_template,
                "condition": condition,
                "checkpoint_id": checkpoint_id,
                "checkpoint_order": checkpoint_order,
                "n": len(values),
                "mean_score": mean([item.score for item in values]),
                "mean_return_pct": mean([item.return_pct for item in values]),
                "mean_max_drawdown": mean([item.max_drawdown for item in values]),
                "mean_turnover": mean([item.turnover for item in values]),
                "violation_rate": mean([1.0 if item.violation_count else 0.0 for item in values]),
            }
        )
    return rows


def _capability_matrix(records: Sequence[EvaluationRecord]) -> List[Dict[str, Any]]:
    groups = _group_records(
        records, ("track_kind", "layer", "stream_template", "condition", "checkpoint_id", "checkpoint_order", "family_id")
    )
    rows: List[Dict[str, Any]] = []
    for (track_kind, layer, stream_template, condition, checkpoint_id, checkpoint_order, family_id), values in sorted(groups.items()):
        rows.append(
            {
                "track_kind": track_kind, "layer": layer, "stream_template": stream_template,
                "condition": condition,
                "checkpoint_id": checkpoint_id,
                "checkpoint_order": checkpoint_order,
                "family_id": family_id,
                "n": len(values),
                "mean_score": mean([item.score for item in values]),
                "mean_return_pct": mean([item.return_pct for item in values]),
            }
        )
    return rows


def _mechanism_capability(records: Sequence[EvaluationRecord]) -> List[Dict[str, Any]]:
    groups: DefaultDict[Tuple[str, str, str, str, str, int, str], List[EvaluationRecord]] = defaultdict(list)
    for record in records:
        for mechanism in record.mechanism_ids:
            groups[(record.track_kind, record.layer, record.stream_template, record.condition, record.checkpoint_id, record.checkpoint_order, mechanism)].append(record)
    rows: List[Dict[str, Any]] = []
    for (track_kind, layer, stream_template, condition, checkpoint_id, checkpoint_order, mechanism), values in sorted(groups.items()):
        rows.append(
            {
                "track_kind": track_kind, "layer": layer, "stream_template": stream_template,
                "condition": condition,
                "checkpoint_id": checkpoint_id,
                "checkpoint_order": checkpoint_order,
                "mechanism": mechanism,
                "n": len(values),
                "mean_score": mean([item.score for item in values]),
                "violation_rate": mean([1.0 if item.violation_count else 0.0 for item in values]),
            }
        )
    return rows


def _paired_ceg(records: Sequence[EvaluationRecord]) -> List[Dict[str, Any]]:
    """Paired capability-evolution gain of every method against baseline."""
    by_key: DefaultDict[Tuple[Any, ...], Dict[str, EvaluationRecord]] = defaultdict(dict)
    for record in records:
        key = (
            record.track_kind,
            record.layer,
            record.campaign_id,
            record.stream_id,
            record.checkpoint_id,
            record.probe_episode_id,
            record.repeat_id,
        )
        by_key[key][record.condition] = record
    grouped: DefaultDict[Tuple[str, str, str, str, str, str], List[float]] = defaultdict(list)
    for values in by_key.values():
        if "baseline" not in values:
            continue
        baseline = values["baseline"]
        for condition, method in values.items():
            if condition in {"baseline", "fixed_expert", "random"} or condition.endswith("_state_off"):
                continue
            grouped[(method.track_kind, method.layer, method.stream_template, condition, method.checkpoint_id, method.family_id)].append(
                method.score - baseline.score
            )
    rows: List[Dict[str, Any]] = []
    for (track_kind, layer, stream_template, condition, checkpoint_id, family_id), differences in sorted(grouped.items()):
        rows.append(
            {
                "track_kind": track_kind, "layer": layer, "stream_template": stream_template,
                "condition": condition,
                "baseline_condition": "baseline",
                "checkpoint_id": checkpoint_id,
                "family_id": family_id,
                "n_pairs": len(differences),
                "mean_ceg": mean(differences),
                "positive_pair_rate": mean([1.0 if value > 0 else 0.0 for value in differences]),
            }
        )
    return rows


def _mechanism_baseline_effect(records: Sequence[EvaluationRecord]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    conditions = sorted(
        {
            record.condition
            for record in records
            if record.condition not in {"baseline", "fixed_expert", "random"}
            and not record.condition.endswith("_state_off")
        }
    )
    for condition in conditions:
        for row in _mechanism_paired_effect(
            records, condition, "baseline", "mean_ceg"
        ):
            rows.append({"condition": condition, "baseline_condition": "baseline", **row})
    return rows


def _mechanism_state_utilization_effect(
    records: Sequence[EvaluationRecord],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    conditions = sorted(
        {
            record.condition[: -len("_state_off")]
            for record in records
            if record.condition.endswith("_state_off")
        }
    )
    for condition in conditions:
        for row in _mechanism_paired_effect(
            records, condition, f"{condition}_state_off", "mean_sue"
        ):
            rows.append({"condition": condition, **row})
    return rows


def _mechanism_paired_effect(
    records: Sequence[EvaluationRecord],
    on_condition: str,
    off_condition: str,
    metric_name: str,
) -> List[Dict[str, Any]]:
    by_key: DefaultDict[Tuple[Any, ...], Dict[str, EvaluationRecord]] = defaultdict(dict)
    for record in records:
        key = (
            record.track_kind,
            record.layer,
            record.campaign_id,
            record.stream_id,
            record.checkpoint_id,
            record.probe_episode_id,
            record.repeat_id,
        )
        by_key[key][record.condition] = record
    grouped: DefaultDict[Tuple[str, str, str, str, str], List[float]] = defaultdict(list)
    for values in by_key.values():
        if on_condition not in values or off_condition not in values:
            continue
        on_record = values[on_condition]
        difference = on_record.score - values[off_condition].score
        for mechanism in on_record.mechanism_ids:
            grouped[(on_record.track_kind, on_record.layer, on_record.stream_template, on_record.checkpoint_id, mechanism)].append(difference)
    return [
        {
            "track_kind": track_kind, "layer": layer, "stream_template": stream_template,
            "checkpoint_id": checkpoint_id,
            "mechanism": mechanism,
            "n_pairs": len(differences),
            metric_name: mean(differences),
            "positive_pair_rate": mean([1.0 if value > 0 else 0.0 for value in differences]),
        }
        for (track_kind, layer, stream_template, checkpoint_id, mechanism), differences in sorted(grouped.items())
    ]


def _state_utilization_effect(records: Sequence[EvaluationRecord]) -> List[Dict[str, Any]]:
    """Paired effect of enabling reads from an identical frozen persistent state."""
    by_key: DefaultDict[Tuple[Any, ...], Dict[str, EvaluationRecord]] = defaultdict(dict)
    for record in records:
        key = (
            record.track_kind,
            record.layer,
            record.campaign_id,
            record.stream_id,
            record.checkpoint_id,
            record.probe_episode_id,
            record.repeat_id,
        )
        by_key[key][record.condition] = record
    grouped: DefaultDict[Tuple[str, str, str, str, str, str], List[float]] = defaultdict(list)
    for values in by_key.values():
        for off_condition, off_record in values.items():
            if not off_condition.endswith("_state_off"):
                continue
            source_condition = off_condition[: -len("_state_off")]
            if source_condition not in values:
                continue
            on_record = values[source_condition]
            grouped[(on_record.track_kind, on_record.layer, on_record.stream_template, source_condition, on_record.checkpoint_id, on_record.family_id)].append(
                on_record.score - off_record.score
            )
    rows: List[Dict[str, Any]] = []
    for (track_kind, layer, stream_template, condition, checkpoint_id, family_id), differences in sorted(grouped.items()):
        rows.append(
            {
                "track_kind": track_kind, "layer": layer, "stream_template": stream_template,
                "condition": condition,
                "checkpoint_id": checkpoint_id,
                "family_id": family_id,
                "n_pairs": len(differences),
                "mean_sue": mean(differences),
                "positive_pair_rate": mean([1.0 if value > 0 else 0.0 for value in differences]),
            }
        )
    return rows


def _forgetting(records: Sequence[EvaluationRecord]) -> List[Dict[str, Any]]:
    cell_groups = _group_records(
        records,
        (
            "track_kind",
            "layer",
            "condition",
            "campaign_id",
            "stream_id",
            "checkpoint_order",
            "checkpoint_id",
            "family_id",
            "probe_episode_id",
        ),
    )
    cells: DefaultDict[Tuple[str, str, str, int, str, str, str], List[Tuple[int, str, float]]] = defaultdict(list)
    for (track_kind, layer, condition, campaign, stream, order, checkpoint, family, probe), values in cell_groups.items():
        cells[(track_kind, layer, condition, campaign, stream, family, probe)].append(
            (order, checkpoint, mean([item.score for item in values]))
        )
    stream_templates = {(record.track_kind, record.layer, record.stream_id): record.stream_template for record in records}
    grouped: DefaultDict[Tuple[str, str, str, str, str], List[float]] = defaultdict(list)
    for (track_kind, layer, condition, _campaign, stream, family, _probe), values in cells.items():
        ordered = sorted(values)
        if len(ordered) < 2:
            continue
        peak_before_final = max(value[2] for value in ordered[:-1])
        final_score = ordered[-1][2]
        grouped[(track_kind, layer, stream_templates[(track_kind, layer, stream)], condition, family)].append(max(0.0, peak_before_final - final_score))
    rows: List[Dict[str, Any]] = []
    for (track_kind, layer, stream_template, condition, family), losses in sorted(grouped.items()):
        rows.append(
            {
                "track_kind": track_kind, "layer": layer, "stream_template": stream_template,
                "condition": condition,
                "family_id": family,
                "n_paths": len(losses),
                "mean_forgetting": mean(losses),
                "cvar_tail_0_2_forgetting": empirical_cvar(losses, 0.80),
                "cvar_tail_0_1_forgetting": empirical_cvar(losses, 0.90),
                "cvar90_forgetting": empirical_cvar(losses, 0.90),
            }
        )
    return rows


def _risk_decomposition(records: Sequence[EvaluationRecord]) -> Dict[str, List[Dict[str, Any]]]:
    # Execution risk: repeats of one frozen state on one scenario.
    execution_cells = _group_records(
        records,
        (
            "track_kind",
            "layer",
            "stream_template",
            "condition",
            "campaign_id",
            "stream_id",
            "checkpoint_id",
            "checkpoint_order",
            "probe_episode_id",
        ),
    )
    execution_summary_20: DefaultDict[Tuple[str, str, str, str, str], List[float]] = defaultdict(list)
    execution_summary_10: DefaultDict[Tuple[str, str, str, str, str], List[float]] = defaultdict(list)
    for (track_kind, layer, template, condition, _campaign, _stream, checkpoint, _order, _probe), values in execution_cells.items():
        key = (track_kind, layer, template, condition, checkpoint)
        execution_summary_20[key].append(
            empirical_cvar([-item.score for item in values], 0.80)
        )
        execution_summary_10[key].append(
            empirical_cvar([-item.score for item in values], 0.90)
        )

    # Environment risk: scenarios for one frozen state and repeat.
    environment_cells = _group_records(
        records,
        (
            "track_kind",
            "layer",
            "stream_template",
            "condition",
            "campaign_id",
            "stream_id",
            "checkpoint_id",
            "checkpoint_order",
            "repeat_id",
        ),
    )
    environment_summary_20: DefaultDict[Tuple[str, str, str, str, str], List[float]] = defaultdict(list)
    environment_summary_10: DefaultDict[Tuple[str, str, str, str, str], List[float]] = defaultdict(list)
    for (track_kind, layer, template, condition, _campaign, _stream, checkpoint, _order, _repeat), values in environment_cells.items():
        key = (track_kind, layer, template, condition, checkpoint)
        environment_summary_20[key].append(
            empirical_cvar([-item.score for item in values], 0.80)
        )
        environment_summary_10[key].append(
            empirical_cvar([-item.score for item in values], 0.90)
        )

    # Path risk: campaign-level degradation from K0 to later checkpoints, averaged over common probes.
    record_lookup: DefaultDict[Tuple[str, str, str, int, str, str, str, int], List[float]] = defaultdict(list)
    checkpoint_orders: Dict[Tuple[str, str, str, str], int] = {}
    for record in records:
        key = (
            record.track_kind,
            record.layer,
            record.condition,
            record.campaign_id,
            record.stream_id,
            record.checkpoint_id,
            record.probe_episode_id,
            record.repeat_id,
        )
        record_lookup[key].append(record.score)
        checkpoint_orders[(record.track_kind, record.layer, record.stream_id, record.checkpoint_id)] = record.checkpoint_order
    stream_templates = {(record.track_kind, record.layer, record.stream_id): record.stream_template for record in records}
    path_losses: DefaultDict[Tuple[str, str, str, str, str], List[float]] = defaultdict(list)
    baseline: Dict[Tuple[str, str, str, int, str, str, int], float] = {}
    for key, values in record_lookup.items():
        track_kind, layer, condition, campaign, stream, checkpoint, probe, repeat = key
        if checkpoint_orders[(track_kind, layer, stream, checkpoint)] == 0:
            baseline[(track_kind, layer, condition, campaign, stream, probe, repeat)] = mean(values)
    campaign_differences: DefaultDict[Tuple[str, str, str, int, str, str], List[float]] = defaultdict(list)
    for key, values in record_lookup.items():
        track_kind, layer, condition, campaign, stream, checkpoint, probe, repeat = key
        if checkpoint_orders[(track_kind, layer, stream, checkpoint)] == 0:
            continue
        baseline_key = (track_kind, layer, condition, campaign, stream, probe, repeat)
        if baseline_key in baseline:
            campaign_differences[(track_kind, layer, condition, campaign, stream, checkpoint)].append(
                baseline[baseline_key] - mean(values)
            )
    for (track_kind, layer, condition, _campaign, _stream, checkpoint), losses in campaign_differences.items():
        path_losses[(track_kind, layer, stream_templates[(track_kind, layer, _stream)], condition, checkpoint)].append(mean(losses))

    def risk_rows(
        source_20: Dict[Tuple[str, str, str, str, str], List[float]],
        source_10: Dict[Tuple[str, str, str, str, str], List[float]],
        prefix: str,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "track_kind": track_kind, "layer": layer, "stream_template": stream_template,
                "condition": condition,
                "checkpoint_id": checkpoint,
                "n_units": len(source_20[key]),
                f"mean_{prefix}_cvar_tail_0_2_score_loss": mean(source_20[key]),
                f"mean_{prefix}_cvar_tail_0_1_score_loss": mean(source_10[key]),
                f"mean_{prefix}_cvar90_score_loss": mean(source_10[key]),
            }
            for key in sorted(source_20)
            for (track_kind, layer, stream_template, condition, checkpoint) in [key]
        ]

    path_rows = [
        {
            "track_kind": track_kind,
            "layer": layer,
            "stream_template": stream_template,
            "condition": condition,
            "checkpoint_id": checkpoint,
            "n_units": len(values),
            "path_cvar_tail_0_2_degradation": empirical_cvar(values, 0.80),
            "path_cvar_tail_0_1_degradation": empirical_cvar(values, 0.90),
            "path_cvar90_degradation": empirical_cvar(values, 0.90),
        }
        for (track_kind, layer, stream_template, condition, checkpoint), values in sorted(path_losses.items())
    ]

    return {
        "execution": risk_rows(execution_summary_20, execution_summary_10, "execution"),
        "environment": risk_rows(environment_summary_20, environment_summary_10, "environment"),
        "path": path_rows,
    }


def mechanism_summary(
    training_records: Sequence[Dict[str, Any]],
    evaluation_records: Sequence[EvaluationRecord],
) -> List[Dict[str, Any]]:
    counters: DefaultDict[Tuple[str, str, str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in training_records:
        condition = record["condition"]
        template = record.get("stream_template", "unspecified")
        track_kind = record.get("track_kind", "unspecified")
        layer = record.get("layer", "unspecified")
        for event in record.get("agent_trace", []):
            kind = event.get("event", "unknown")
            counters[(track_kind, layer, template, condition)][kind] += 1
            if kind == "model_call":
                counters[(track_kind, layer, template, condition)]["training_input_tokens"] += int(
                    event.get("usage", {}).get("input_tokens", 0) or 0
                )
                counters[(track_kind, layer, template, condition)]["training_output_tokens"] += int(
                    event.get("usage", {}).get("output_tokens", 0) or 0
                )
                counters[(track_kind, layer, template, condition)]["training_reasoning_tokens"] += int(
                    event.get("usage", {}).get("reasoning_tokens", 0) or 0
                )
                if not event.get("ok"):
                    counters[(track_kind, layer, template, condition)]["training_model_errors"] += 1
            if kind == "parse_error":
                counters[(track_kind, layer, template, condition)]["training_parse_errors"] += 1
            if kind == "retrieve" and event.get("hit"):
                counters[(track_kind, layer, template, condition)]["retrieve_hit"] += 1
            if kind == "write" and event.get("committed"):
                counters[(track_kind, layer, template, condition)]["write_committed"] += 1
    for record in evaluation_records:
        key = (record.track_kind, record.layer, record.stream_template, record.condition)
        counters[key]["probe_retrieve"] += record.retrieval_count
        counters[key]["probe_retrieve_hit"] += record.retrieval_hit_count
        counters[key]["probe_apply"] += record.application_count
        counters[key]["probe_action"] += record.action_count
        counters[key]["probe_exact_retrieve"] += record.exact_retrieval_count
        counters[key]["probe_scope_transfer"] += record.scope_transfer_retrieval_count
        counters[key]["probe_model_calls"] += record.model_call_count
        counters[key]["probe_model_errors"] += record.model_error_count
        counters[key]["probe_input_tokens"] += record.input_tokens
        counters[key]["probe_output_tokens"] += record.output_tokens
        counters[key]["probe_reasoning_tokens"] += record.reasoning_tokens
    rows: List[Dict[str, Any]] = []
    for (track_kind, layer, stream_template, condition), values in sorted(counters.items()):
        retrieves = values.get("retrieve", 0)
        writes = values.get("write", 0)
        rows.append(
            {
                "track_kind": track_kind, "layer": layer, "stream_template": stream_template,
                "condition": condition,
                "writes": writes,
                "committed_writes": values.get("write_committed", 0),
                "retrievals": retrieves,
                "retrieval_hit_rate": values.get("retrieve_hit", 0) / retrieves if retrieves else 0.0,
                "applications": values.get("apply", 0),
                "probe_retrievals": values.get("probe_retrieve", 0),
                "probe_retrieval_hit_rate": (
                    values.get("probe_retrieve_hit", 0) / values.get("probe_retrieve", 1)
                    if values.get("probe_retrieve", 0)
                    else 0.0
                ),
                "probe_applications": values.get("probe_apply", 0),
                "probe_exact_retrievals": values.get("probe_exact_retrieve", 0),
                "probe_scope_transfer_retrievals": values.get("probe_scope_transfer", 0),
                "training_actions": values.get("action", 0),
                "probe_actions": values.get("probe_action", 0),
                "training_model_calls": values.get("model_call", 0),
                "training_model_errors": (
                    values.get("training_model_errors", 0)
                    + values.get("training_parse_errors", 0)
                ),
                "probe_model_calls": values.get("probe_model_calls", 0),
                "probe_model_errors": values.get("probe_model_errors", 0),
                "input_tokens": (
                    values.get("training_input_tokens", 0)
                    + values.get("probe_input_tokens", 0)
                ),
                "output_tokens": (
                    values.get("training_output_tokens", 0)
                    + values.get("probe_output_tokens", 0)
                ),
                "reasoning_tokens": (
                    values.get("training_reasoning_tokens", 0)
                    + values.get("probe_reasoning_tokens", 0)
                ),
            }
        )
    return rows


def mechanism_operation_summary(training_records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counters: DefaultDict[Tuple[str, str, str, str, str, str], int] = defaultdict(int)
    for record in training_records:
        condition = record["condition"]
        template = record.get("stream_template", "unspecified")
        track_kind = record.get("track_kind", "unspecified")
        layer = record.get("layer", "unspecified")
        for event in record.get("agent_trace", []):
            mechanisms = event.get("mechanisms") or [event.get("mechanism")]
            for mechanism in (item for item in mechanisms if item):
                counters[(track_kind, layer, template, condition, mechanism, event.get("event", "unknown"))] += 1
    return [
        {
            "track_kind": track_kind, "layer": layer, "stream_template": template,
            "condition": condition,
            "mechanism": mechanism,
            "event": event,
            "count": count,
        }
        for (track_kind, layer, template, condition, mechanism, event), count in sorted(counters.items())
    ]


def artifact_state_summary(state_records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: DefaultDict[Tuple[str, str, str, str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for record in state_records:
        mechanism_state = record.get("state", {}).get("mechanism_state")
        if mechanism_state is None:
            continue
        groups[(record.get("track_kind", "unspecified"), record.get("layer", "unspecified"), record.get("stream_template", "unspecified"), record["condition"], record["checkpoint_id"], record["checkpoint_order"])].append(
            mechanism_state
        )
    return [
        {
            "track_kind": track_kind, "layer": layer, "stream_template": template,
            "condition": condition,
            "checkpoint_id": checkpoint_id,
            "checkpoint_order": checkpoint_order,
            "n_states": len(values),
            "mean_active_artifacts": mean([item["active_artifacts"] for item in values]),
            "mean_pending_hypotheses": mean([item["pending_hypotheses"] for item in values]),
            "mean_superseded_artifacts": mean([item["superseded_artifacts"] for item in values]),
        }
        for (track_kind, layer, template, condition, checkpoint_id, checkpoint_order), values in sorted(groups.items())
    ]


def summarize(
    records: Sequence[EvaluationRecord],
    training_records: Sequence[Dict[str, Any]],
    state_records: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    return {
        "record_count": len(records),
        "primary": _primary_summary(records),
        "capability_matrix": _capability_matrix(records),
        "mechanism_capability": _mechanism_capability(records),
        "paired_ceg": _paired_ceg(records),
        "mechanism_ceg": _mechanism_baseline_effect(records),
        "state_utilization_effect": _state_utilization_effect(records),
        "mechanism_sue": _mechanism_state_utilization_effect(records),
        "forgetting": _forgetting(records),
        "risk": _risk_decomposition(records),
        "mechanism": mechanism_summary(training_records, records),
        "mechanism_operations": mechanism_operation_summary(training_records),
        "artifact_state": artifact_state_summary(state_records),
        "interpretation": {
            "ceg": "Each evolving method minus baseline on paired complete campaigns.",
            "sue": "State-on minus state-off on the same frozen checkpoint, scenario, and execution seed.",
            "execution_cvar": "Score-loss CVaR across repeats of one frozen state and scenario.",
            "environment_cvar": "Score-loss CVaR across probe scenarios for one frozen state.",
            "path_cvar": "CVaR of campaign-level degradation from checkpoint zero on common probes.",
            "cvar_tail_0_2": "Mean loss in the worst 20% probability mass (equivalent to confidence-level CVaR at alpha=0.80).",
            "cvar_tail_0_1": "Mean loss in the worst 10% probability mass (equivalent to confidence-level CVaR at alpha=0.90).",
            "forgetting": "Peak-to-final score loss computed only on probe episode IDs shared across checkpoints.",
            "warning": "Small samples are diagnostic only; do not treat pilot CVaR as a tail-risk certificate.",
        },
    }
