"""Compile frozen Binance windows into longitudinal EvoPathBench tasks."""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .dataset import ROLES, _opponent_mix, _risk_contract, _role_split, _role_variant, write_dataset
from .market_data import REPLAY_SCHEMA, ReplayWindow, build_replay_windows, load_frozen_bars
from .mechanisms import ROLE_MECHANISMS, TEMPLATE_MECHANISMS
from .models import CheckpointSpec, MarketSourceSpec, ScenarioSpec, StreamEvent, StreamSpec
from .streams import StreamGenerator, validate_streams


BINANCE_FAMILIES: Tuple[str, ...] = (
    "trend",
    "mean_reversion",
    "liquidity",
    "event_jump",
    "orderflow_response",
    "risk_contract",
)


def _canonical_jsonl(items: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for item in items
    )


def _partition_for_role(role: str) -> str:
    if role in {"learn_near", "update"}:
        return "calibration"
    if role == "probe_near":
        return "development"
    return "test"


def _ranking_key(family: str, role: str) -> str:
    reversed_role = role in {"update", "probe_update"}
    if family == "trend":
        return "trend_negative" if reversed_role else "trend_positive"
    if family == "mean_reversion":
        return "momentum_persistence" if reversed_role else "mean_reversion"
    if family == "liquidity":
        return "liquidity_high" if reversed_role else "liquidity_low"
    if family == "event_jump":
        return "event_negative" if reversed_role else "event_positive"
    if family == "orderflow_response":
        return "reflexivity_negative" if reversed_role else "reflexivity_positive"
    if family == "risk_contract":
        return "risk_stress" if reversed_role or role == "stress" else "risk_calm"
    raise ValueError(f"unknown Binance family: {family}")


def _ranked_candidates(windows: Sequence[ReplayWindow], family: str, role: str) -> List[ReplayWindow]:
    key = _ranking_key(family, role)
    return sorted(
        windows,
        key=lambda item: (item.feature_scores.get(key, float("-inf")), item.resource_id),
        reverse=True,
    )


def _select_windows(
    windows: Sequence[ReplayWindow], instances_per_role: int, seed: int
) -> Dict[Tuple[str, str], List[ReplayWindow]]:
    by_partition: Dict[str, List[ReplayWindow]] = {"calibration": [], "development": [], "test": []}
    for window in windows:
        by_partition.setdefault(window.partition, []).append(window)
    required = Counter(_partition_for_role(role) for role in ROLES)
    for partition, role_count in required.items():
        needed = role_count * len(BINANCE_FAMILIES) * instances_per_role
        if len(by_partition.get(partition, [])) < needed:
            raise ValueError(
                f"not enough non-overlapping {partition} windows: need {needed}, found {len(by_partition.get(partition, []))}"
            )

    used = set()
    selected: Dict[Tuple[str, str], List[ReplayWindow]] = {}
    rng = random.Random(seed)
    role_order = ["learn_near", "update", "probe_near", "probe_transfer", "retention_anchor", "probe_update", "stress", "shortcut_control"]
    for role in role_order:
        partition = _partition_for_role(role)
        for family in BINANCE_FAMILIES:
            ranked = _ranked_candidates(by_partition[partition], family, role)
            if role == "update" and (family, "learn_near") in selected:
                latest_learning_time = max(
                    item.start_time_us for item in selected[(family, "learn_near")]
                )
                ranked = [item for item in ranked if item.start_time_us > latest_learning_time]
            # Deterministically jitter only exact/near ties without changing the
            # mechanism ranking. This avoids always selecting one symbol first.
            buckets: Dict[int, List[ReplayWindow]] = {}
            key = _ranking_key(family, role)
            for item in ranked:
                bucket = int(round(item.feature_scores.get(key, 0.0) * 1000))
                buckets.setdefault(bucket, []).append(item)
            ordered: List[ReplayWindow] = []
            for bucket in sorted(buckets, reverse=True):
                values = list(buckets[bucket])
                rng.shuffle(values)
                ordered.extend(values)
            picked = [item for item in ordered if item.resource_id not in used][:instances_per_role]
            if len(picked) != instances_per_role:
                raise ValueError(f"could not allocate unique replay windows for {family}/{role}")
            selected[(family, role)] = picked
            used.update(item.resource_id for item in picked)
    return selected


def _calendar_streams(
    scenarios: Sequence[ScenarioSpec], windows: Sequence[ReplayWindow], stream_seed: int
) -> List[StreamSpec]:
    grouped: Dict[Tuple[str, str], List[ScenarioSpec]] = {}
    for scenario in scenarios:
        grouped.setdefault((scenario.family_id, scenario.role), []).append(scenario)
    window_index = {item.resource_id: item for item in windows}
    streams: List[StreamSpec] = []
    for family_index, family in enumerate(BINANCE_FAMILIES):
        learning = sorted(
            grouped[(family, "learn_near")],
            key=lambda item: window_index[item.market_source.resource_id].start_time_us,
        )[:3]
        updates = sorted(
            grouped[(family, "update")],
            key=lambda item: window_index[item.market_source.resource_id].start_time_us,
        )[:2]
        ordered = sorted(
            learning + updates,
            key=lambda item: (
                window_index[item.market_source.resource_id].start_time_us,
                item.episode_id,
            ),
        )
        standard_probe_roles = (
            "probe_near",
            "probe_transfer",
            "retention_anchor",
            "stress",
            "shortcut_control",
        )
        standard_probes = [grouped[(family, role)][0].episode_id for role in standard_probe_roles]
        update_probe = grouped[(family, "probe_update")][0].episode_id
        streams.append(
            StreamSpec(
                stream_id=f"chronological-exogenous-{family}",
                template="chronological",
                layer="exogenous",
                focal_family=family,
                events=[StreamEvent(item.episode_id, True, True) for item in ordered],
                checkpoints=[
                    CheckpointSpec("K0", 0, "cold", standard_probes + [update_probe], []),
                    CheckpointSpec(
                        "K1", 1, "after_first_chronological_event", standard_probes + [update_probe],
                        ["experience_compression", "memory_invocation"],
                    ),
                    CheckpointSpec(
                        "K2", 2, "after_second_chronological_event", standard_probes + [update_probe],
                        ["evidence_consolidation", "memory_invocation"],
                    ),
                    CheckpointSpec(
                        "K3", 3, "after_third_chronological_event", standard_probes + [update_probe],
                        ["capability_endurance", "memory_invocation"],
                    ),
                    CheckpointSpec(
                        "K4", 4, "after_first_chronological_update", standard_probes + [update_probe],
                        ["conflict_revision", "capability_endurance", "risk_calibration"],
                    ),
                    CheckpointSpec(
                        "K5", 5, "after_second_chronological_update", standard_probes + [update_probe],
                        ["conflict_revision", "capability_endurance", "risk_calibration"],
                    ),
                ],
                stream_seed=stream_seed * 1009 + family_index * 97 + 41,
                target_mechanisms=list(TEMPLATE_MECHANISMS["chronological"]),
            )
        )
    return streams


def _scenario_from_window(
    family: str, role: str, index: int, window: ReplayWindow, seed: int
) -> ScenarioSpec:
    rng = random.Random(seed ^ int(window.resource_id[-12:], 16))
    split, hidden = _role_split(role)
    params = {
        # These are an explicit simulated execution contract. They are not
        # claimed to be observed Binance bid/ask or account-specific fees.
        "spread_bps": 5.0,
        "liquidity": 20.0,
        "fee_bps": 2.0,
        "signal_noise": 0.0,
    }
    digest = hashlib.sha256(f"{family}|{role}|{index}|{window.resource_id}|{seed}".encode("utf-8")).hexdigest()
    return ScenarioSpec(
        episode_id=f"binance-{family}-{role}-{index:02d}-{digest[:10]}",
        family_id=family,
        role=role,
        layer="exogenous",
        variant=_role_variant(role),
        split=split,
        hidden=hidden,
        horizon=len(window.decision_prices),
        initial_price=window.decision_prices[0],
        mechanism_params=params,
        opponent_mix=_opponent_mix("opponent_reflexivity" if family == "orderflow_response" else family, role),
        risk_contract=_risk_contract(family, role),
        observation_mapping_seed=rng.randrange(1, 2**31 - 1),
        environment_seed=rng.randrange(1, 2**31 - 1),
        evolution_mechanisms=list(ROLE_MECHANISMS[role]),
        scoring_spec={"violation_penalty": 0.01, "drawdown_excess_penalty": 2.0, "valuation": 0.0},
        market_source=MarketSourceSpec(
            kind="binance_replay",
            resource_id=window.resource_id,
            clock="bar_close_next_open",
        ),
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _realism_report(
    windows: Sequence[ReplayWindow], source_manifest: Mapping[str, Any]
) -> Dict[str, Any]:
    returns: List[float] = []
    intervals: List[Tuple[str, int, int, str]] = []
    causal_violations = 0
    field_contract_violations = 0
    for window in windows:
        returns.extend(
            math.log(current / previous)
            for previous, current in zip(window.decision_prices, window.decision_prices[1:])
        )
        end_time = max(window.execution_time_us)
        intervals.append((window.symbol, window.start_time_us, end_time, window.resource_id))
        causal_violations += sum(
            1
            for decision, execution in zip(window.decision_time_us, window.execution_time_us)
            if decision >= execution
        )
        required_contracts = {"price_path", "reference_value", "volume_and_orderflow", "fee_spread_impact"}
        if set(window.field_contract) != required_contracts:
            field_contract_violations += 1
    overlaps = 0
    by_symbol: Dict[str, List[Tuple[int, int, str]]] = {}
    for symbol, start, end, resource_id in intervals:
        by_symbol.setdefault(symbol, []).append((start, end, resource_id))
    for values in by_symbol.values():
        values.sort()
        overlaps += sum(1 for left, right in zip(values, values[1:]) if right[0] <= left[1])
    checksum_status = Counter(source.get("checksum_status", "missing") for source in source_manifest["sources"])
    return {
        "report_version": "evolens-binance-realism-v1",
        "scope": "external validity and causal replay integrity; not a deployment or profitability certificate",
        "source_integrity": {
            "provider": source_manifest["source_provider"],
            "archive_count": len(source_manifest["sources"]),
            "bar_count": source_manifest["bar_count"],
            "symbols": source_manifest["symbols"],
            "official_checksum_status": dict(sorted(checksum_status.items())),
            "canonical_bars_sha256": source_manifest["sha256"]["bars.jsonl.gz"],
        },
        "selected_windows": {
            "count": len(windows),
            "unique_resource_count": len({window.resource_id for window in windows}),
            "partitions": dict(sorted(Counter(window.partition for window in windows).items())),
            "symbols": dict(sorted(Counter(window.symbol for window in windows).items())),
            "time_range_us": [min(window.start_time_us for window in windows), max(max(window.execution_time_us) for window in windows)],
            "same_symbol_interval_overlaps": overlaps,
        },
        "causal_contract": {
            "clock": "closed bar t -> next bar open execution -> next bar close valuation",
            "decision_execution_violations": causal_violations,
            "field_contract_violations": field_contract_violations,
            "network_required_during_evaluation": False,
        },
        "observed_return_summary": {
            "count": len(returns),
            "mean_log_return": statistics.fmean(returns) if returns else 0.0,
            "std_log_return": statistics.pstdev(returns) if len(returns) > 1 else 0.0,
            "q01": _quantile(returns, 0.01),
            "q05": _quantile(returns, 0.05),
            "median": _quantile(returns, 0.50),
            "q95": _quantile(returns, 0.95),
            "q99": _quantile(returns, 0.99),
        },
        "claim_boundaries": [
            "Prices, volumes, trade counts, and taker-buy shares originate from checksummed Binance Spot klines.",
            "Window prices are normalized to preserve returns while preventing nominal-price shortcuts.",
            "Spread, fees, focal-agent impact, and fills are simulated benchmark contracts, not historical Binance account executions.",
            "Public bulk klines are not a historical order-book feed and cannot identify queue priority or counterfactual market impact.",
            "Historical replay is exogenous; endogenous multi-agent claims require a separate empirically calibrated simulator.",
        ],
    }


def generate_binance_replay_dataset(
    snapshot_dir: Path,
    output_dir: Path,
    *,
    horizon: int = 24,
    stride: int = 25,
    bar_size: int = 5,
    instances_per_role: int = 5,
    seed: int = 7,
    stream_seed: int = 17,
    templates: Sequence[str] = ("accumulation", "interference", "reversal", "chronological"),
) -> Dict[str, Any]:
    if instances_per_role < 5:
        raise ValueError("instances_per_role must be at least 5 for the five-update stream protocol")
    bars, source_manifest = load_frozen_bars(snapshot_dir)
    snapshot_hash = source_manifest["sha256"]["bars.jsonl.gz"]
    windows = build_replay_windows(
        bars, snapshot_hash, horizon=horizon, stride=stride, bar_size=bar_size
    )
    selected = _select_windows(windows, instances_per_role=instances_per_role, seed=seed)
    scenarios: List[ScenarioSpec] = []
    selected_windows: List[ReplayWindow] = []
    for family in BINANCE_FAMILIES:
        for role in ROLES:
            for index, window in enumerate(selected[(family, role)]):
                scenarios.append(_scenario_from_window(family, role, index, window, seed))
                selected_windows.append(window)
    controlled_templates = [item for item in templates if item != "chronological"]
    streams = StreamGenerator(seed=stream_seed).generate(scenarios, templates=controlled_templates)
    if "chronological" in templates:
        streams.extend(_calendar_streams(scenarios, selected_windows, stream_seed))
    validate_streams(
        streams,
        [scenario.episode_id for scenario in scenarios],
        [scenario.episode_id for scenario in scenarios if scenario.hidden],
    )
    resource_items = [item.to_dict() for item in sorted(selected_windows, key=lambda value: value.resource_id)]
    provenance = {
        "source_manifest": source_manifest,
        "source_snapshot_sha256": snapshot_hash,
            "selection_policy": {
            "version": "binance-window-selector-v1",
            "split": "chronological 20/40/20/20 predeployment-fit/calibration/development/test with crossing windows purged",
            "overlap": "stride >= horizon+1; selected resource ids are globally unique",
            "labels": "ranked causal statistics; all thresholds and rankings computed without future bars outside each window",
            "stream_order": (
                "accumulation/interference/reversal use controlled mechanism order over non-overlapping historical windows; "
                "chronological streams preserve source time. Probes come from later chronological partitions."
            ),
        },
        "execution_contract": {
            "clock": "observe closed bar t; execute at bar t+1 open; mark at bar t+1 close",
            "price_normalization": "each window starts near 100; observed simple/log returns are preserved",
            "spread_bps": 5.0,
            "fee_bps": 2.0,
            "claim_boundary": "real Binance price/volume/orderflow replay with simulated execution costs; not order-book replay",
        },
    }
    realism_report = _realism_report(selected_windows, source_manifest)
    manifest = write_dataset(
        output_dir,
        scenarios,
        streams,
        {
            "seed": seed,
            "stream_seed": stream_seed,
            "horizon": horizon,
            "stride": stride,
            "bar_size": bar_size,
            "instances_per_role": instances_per_role,
            "families": list(BINANCE_FAMILIES),
            "layers": ["exogenous"],
            "templates": list(templates),
        },
        schema_version=REPLAY_SCHEMA,
        track_spec={
            "kind": "binance_replay",
            "market": "Binance Spot",
            "resource_catalog": "resources/replay_windows.jsonl",
            "resource_count": len(selected_windows),
            "network_during_evaluation": False,
        },
        resource_payloads={
            "resources/replay_windows.jsonl": _canonical_jsonl(resource_items),
            "resources/source_provenance.json": (
                json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8"),
            "resources/realism_report.json": (
                json.dumps(realism_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8"),
        },
        extra_manifest={
            "provenance_policy": "official archive checksum plus content-addressed local snapshot",
            "claim_boundary": "historical normalized price/volume replay; execution is simulated and exogenous",
        },
    )
    return manifest
