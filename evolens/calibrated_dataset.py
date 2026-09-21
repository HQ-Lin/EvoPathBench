"""Large-scale Binance-calibrated semi-synthetic task-pool compiler."""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Mapping, Sequence, Tuple

from .calibration import CALIBRATED_TRACK_SCHEMA, CalibrationProfile
from .calibration import CalibrationCatalog
from .dataset import (
    FAMILIES,
    LAYERS,
    ROLES,
    _base_params,
    _opponent_mix,
    _risk_contract,
    _role_split,
    _role_variant,
    write_dataset,
)
from .mechanisms import ROLE_MECHANISMS
from .models import MarketSourceSpec, ScenarioSpec, StreamSpec
from .market import MarketSimulator
from .streams import StreamGenerator, validate_streams


TRANSFORM_VERSION = "binance-calibrated-family-role-transform-v2"


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


def _corr(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 3:
        return 0.0
    ml, mr = statistics.fmean(left), statistics.fmean(right)
    denominator = math.sqrt(sum((x - ml) ** 2 for x in left) * sum((y - mr) ** 2 for y in right))
    return sum((x - ml) * (y - mr) for x, y in zip(left, right)) / max(denominator, 1e-15)


def _synthetic_realism_audit(
    profile: CalibrationProfile,
    scenarios: Sequence[ScenarioSpec],
    holdout_validation: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compare untouched holdout targets with predeclared generated reference paths."""
    simulator = MarketSimulator(calibration_catalog=CalibrationCatalog([profile]))
    shock_scale = scenarios[0].mechanism_params["calibration_shock_scale"]
    reference = _neutral_reference_scenarios(profile, shock_scale, scenarios[0].horizon, count=500)
    returns: List[float] = []
    abs_pairs: List[List[Tuple[float, float]]] = [[] for _ in range(12)]
    drawdowns: List[float] = []
    worst_returns: List[float] = []
    for scenario in reference:
        latent = simulator._latent_market_path(scenario)
        prices = simulator._layered_exogenous_price_path(scenario, latent)
        local = [math.log(b / a) for a, b in zip(prices, prices[1:])]
        returns.extend(local)
        for lag in range(1, 13):
            abs_pairs[lag - 1].extend(zip([abs(item) for item in local[:-lag]], [abs(item) for item in local[lag:]]))
        peak = prices[0]
        drawdown = 0.0
        for price in prices:
            peak = max(peak, price)
            drawdown = max(drawdown, 1.0 - price / peak)
        drawdowns.append(drawdown)
        worst_returns.append(min(local))
    holdout = holdout_validation["estimate"]
    synthetic_volatility = statistics.pstdev(returns)
    synthetic_acf = [
        _corr([left for left, _ in pairs], [right for _, right in pairs]) for pairs in abs_pairs
    ]
    quantile_rows = []
    for index in (1, 5, 50, 95, 99):
        synthetic = _quantile(returns, index / 100.0)
        observed = holdout["return_quantiles"][f"q{index:03d}"]
        quantile_rows.append(
            {
                "quantile": index / 100.0,
                "synthetic": synthetic,
                "locked_holdout": observed,
                "standardized_absolute_error": abs(synthetic - observed) / max(holdout["robust_volatility"], 1e-12),
            }
        )
    drawdown_rows = {}
    for label, probability in (("q50", 0.50), ("q90", 0.90), ("q95", 0.95), ("q99", 0.99)):
        drawdown_rows[label] = {
            "synthetic": _quantile(drawdowns, probability),
            "locked_holdout": holdout["path24_max_drawdown_quantiles"][label],
        }
    return {
        "report_version": "evolens-binance-synthetic-realism-v1",
        "reference_path_rule": "500 neutral exogenous paths; shock scale fitted on pre-2024 data only; family/role interventions excluded",
        "reference_path_count": len(reference),
        "synthetic_return_count": len(returns),
        "volatility_ratio_synthetic_to_holdout": synthetic_volatility / max(holdout["price_volatility"], 1e-12),
        "return_quantile_comparison": quantile_rows,
        "mean_standardized_quantile_error": statistics.fmean(
            row["standardized_absolute_error"] for row in quantile_rows
        ),
        "abs_return_acf_l1": statistics.fmean(
            abs(a - b) for a, b in zip(synthetic_acf, holdout["abs_return_acf"])
        ),
        "path24_max_drawdown": drawdown_rows,
        "path24_worst_return_q05": {
            "synthetic": _quantile(worst_returns, 0.05),
            "locked_holdout": holdout["path24_worst_return_quantiles"]["q05"],
        },
        "status": "DESCRIPTIVE_NOT_TUNED_ON_HOLDOUT",
        "claim_boundary": "A fidelity diagnostic for generated task paths, not a deployment or profitability certificate.",
    }


def _neutral_reference_scenarios(
    profile: CalibrationProfile, shock_scale: float, horizon: int, count: int
) -> List[ScenarioSpec]:
    estimate = profile.fit_estimate
    prior = profile.protocol_priors
    params = _base_params("trend", horizon=horizon, price_process_version=2)
    params.update(
        price_drift=estimate["price_drift"],
        fundamental_drift=0.0,
        price_volatility=estimate["price_volatility"],
        fundamental_volatility=estimate["price_volatility"] * prior["fundamental_volatility_ratio"],
        initial_gap=0.0,
        volatility_persistence=estimate["volatility_persistence"],
        volatility_of_volatility=estimate["volatility_of_volatility"],
        regime_persistence=estimate["regime_persistence"],
        jump_intensity=estimate["jump_intensity"],
        jump_scale=estimate["jump_scale"],
        jump_tail_df=estimate["jump_tail_df"],
        jump_aftershock_decay=estimate["jump_aftershock_decay"],
        permanent_event_probability=prior["permanent_event_probability"],
        microstructure_noise=min(0.01, max(0.00005, 0.18 * estimate["median_log_high_low_range"])),
        calibration_shock_scale=shock_scale,
    )
    for regime, weight in estimate["regime_weights"].items():
        params[f"regime_weight_{regime}"] = weight
    return [
        ScenarioSpec(
            episode_id=f"internal-calibration-reference-{index:04d}",
            family_id="calibration_reference",
            role="internal_reference",
            layer="exogenous",
            variant="standard",
            split="internal_calibration",
            hidden=True,
            horizon=horizon,
            initial_price=100.0,
            mechanism_params={key: float(value) for key, value in params.items()},
            opponent_mix={},
            risk_contract=_risk_contract("trend", "learn_near"),
            observation_mapping_seed=1000 + index,
            environment_seed=7919 * (index + 1),
            market_source=MarketSourceSpec(
                kind="calibrated_procedural", calibration_id=profile.calibration_id
            ),
        )
        for index in range(count)
    ]


def _reference_volatility(profile: CalibrationProfile, shock_scale: float, horizon: int) -> float:
    simulator = MarketSimulator(calibration_catalog=CalibrationCatalog([profile]))
    returns: List[float] = []
    for scenario in _neutral_reference_scenarios(profile, shock_scale, horizon, count=300):
        latent = simulator._latent_market_path(scenario)
        prices = simulator._layered_exogenous_price_path(scenario, latent)
        returns.extend(math.log(b / a) for a, b in zip(prices, prices[1:]))
    return statistics.pstdev(returns)


def _fit_shock_scale(profile: CalibrationProfile, horizon: int) -> float:
    """One-dimensional indirect inference using fit data and fixed common seeds."""
    target = profile.fit_estimate["price_volatility"]
    scale = 1.0
    for _ in range(4):
        simulated = _reference_volatility(profile, scale, horizon)
        scale = min(2.0, max(0.01, scale * target / max(simulated, 1e-12)))
    return round(scale, 8)


def _orderflow_response_validation(profile: CalibrationProfile) -> Dict[str, Any]:
    mapping = profile.local_projection.get("simulator_indirect_mapping", {})
    impact = float(mapping.get("orderflow_impact", 0.0))
    decay = float(mapping.get("impact_decay", 0.0))
    rows = []
    squared_errors = []
    for estimate in profile.local_projection.get("estimates", []):
        horizon = int(estimate["horizon_steps"])
        simulated = impact * sum(decay ** step for step in range(horizon + 1))
        observed = float(estimate["cumulative_return_response"])
        interval = [float(item) for item in estimate.get("bootstrap_ci95", [observed, observed])]
        squared_errors.append((simulated - observed) ** 2)
        rows.append(
            {
                "horizon_steps": horizon,
                "horizon_minutes": estimate["horizon_minutes"],
                "observed_reduced_form_response": observed,
                "bootstrap_ci95": interval,
                "simulator_controlled_response": simulated,
                "simulator_inside_empirical_ci": interval[0] <= simulated <= interval[1],
            }
        )
    return {
        "schema_version": "evolens-orderflow-response-validation-v1",
        "fit_partition_only": True,
        "rows": rows,
        "root_mean_squared_curve_error": math.sqrt(statistics.fmean(squared_errors)) if squared_errors else 0.0,
        "inside_interval_rate": statistics.fmean(
            1.0 if row["simulator_inside_empirical_ci"] else 0.0 for row in rows
        ) if rows else 0.0,
        "claim": "response-shape consistency with a reduced-form Binance target; not causal validation",
    }


def _finite(value: float, lower: float, upper: float) -> float:
    if not math.isfinite(value):
        raise ValueError("non-finite calibration estimate")
    return min(upper, max(lower, value))


def _calibrated_params(
    profile: CalibrationProfile,
    family: str,
    role: str,
    rng: random.Random,
    horizon: int,
    shock_scale: float,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    baseline = _base_params(family, horizon=horizon, price_process_version=2)
    empirical = profile.fit_estimate
    symbol_names = sorted(profile.symbol_estimates)
    member_name = symbol_names[rng.randrange(len(symbol_names))]
    member = profile.symbol_estimates[member_name]

    # Cross-sectional symbol draws preserve empirically observed heterogeneity;
    # family ratios remain explicit benchmark interventions rather than claims
    # that Binance contains six ground-truth task families.
    global_vol = max(empirical["price_volatility"], 1e-8)
    symbol_vol_ratio = _finite(member["price_volatility"] / global_vol, 0.55, 1.80)
    family_vol_ratio = baseline["price_volatility"] / 0.007
    price_volatility = global_vol * family_vol_ratio * symbol_vol_ratio
    baseline.update(
        price_drift=empirical["price_drift"] + baseline["price_drift"],
        price_volatility=_finite(price_volatility, 1e-5, 0.08),
        fundamental_volatility=_finite(
            price_volatility * profile.protocol_priors["fundamental_volatility_ratio"], 1e-6, 0.04
        ),
        volatility_persistence=_finite(member["volatility_persistence"], 0.05, 0.98),
        volatility_of_volatility=_finite(member["volatility_of_volatility"], 0.03, 0.80),
        regime_persistence=_finite(member["regime_persistence"], 0.05, 0.98),
        jump_intensity=_finite(
            empirical["jump_intensity"] * (baseline["jump_intensity"] / 0.018), 0.0001, 0.35
        ),
        jump_scale=_finite(
            member["jump_scale"] * (baseline["jump_scale"] / 0.018), 1e-5, 0.22
        ),
        jump_tail_df=_finite(empirical["jump_tail_df"], 2.2, 12.0),
        jump_aftershock_decay=_finite(empirical["jump_aftershock_decay"], 0.05, 0.95),
        impact_decay=_finite(empirical["impact_decay"], 0.05, 0.95),
        # The observable taker-flow slope is only a reduced-form anchor.  The
        # mapping below is dimensionless and deliberately bounded.
        orderflow_impact=_finite(
            empirical.get("orderflow_impact", abs(empirical["orderflow_slope"]))
            * (baseline["orderflow_impact"] / 0.004),
            0.00001,
            0.02,
        ),
        microstructure_noise=_finite(
            0.18 * empirical["median_log_high_low_range"] * (baseline["microstructure_noise"] / 0.0008),
            0.00005,
            0.01,
        ),
        spread_bps=profile.protocol_priors["spread_bps"] * (baseline["spread_bps"] / 10.0),
        fee_bps=profile.protocol_priors["fee_bps"] * (baseline["fee_bps"] / 2.0),
        liquidity=profile.protocol_priors["liquidity_index"] * (baseline["liquidity"] / 20.0),
        fundamental_price_coupling=profile.protocol_priors["fundamental_price_coupling"],
        permanent_event_probability=profile.protocol_priors["permanent_event_probability"],
        calibration_member=float(symbol_names.index(member_name)),
        calibration_shock_scale=shock_scale,
    )
    for regime, weight in empirical["regime_weights"].items():
        baseline[f"regime_weight_{regime}"] = _finite(weight, 0.001, 0.997)
    transition = empirical.get("regime_transition_matrix", {})
    for source in ("balanced", "directional", "volatile", "reversal"):
        for target in ("balanced", "directional", "volatile", "reversal"):
            if source in transition and target in transition[source]:
                baseline[f"regime_transition_{source}_{target}"] = _finite(
                    transition[source][target], 0.000001, 0.999999
                )

    # Deterministic task interventions are applied after empirical anchoring.
    if role == "probe_transfer":
        baseline["price_volatility"] *= 1.25
        baseline["spread_bps"] *= 1.20
        baseline["liquidity"] *= 0.85
    elif role == "stress":
        baseline["price_volatility"] *= 1.80
        baseline["fundamental_volatility"] *= 1.50
        baseline["spread_bps"] *= 1.75
        baseline["liquidity"] *= 0.45
        baseline["jump_intensity"] = min(0.35, baseline["jump_intensity"] * 2.2)
        baseline["jump_scale"] = min(0.22, max(0.01, baseline["jump_scale"] * 1.6))
        baseline["volatility_of_volatility"] = min(0.80, baseline["volatility_of_volatility"] * 1.35)
        baseline["microstructure_noise"] = min(0.01, baseline["microstructure_noise"] * 1.8)
    elif role == "shortcut_control":
        baseline["signal_noise"] *= 1.15

    origins: Dict[str, str] = {}
    empirical_keys = {
        "price_drift", "price_volatility", "volatility_persistence", "volatility_of_volatility",
        "regime_persistence", "jump_intensity", "jump_scale", "jump_tail_df",
        "jump_aftershock_decay", "impact_decay", "orderflow_impact", "microstructure_noise",
    }
    prior_keys = {
        "fundamental_volatility", "spread_bps", "fee_bps", "liquidity",
        "fundamental_price_coupling", "permanent_event_probability",
    }
    for key in baseline:
        if key.startswith("regime_weight_") or key.startswith("regime_transition_") or key in empirical_keys:
            origins[key] = "empirical_anchor_then_designed_transform"
        elif key in prior_keys:
            origins[key] = "fixed_protocol_prior_then_designed_transform"
        elif key == "calibration_member":
            origins[key] = "cross_sectional_calibration_draw"
        elif key == "calibration_shock_scale":
            origins[key] = "fit_only_indirect_inference"
        else:
            origins[key] = "designed_task_intervention"
    return {key: round(float(value), 8) for key, value in baseline.items()}, origins


def generate_calibrated_scenarios(
    profile: CalibrationProfile,
    instances_per_role: int,
    seed: int,
    horizon: int = 24,
) -> Tuple[List[ScenarioSpec], Dict[str, str]]:
    if instances_per_role < 1:
        raise ValueError("instances_per_role must be positive")
    master = random.Random(seed)
    shock_scale = _fit_shock_scale(profile, horizon)
    scenarios: List[ScenarioSpec] = []
    origin_ledger: Dict[str, str] = {}
    for family in FAMILIES:
        for layer in LAYERS:
            for role in ROLES:
                for index in range(instances_per_role):
                    local_seed = master.randrange(1, 2**31 - 1)
                    rng = random.Random(local_seed)
                    params, origins = _calibrated_params(profile, family, role, rng, horizon, shock_scale)
                    params["initial_gap"] *= rng.choice((-1.0, 1.0))
                    split, hidden = _role_split(role)
                    identity = f"{profile.calibration_id}|{family}|{layer}|{role}|{index}|{local_seed}"
                    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
                    scenarios.append(
                        ScenarioSpec(
                            episode_id=f"bcal-{family}-{layer}-{role}-{index:03d}-{digest[:10]}",
                            family_id=family,
                            role=role,
                            layer=layer,
                            variant=_role_variant(role),
                            split=split,
                            hidden=hidden,
                            horizon=horizon,
                            initial_price=round(rng.uniform(80.0, 120.0), 4),
                            mechanism_params=params,
                            opponent_mix=_opponent_mix(family, role),
                            risk_contract=_risk_contract(family, role),
                            observation_mapping_seed=rng.randrange(1, 2**31 - 1),
                            environment_seed=rng.randrange(1, 2**31 - 1),
                            evolution_mechanisms=list(ROLE_MECHANISMS[role]),
                            scoring_spec={
                                "violation_penalty": 0.01,
                                "drawdown_excess_penalty": 2.0,
                                "valuation": 1.0 if layer == "endogenous" else 0.0,
                            },
                            market_source=MarketSourceSpec(
                                kind="calibrated_procedural",
                                clock="step",
                                calibration_id=profile.calibration_id,
                            ),
                        )
                    )
                    for key, value in origins.items():
                        previous = origin_ledger.setdefault(key, value)
                        if previous != value:
                            raise ValueError(f"inconsistent parameter origin for {key}")
    return scenarios, origin_ledger


def generate_sharded_streams(
    scenarios: Sequence[ScenarioSpec], instances_per_role: int, seed: int
) -> List[StreamSpec]:
    """Rotate every role instance through executable longitudinal streams."""
    grouped: DefaultDict[Tuple[str, str, str], List[ScenarioSpec]] = defaultdict(list)
    for scenario in scenarios:
        grouped[(scenario.family_id, scenario.layer, scenario.role)].append(scenario)
    for values in grouped.values():
        values.sort(key=lambda item: item.episode_id)
    streams: List[StreamSpec] = []
    for shard in range(instances_per_role):
        subset: List[ScenarioSpec] = []
        for family in FAMILIES:
            for layer in LAYERS:
                for role in ROLES:
                    values = grouped[(family, layer, role)]
                    if len(values) != instances_per_role:
                        raise ValueError(f"incomplete calibrated pool for {family}/{layer}/{role}")
                    if role == "learn_near":
                        offsets = (0, 1, 2, 3, 4)
                    elif role == "update":
                        offsets = (0, 1)
                    else:
                        offsets = (0,)
                    subset.extend(values[(shard + offset) % instances_per_role] for offset in offsets)
        shard_streams = StreamGenerator(seed=seed + shard * 7919).generate(subset)
        for stream in shard_streams:
            payload = stream.to_dict()
            payload["stream_id"] = f"{stream.stream_id}-shard-{shard:03d}"
            streams.append(StreamSpec.from_dict(payload))
    validate_streams(
        streams,
        [scenario.episode_id for scenario in scenarios],
        [scenario.episode_id for scenario in scenarios if scenario.hidden],
    )
    event_ids = {event.episode_id for stream in streams for event in stream.events}
    probe_ids = {
        episode_id for stream in streams for checkpoint in stream.checkpoints for episode_id in checkpoint.probe_episode_ids
    }
    expected_events = {item.episode_id for item in scenarios if item.role in {"learn_near", "update"}}
    expected_probes = {item.episode_id for item in scenarios if item.hidden}
    if event_ids != expected_events or probe_ids != expected_probes:
        raise ValueError("sharded streams do not exhaust the calibrated task pool")
    return streams


def generate_binance_calibrated_dataset(
    profile: CalibrationProfile,
    holdout_validation: Mapping[str, Any],
    output_dir: Path,
    instances_per_role: int = 50,
    seed: int = 7,
    stream_seed: int = 17,
    horizon: int = 24,
    bootstrap_replicates: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    scenarios, origin_ledger = generate_calibrated_scenarios(
        profile, instances_per_role=instances_per_role, seed=seed, horizon=horizon
    )
    streams = generate_sharded_streams(scenarios, instances_per_role, stream_seed)
    expected_schedule = list(range(6))
    expected_checkpoint_ids = [f"K{index}" for index in expected_schedule]
    for stream in streams:
        if (
            len(stream.events) != 5
            or [item.checkpoint_id for item in stream.checkpoints] != expected_checkpoint_ids
            or [item.after_event for item in stream.checkpoints] != expected_schedule
        ):
            raise ValueError(f"invalid five-update checkpoint schedule in {stream.stream_id}")
    profile_payload = (json.dumps(profile.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    enriched_holdout = dict(holdout_validation)
    enriched_holdout["synthetic_track_audit"] = _synthetic_realism_audit(profile, scenarios, holdout_validation)
    holdout_payload = (json.dumps(enriched_holdout, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ledger = {
        "schema_version": "evolens-calibrated-parameter-ledger-v1",
        "transform_version": TRANSFORM_VERSION,
        "calibration_id": profile.calibration_id,
        "parameter_origins": dict(sorted(origin_ledger.items())),
        "claim_boundary": (
            "Binance Spot klines constrain selected observable moments and reduced-form response proxies; "
            "trajectories, fills, latent fundamentals, spreads, opponents, and counterfactual impact remain synthetic."
        ),
    }
    ledger_payload = (json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    regime_payload = (json.dumps(profile.regime_model, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    projection_payload = (json.dumps(profile.local_projection, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    response_payload = (json.dumps(_orderflow_response_validation(profile), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    bootstrap_payload = b"".join(
        (json.dumps(dict(item), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for item in bootstrap_replicates
    )
    referenced = {
        event.episode_id for stream in streams for event in stream.events
    } | {
        episode_id for stream in streams for checkpoint in stream.checkpoints for episode_id in checkpoint.probe_episode_ids
    }
    return write_dataset(
        output_dir,
        scenarios,
        streams,
        {
            "seed": seed,
            "stream_seed": stream_seed,
            "horizon": horizon,
            "instances_per_role": instances_per_role,
            "stream_shards": instances_per_role,
            "evolution_updates_per_stream": 5,
            "checkpoint_schedule": ["K0", "K1", "K2", "K3", "K4", "K5"],
            "paper_decisions_per_episode": 3,
            "transform_version": TRANSFORM_VERSION,
        },
        schema_version=CALIBRATED_TRACK_SCHEMA,
        track_spec={
            "kind": "binance_calibrated_semi_synthetic",
            "calibration_id": profile.calibration_id,
            "calibration_profile": "resources/calibration_profile.json",
            "locked_holdout_validation": "resources/locked_holdout_validation.json",
            "parameter_ledger": "resources/parameter_origin_ledger.json",
            "regime_model": "resources/regime_model.json",
            "orderflow_local_projection": "resources/orderflow_local_projection.json",
            "bootstrap_replicates": "resources/bootstrap_replicates_summary.jsonl",
            "orderflow_response_validation": "resources/orderflow_response_validation.json",
            "path_type": "semi_synthetic_v2_rng_with_empirical_innovations",
            "leaderboard_isolation": "must_not_pool_with_replay_or_uncalibrated_procedural_tracks",
        },
        resource_payloads={
            "resources/calibration_profile.json": profile_payload,
            "resources/locked_holdout_validation.json": holdout_payload,
            "resources/parameter_origin_ledger.json": ledger_payload,
            "resources/regime_model.json": regime_payload,
            "resources/orderflow_local_projection.json": projection_payload,
            "resources/bootstrap_replicates_summary.jsonl": bootstrap_payload,
            "resources/orderflow_response_validation.json": response_payload,
        },
        extra_manifest={
            "coverage_audit": {
                "episode_count": len(scenarios),
                "referenced_episode_count": len(referenced),
                "episode_coverage": len(referenced) / len(scenarios),
                "event_episode_count": len({event.episode_id for stream in streams for event in stream.events}),
                "probe_episode_count": len({episode_id for stream in streams for checkpoint in stream.checkpoints for episode_id in checkpoint.probe_episode_ids}),
            }
        },
    )
