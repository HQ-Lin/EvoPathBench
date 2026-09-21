"""Procedural task-pool generation and validation."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import RiskContract, ScenarioSpec, StreamSpec
from .mechanisms import EVOLUTION_MECHANISMS, MECHANISM_DESCRIPTIONS, ROLE_MECHANISMS

SCHEMA_VERSION = "evolens-market-v2"
FAMILIES: Tuple[str, ...] = (
    "trend",
    "mean_reversion",
    "liquidity",
    "event_jump",
    "opponent_reflexivity",
    "risk_contract",
)
ROLES: Tuple[str, ...] = (
    "learn_near",
    "probe_near",
    "probe_transfer",
    "retention_anchor",
    "update",
    "probe_update",
    "stress",
    "shortcut_control",
)
LAYERS: Tuple[str, ...] = ("exogenous", "endogenous")


def _base_params(family: str, horizon: int, price_process_version: int = 2) -> Dict[str, float]:
    if price_process_version == 1:
        common = {
            "fundamental_drift": 0.0,
            "fundamental_volatility": 0.002,
            "price_drift": 0.0,
            "price_volatility": 0.007,
            "spread_bps": 10.0,
            "liquidity": 20.0,
            "fee_bps": 2.0,
            "signal_noise": 0.003,
            "jump_step": float(horizon // 2),
            "jump_size": 0.0,
            "initial_gap": 0.0,
        }
        if family == "trend":
            common.update(price_drift=0.0035, fundamental_drift=0.0010, price_volatility=0.005)
        elif family == "mean_reversion":
            common.update(initial_gap=0.035, price_volatility=0.012, fundamental_volatility=0.001)
        elif family == "liquidity":
            common.update(spread_bps=28.0, liquidity=5.0, fee_bps=5.0, price_volatility=0.009)
        elif family == "event_jump":
            common.update(jump_size=0.065, price_volatility=0.006, fundamental_volatility=0.0025)
        elif family == "opponent_reflexivity":
            common.update(spread_bps=13.0, liquidity=14.0, price_volatility=0.008)
        elif family == "risk_contract":
            common.update(price_volatility=0.014, spread_bps=16.0, liquidity=10.0, jump_size=0.035)
        else:
            raise ValueError(f"unknown family: {family}")
        return common
    if price_process_version != 2:
        raise ValueError(f"unsupported price process version: {price_process_version}")
    common = {
        # Version 2 replaces a single homoskedastic geometric path with a
        # reproducible latent-regime, stochastic-volatility, random-event
        # process.  Keeping the version in each episode lets the simulator
        # replay published V1 datasets with their original dynamics.
        "price_process_version": 2.0,
        "fundamental_drift": 0.0,
        "fundamental_volatility": 0.002,
        "price_drift": 0.0,
        "price_volatility": 0.007,
        "spread_bps": 10.0,
        "liquidity": 20.0,
        "fee_bps": 2.0,
        "signal_noise": 0.003,
        "initial_gap": 0.0,
        "regime_persistence": 0.82,
        "volatility_persistence": 0.88,
        "volatility_of_volatility": 0.18,
        "jump_intensity": 0.018,
        "jump_scale": 0.018,
        "jump_tail_df": 4.0,
        "jump_aftershock_decay": 0.68,
        "permanent_event_probability": 0.55,
        "mispricing_persistence": 0.75,
        "microstructure_noise": 0.0008,
        "orderflow_impact": 0.004,
        "impact_decay": 0.55,
        "fundamental_price_coupling": 0.16,
    }
    if family == "trend":
        common.update(
            price_drift=0.0022,
            fundamental_drift=0.0008,
            price_volatility=0.0055,
            regime_persistence=0.88,
            mispricing_persistence=0.94,
        )
    elif family == "mean_reversion":
        common.update(
            initial_gap=0.025,
            price_volatility=0.010,
            fundamental_volatility=0.0012,
            mispricing_persistence=0.55,
            fundamental_price_coupling=0.28,
        )
    elif family == "liquidity":
        common.update(
            spread_bps=24.0,
            liquidity=7.0,
            fee_bps=5.0,
            price_volatility=0.009,
            microstructure_noise=0.0022,
            jump_intensity=0.028,
        )
    elif family == "event_jump":
        common.update(
            jump_intensity=0.075,
            jump_scale=0.035,
            permanent_event_probability=0.65,
            price_volatility=0.0065,
            fundamental_volatility=0.0025,
            volatility_of_volatility=0.24,
        )
    elif family == "opponent_reflexivity":
        common.update(
            spread_bps=13.0,
            liquidity=14.0,
            price_volatility=0.008,
            mispricing_persistence=0.86,
            orderflow_impact=0.009,
            impact_decay=0.72,
        )
    elif family == "risk_contract":
        common.update(
            price_volatility=0.012,
            spread_bps=16.0,
            liquidity=10.0,
            jump_intensity=0.050,
            jump_scale=0.030,
            volatility_of_volatility=0.28,
        )
    else:
        raise ValueError(f"unknown family: {family}")
    return common


def _risk_contract(family: str, role: str) -> RiskContract:
    if family == "risk_contract":
        return RiskContract(
            initial_cash=10_000.0,
            initial_position=20,
            max_abs_position=30,
            max_order_size=5,
            max_turnover=1.25,
            max_drawdown=0.08,
            allow_short=False,
        )
    if role == "stress":
        return RiskContract(max_abs_position=40, max_order_size=6, max_turnover=2.5, max_drawdown=0.12)
    return RiskContract()


def _opponent_mix(family: str, role: str) -> Dict[str, int]:
    mix = {"fundamental": 1, "momentum": 1, "contrarian": 1, "market_maker": 1, "noise": 1}
    if family == "opponent_reflexivity":
        mix = {"fundamental": 1, "momentum": 2, "contrarian": 0, "market_maker": 1, "noise": 1}
    if role == "probe_transfer":
        mix = {"fundamental": 2, "momentum": 0, "contrarian": 1, "market_maker": 1, "noise": 1}
    if role == "stress":
        mix = {"fundamental": 0, "momentum": 2, "contrarian": 0, "market_maker": 1, "noise": 2}
    return mix


def _role_variant(role: str) -> str:
    if role in {"update", "probe_update"}:
        return "reversed"
    if role == "probe_transfer":
        return "transfer"
    if role == "stress":
        return "stress"
    if role == "shortcut_control":
        return "surface_randomized"
    return "standard"


def _role_split(role: str) -> Tuple[str, bool]:
    if role in {"learn_near", "update"}:
        return "train", False
    if role == "probe_near":
        return "dev", True
    return "test", True


class ScenarioGenerator:
    def __init__(self, seed: int = 7, horizon: int = 24, price_process_version: int = 2) -> None:
        if price_process_version not in {1, 2}:
            raise ValueError("price_process_version must be 1 or 2")
        self.seed = seed
        self.horizon = horizon
        self.price_process_version = price_process_version

    def generate(
        self,
        instances_per_role: int = 3,
        families: Sequence[str] = FAMILIES,
        layers: Sequence[str] = LAYERS,
    ) -> List[ScenarioSpec]:
        if instances_per_role < 1:
            raise ValueError("instances_per_role must be positive")
        scenarios: List[ScenarioSpec] = []
        master_rng = random.Random(self.seed)
        for family in families:
            if family not in FAMILIES:
                raise ValueError(f"unsupported family: {family}")
            for layer in layers:
                if layer not in LAYERS:
                    raise ValueError(f"unsupported layer: {layer}")
                for role in ROLES:
                    for index in range(instances_per_role):
                        local_seed = master_rng.randrange(1, 2**31 - 1)
                        rng = random.Random(local_seed)
                        params = _base_params(family, self.horizon, self.price_process_version)
                        params["price_drift"] *= rng.uniform(0.80, 1.20)
                        params["price_volatility"] *= rng.uniform(0.85, 1.20)
                        params["fundamental_volatility"] *= rng.uniform(0.85, 1.15)
                        if self.price_process_version >= 2:
                            params["spread_bps"] *= rng.uniform(0.80, 1.25)
                            params["liquidity"] *= rng.uniform(0.75, 1.25)
                            params["jump_intensity"] *= rng.uniform(0.70, 1.30)
                            params["jump_scale"] *= rng.uniform(0.75, 1.30)
                            params["volatility_of_volatility"] *= rng.uniform(0.80, 1.20)
                            params["regime_persistence"] = min(
                                0.96,
                                max(0.55, params["regime_persistence"] + rng.uniform(-0.05, 0.05)),
                            )
                            params["mispricing_persistence"] = min(
                                0.98,
                                max(0.35, params["mispricing_persistence"] + rng.uniform(-0.06, 0.06)),
                            )
                        params["initial_gap"] *= rng.choice((-1.0, 1.0))
                        if role == "probe_transfer":
                            params["price_volatility"] *= 1.25
                            params["spread_bps"] *= 1.20
                            params["liquidity"] *= 0.85
                        elif role == "stress":
                            params["price_volatility"] *= 1.80
                            params["fundamental_volatility"] *= 1.50
                            params["spread_bps"] *= 1.75
                            params["liquidity"] *= 0.45
                            if self.price_process_version >= 2:
                                params["jump_intensity"] = min(0.35, params["jump_intensity"] * 2.2)
                                params["jump_scale"] = max(0.05, params["jump_scale"] * 1.6)
                                params["volatility_of_volatility"] *= 1.35
                                params["microstructure_noise"] *= 1.8
                            else:
                                params["jump_size"] = max(0.08, abs(params["jump_size"]) * 1.5)
                        elif role == "shortcut_control":
                            params["signal_noise"] *= 1.15
                        split, hidden = _role_split(role)
                        episode_id = f"{family}-{layer}-{role}-{index:02d}-{local_seed:08x}"
                        scenarios.append(
                            ScenarioSpec(
                                episode_id=episode_id,
                                family_id=family,
                                role=role,
                                layer=layer,
                                variant=_role_variant(role),
                                split=split,
                                hidden=hidden,
                                horizon=self.horizon,
                                initial_price=round(rng.uniform(80.0, 120.0), 4),
                                mechanism_params={key: round(float(value), 8) for key, value in params.items()},
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
                            )
                        )
        return scenarios


def scenario_index(scenarios: Iterable[ScenarioSpec]) -> Dict[str, ScenarioSpec]:
    result: Dict[str, ScenarioSpec] = {}
    for scenario in scenarios:
        if scenario.episode_id in result:
            raise ValueError(f"duplicate episode_id: {scenario.episode_id}")
        result[scenario.episode_id] = scenario
    return result


def validate_scenarios(scenarios: Sequence[ScenarioSpec]) -> Dict[str, Any]:
    index = scenario_index(scenarios)
    counts: Counter = Counter()
    errors: List[str] = []
    for scenario in scenarios:
        counts[(scenario.family_id, scenario.layer, scenario.role)] += 1
        contract = scenario.risk_contract
        if scenario.horizon < 5:
            errors.append(f"{scenario.episode_id}: horizon below 5")
        if scenario.initial_price <= 0:
            errors.append(f"{scenario.episode_id}: non-positive initial price")
        if contract.initial_position > contract.max_abs_position:
            errors.append(f"{scenario.episode_id}: infeasible initial position")
        if contract.max_order_size <= 0 or contract.max_turnover <= 0:
            errors.append(f"{scenario.episode_id}: invalid risk contract")
        if scenario.layer == "endogenous" and sum(scenario.opponent_mix.values()) < 3:
            errors.append(f"{scenario.episode_id}: too few opponents")
        if scenario.mechanism_params.get("liquidity", 0.0) <= 0:
            errors.append(f"{scenario.episode_id}: non-positive liquidity")
        source_kind = scenario.market_source.kind
        if source_kind == "binance_replay":
            if scenario.layer != "exogenous":
                errors.append(f"{scenario.episode_id}: Binance replay only supports exogenous layer")
            if not scenario.market_source.resource_id:
                errors.append(f"{scenario.episode_id}: missing replay resource_id")
            if scenario.market_source.clock != "bar_close_next_open":
                errors.append(f"{scenario.episode_id}: invalid replay clock")
        elif source_kind == "calibrated_procedural":
            if not scenario.market_source.calibration_id:
                errors.append(f"{scenario.episode_id}: missing calibration_id")
            if scenario.market_source.resource_id:
                errors.append(f"{scenario.episode_id}: calibrated track must not point to a replay window")
            if scenario.market_source.clock != "step":
                errors.append(f"{scenario.episode_id}: invalid calibrated simulator clock")
            if scenario.mechanism_params.get("price_process_version") != 2.0:
                errors.append(f"{scenario.episode_id}: calibrated track requires V2 process")
        elif source_kind != "procedural":
            errors.append(f"{scenario.episode_id}: unknown market source kind {source_kind}")
        if source_kind in {"procedural", "calibrated_procedural"} and scenario.mechanism_params.get("price_process_version", 1.0) >= 2.0:
            bounded = (
                "regime_persistence",
                "volatility_persistence",
                "jump_intensity",
                "permanent_event_probability",
                "mispricing_persistence",
                "impact_decay",
                "fundamental_price_coupling",
            )
            for key in bounded:
                value = scenario.mechanism_params.get(key, -1.0)
                if not 0.0 <= value <= 1.0:
                    errors.append(f"{scenario.episode_id}: {key} outside [0, 1]")
            for key in ("jump_scale", "jump_tail_df", "volatility_of_volatility"):
                if scenario.mechanism_params.get(key, 0.0) <= 0.0:
                    errors.append(f"{scenario.episode_id}: non-positive {key}")
            if source_kind == "calibrated_procedural":
                names = ("balanced", "directional", "volatile", "reversal")
                transition_keys = [
                    f"regime_transition_{source}_{target}"
                    for source in names for target in names
                ]
                if any(key in scenario.mechanism_params for key in transition_keys):
                    if not all(key in scenario.mechanism_params for key in transition_keys):
                        errors.append(f"{scenario.episode_id}: incomplete regime transition matrix")
                    for source in names:
                        total = sum(
                            scenario.mechanism_params.get(f"regime_transition_{source}_{target}", 0.0)
                            for target in names
                        )
                        if abs(total - 1.0) > 1e-5:
                            errors.append(f"{scenario.episode_id}: transition row {source} does not sum to one")
        unknown_mechanisms = set(scenario.evolution_mechanisms) - set(EVOLUTION_MECHANISMS)
        if unknown_mechanisms:
            errors.append(f"{scenario.episode_id}: unknown evolution mechanisms {sorted(unknown_mechanisms)}")
        if set(scenario.evolution_mechanisms) != set(ROLE_MECHANISMS[scenario.role]):
            errors.append(f"{scenario.episode_id}: mechanism labels do not match role {scenario.role}")
    present_pairs = {(scenario.family_id, scenario.layer) for scenario in scenarios}
    for family, layer in sorted(present_pairs):
        for role in ROLES:
            if counts[(family, layer, role)] == 0:
                errors.append(f"missing role {role} for {family}/{layer}")
    if errors:
        raise ValueError("scenario validation failed:\n" + "\n".join(errors))
    return {
        "status": "ok",
        "scenario_count": len(index),
        "families": sorted({scenario.family_id for scenario in scenarios}),
        "layers": sorted({scenario.layer for scenario in scenarios}),
        "roles": list(ROLES),
        "evolution_mechanisms": list(EVOLUTION_MECHANISMS),
        "counts": {"/".join(key): value for key, value in sorted(counts.items())},
    }


def _jsonl_bytes(items: Iterable[Dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for item in items
    )


def write_dataset(
    output_dir: Path,
    scenarios: Sequence[ScenarioSpec],
    streams: Sequence[StreamSpec],
    generation_config: Dict[str, Any],
    *,
    schema_version: str = "",
    track_spec: Optional[Dict[str, Any]] = None,
    resource_payloads: Optional[Dict[str, bytes]] = None,
    extra_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not schema_version:
        process_versions = {
            int(scenario.mechanism_params.get("price_process_version", 1.0)) for scenario in scenarios
        }
        if len(process_versions) != 1:
            raise ValueError(f"mixed price process versions are not supported: {sorted(process_versions)}")
        process_version = next(iter(process_versions))
        schema_version = f"evolens-market-v{process_version}"
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_payload = _jsonl_bytes(scenario.to_dict() for scenario in scenarios)
    stream_payload = _jsonl_bytes(stream.to_dict() for stream in streams)
    (output_dir / "episodes.jsonl").write_bytes(scenario_payload)
    (output_dir / "streams.jsonl").write_bytes(stream_payload)
    hashes = {
        "episodes.jsonl": hashlib.sha256(scenario_payload).hexdigest(),
        "streams.jsonl": hashlib.sha256(stream_payload).hexdigest(),
    }
    for relative, payload in sorted((resource_payloads or {}).items()):
        resource_path = Path(relative)
        if resource_path.is_absolute() or ".." in resource_path.parts:
            raise ValueError(f"unsafe dataset resource path: {relative}")
        destination = output_dir / resource_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        hashes[relative] = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema_version": schema_version,
        "generation_config": generation_config,
        "scenario_count": len(scenarios),
        "stream_count": len(streams),
        "families": sorted({scenario.family_id for scenario in scenarios}),
        "roles": list(ROLES),
        "evolution_mechanisms": {
            mechanism: MECHANISM_DESCRIPTIONS[mechanism] for mechanism in EVOLUTION_MECHANISMS
        },
        "layers": sorted({scenario.layer for scenario in scenarios}),
        "sha256": hashes,
    }
    if track_spec is not None:
        manifest["track"] = track_spec
    if extra_manifest:
        for key, value in extra_manifest.items():
            if key in manifest:
                raise ValueError(f"extra manifest key collides with reserved key: {key}")
            manifest[key] = value
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_dataset(dataset_dir: Path) -> Tuple[List[ScenarioSpec], List[StreamSpec], Dict[str, Any]]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    required_hashes = {"episodes.jsonl", "streams.jsonl"}
    missing_hashes = required_hashes - set(manifest.get("sha256", {}))
    if missing_hashes:
        raise ValueError(f"dataset manifest is missing required hashes: {sorted(missing_hashes)}")
    root = dataset_dir.resolve()
    for relative, expected_hash in manifest["sha256"].items():
        resource = Path(relative)
        if resource.is_absolute() or ".." in resource.parts:
            raise ValueError(f"unsafe manifest resource path: {relative}")
        path = (dataset_dir / resource).resolve()
        if root not in path.parents:
            raise ValueError(f"manifest resource escapes dataset: {relative}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"dataset hash mismatch for {relative}: expected={expected_hash}, actual={actual_hash}"
            )
    episode_bytes = (dataset_dir / "episodes.jsonl").read_bytes()
    stream_bytes = (dataset_dir / "streams.jsonl").read_bytes()
    scenarios = [ScenarioSpec.from_dict(json.loads(line)) for line in episode_bytes.decode("utf-8").splitlines() if line]
    streams = [StreamSpec.from_dict(json.loads(line)) for line in stream_bytes.decode("utf-8").splitlines() if line]
    if len(scenarios) != manifest["scenario_count"] or len(streams) != manifest["stream_count"]:
        raise ValueError("dataset manifest count mismatch")
    validate_scenarios(scenarios)
    from .streams import validate_streams

    validate_streams(
        streams,
        [scenario.episode_id for scenario in scenarios],
        [scenario.episode_id for scenario in scenarios if scenario.hidden],
    )
    return scenarios, streams, manifest
