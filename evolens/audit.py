"""Structural and simulation-backed dataset checks."""
from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from .agents import OracleFamilyAgent, StaticTradingAgent
from .dataset import validate_scenarios
from .market import MarketSimulator
from .models import ScenarioSpec, StreamSpec
from .streams import validate_streams


def audit_dataset(
    scenarios: Sequence[ScenarioSpec],
    streams: Sequence[StreamSpec],
    sample_size: int = 24,
    market_simulator: Optional[MarketSimulator] = None,
) -> Dict[str, Any]:
    structural = validate_scenarios(scenarios)
    stream_audit = validate_streams(
        streams,
        [scenario.episode_id for scenario in scenarios],
        [scenario.episode_id for scenario in scenarios if scenario.hidden],
    )
    market = market_simulator or MarketSimulator()
    sample = sorted(scenarios, key=lambda item: item.episode_id)[: max(1, min(sample_size, len(scenarios)))]
    oracle_scores: List[float] = []
    random_scores: List[float] = []
    conservation_errors: List[float] = []
    deterministic_mismatches: List[str] = []
    finite_failures: List[str] = []
    event_counts: List[float] = []
    event_counts_by_family: Counter = Counter()
    event_jump_without_event: List[str] = []
    regime_switch_counts: List[float] = []
    volatility_multipliers: List[float] = []
    for scenario in sample:
        oracle_a = OracleFamilyAgent(mode="conservative", seed=11)
        oracle_b = OracleFamilyAgent(mode="conservative", seed=11)
        result_a = market.run(scenario, oracle_a, execution_seed=5)
        result_b = market.run(scenario, oracle_b, execution_seed=5)
        if result_a.to_dict() != result_b.to_dict():
            deterministic_mismatches.append(scenario.episode_id)
        random_agent = StaticTradingAgent(mode="random", seed=23)
        random_result = market.run(scenario, random_agent, execution_seed=5)
        oracle_scores.append(result_a.score)
        random_scores.append(random_result.score)
        conservation_errors.append(abs(result_a.market_diagnostics.get("cash_conservation_error", 0.0)))
        conservation_errors.append(abs(result_a.market_diagnostics.get("asset_conservation_error", 0.0)))
        if (
            scenario.market_source.kind in {"procedural", "calibrated_procedural"}
            and result_a.market_diagnostics.get("price_process_version", 1.0) >= 2.0
        ):
            event_count = result_a.market_diagnostics.get("event_count", 0.0)
            event_counts.append(event_count)
            event_counts_by_family[scenario.family_id] += int(event_count)
            if scenario.family_id == "event_jump" and event_count <= 0.0:
                event_jump_without_event.append(scenario.episode_id)
            regime_switch_counts.append(result_a.market_diagnostics.get("regime_switch_count", 0.0))
            volatility_multipliers.append(
                result_a.market_diagnostics.get("max_volatility_multiplier", 1.0)
            )
        numeric = [
            result_a.score,
            result_a.final_wealth,
            result_a.max_drawdown,
            result_a.turnover,
        ]
        if not all(math.isfinite(value) for value in numeric):
            finite_failures.append(scenario.episode_id)
    if deterministic_mismatches:
        raise ValueError(f"non-deterministic scripted reruns: {deterministic_mismatches}")
    if finite_failures:
        raise ValueError(f"non-finite market outputs: {finite_failures}")
    max_conservation_error = max(conservation_errors) if conservation_errors else 0.0
    if max_conservation_error > 1e-6:
        raise ValueError(f"market conservation error exceeds tolerance: {max_conservation_error}")
    oracle_mean = statistics.fmean(oracle_scores)
    random_mean = statistics.fmean(random_scores)
    warnings: List[str] = []
    if oracle_mean <= random_mean:
        warnings.append(
            "Diagnostic oracle did not outperform the random baseline on this small structural sample; "
            "inspect family calibration before using difficulty claims."
        )
    if event_jump_without_event:
        warnings.append(
            "V2 event_jump episodes without an event were found: "
            + ", ".join(event_jump_without_event[:8])
        )
    replay_count = sum(1 for scenario in sample if scenario.market_source.kind == "binance_replay")
    return {
        "status": "ok_with_warnings" if warnings else "ok",
        "structural": structural,
        "streams": stream_audit,
        "simulation": {
            "sample_size": len(sample),
            "deterministic_rerun": True,
            "max_conservation_error": max_conservation_error,
            "all_outputs_finite": True,
            "diagnostic_oracle_mean_score": oracle_mean,
            "random_mean_score": random_mean,
            "oracle_minus_random": oracle_mean - random_mean,
            "episodes_with_events": sum(1 for count in event_counts if count > 0.0),
            "total_random_events": int(sum(event_counts)),
            "event_counts_by_family": dict(sorted(event_counts_by_family.items())),
            "event_jump_without_event": event_jump_without_event,
            "mean_regime_switches": statistics.fmean(regime_switch_counts) if regime_switch_counts else 0.0,
            "max_volatility_multiplier": max(volatility_multipliers) if volatility_multipliers else 1.0,
            "binance_replay_episodes": replay_count,
            "replay_clock": "bar_close_next_open" if replay_count else None,
        },
        "warnings": warnings,
        "limitations": [
            "The scripted oracle is a data diagnostic, not an evaluated self-evolving agent.",
            "A positive oracle gap demonstrates basic separability, not realism or absence of exploitable shortcuts.",
            "Mechanism labels identify intended diagnostic contrasts; they do not by themselves prove an agent's internal causal mechanism.",
        ],
    }
