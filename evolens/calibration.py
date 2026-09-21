"""Binance-calibrated semi-synthetic market profiles.

The estimator consumes checksummed monthly Binance Spot kline archives.  It
fits only quantities that are observable from OHLCV/trade-count/taker-volume
bars and records protocol priors separately for quantities that klines cannot
identify.  Evaluation never contacts Binance.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .market_data import FrozenBar, _iter_archive, sha256_file
from .advanced_calibration import (
    evaluate_regime_model,
    fit_advanced_calibration,
    fit_local_projections,
)


CALIBRATION_SCHEMA = "evolens-binance-calibration-v2"
CALIBRATED_TRACK_SCHEMA = "evolens-market-binance-calibrated-v2"
SUPPORTED_CALIBRATION_SCHEMAS = {"evolens-binance-calibration-v1", CALIBRATION_SCHEMA}


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    return _quantile_sorted(sorted(values), probability)


def _quantile_sorted(ordered: Sequence[float], probability: float) -> float:
    if not ordered:
        return 0.0
    position = min(1.0, max(0.0, probability)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _corr(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 3:
        return 0.0
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - mean_left) ** 2 for x in left) * sum((y - mean_right) ** 2 for y in right)
    )
    return numerator / max(denominator, 1e-15)


def _slope(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 3:
        return 0.0
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    denominator = sum((item - mean_left) ** 2 for item in left)
    return sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right)) / max(denominator, 1e-15)


def _period(path: Path) -> str:
    parts = path.stem.rsplit("-", 2)
    if len(parts) < 3:
        raise ValueError(f"cannot infer period from Binance archive: {path.name}")
    return parts[-2] + "-" + parts[-1]


def _symbol(path: Path) -> str:
    return path.name.split("-", 1)[0]


def _aggregate(items: Sequence[FrozenBar]) -> Dict[str, float]:
    quote = sum(float(item.quote_volume) for item in items)
    taker = sum(float(item.taker_buy_quote_volume) for item in items)
    return {
        "open_time_us": float(items[0].open_time_us),
        "open": float(items[0].open),
        "high": max(float(item.high) for item in items),
        "low": min(float(item.low) for item in items),
        "close": float(items[-1].close),
        "quote_volume": quote,
        "trade_count": float(sum(item.trade_count for item in items)),
        "imbalance": 2.0 * taker / quote - 1.0 if quote > 0 else 0.0,
    }


def _read_series(paths: Sequence[Path], bar_size: int) -> Tuple[Dict[str, List[Dict[str, float]]], List[Dict[str, Any]]]:
    series: Dict[str, List[Dict[str, float]]] = {}
    sources: List[Dict[str, Any]] = []
    for path in sorted((Path(item) for item in paths), key=lambda item: item.name):
        iterator, source = _iter_archive(path, allow_truncated_intervals=True)
        if source["checksum_status"] != "verified":
            raise ValueError(f"calibration requires official checksum: {path.name}")
        if "acquisition" not in source:
            raise ValueError(f"calibration requires acquisition provenance: {path.name}")
        buffer: List[FrozenBar] = []
        count = 0
        used_count = 0
        gap_count = 0
        irregular_interval_count = 0
        discarded_partial_count = 0
        previous_open: Optional[int] = None
        duration = 60 * 1_000_000
        target = series.setdefault(source["symbol"], [])
        for bar in iterator:
            count += 1
            expected_tick = 1 if source["timestamp_unit"] == "us" else 1000
            if bar.close_time_us - bar.open_time_us != duration - expected_tick:
                discarded_partial_count += len(buffer)
                buffer = []
                irregular_interval_count += 1
                previous_open = bar.open_time_us
                continue
            if previous_open is not None and bar.open_time_us - previous_open != duration:
                discarded_partial_count += len(buffer)
                buffer = []
                gap_count += 1
            previous_open = bar.open_time_us
            buffer.append(bar)
            used_count += 1
            if len(buffer) == bar_size:
                target.append(_aggregate(buffer))
                buffer = []
        discarded_partial_count += len(buffer)
        source = dict(source)
        source["row_count"] = count
        source["used_complete_row_count"] = used_count
        source["aggregated_row_count"] = (used_count - discarded_partial_count) // bar_size
        source["quality_control"] = {
            "irregular_interval_rows_excluded": irregular_interval_count,
            "timestamp_gaps": gap_count,
            "complete_rows_discarded_at_gap_or_month_boundary": discarded_partial_count,
            "gap_policy": "do not interpolate; discard irregular-duration bar and any incomplete aggregation block",
        }
        sources.append(source)
    return series, sources


def _estimate(values: Sequence[Dict[str, float]]) -> Dict[str, Any]:
    closes = [item["close"] for item in values]
    returns = [math.log(current / previous) for previous, current in zip(closes, closes[1:])]
    if len(returns) < 100:
        raise ValueError("at least 100 aggregated returns are required per calibration slice")
    median = statistics.median(returns)
    mad = statistics.median(abs(item - median) for item in returns)
    robust_sigma = max(1e-12, 1.4826 * mad)
    jump_threshold = max(4.0 * robust_sigma, _quantile([abs(item) for item in returns], 0.995))
    jump_sizes = [abs(item) for item in returns if abs(item) >= jump_threshold]
    tail_ratios = [math.log(item / jump_threshold) for item in jump_sizes if item > jump_threshold]
    hill = statistics.fmean(tail_ratios) if tail_ratios else 0.25
    tail_df = min(12.0, max(2.2, 1.0 / max(hill, 1e-6)))

    abs_returns = [max(abs(item), robust_sigma * 0.05) for item in returns]
    log_abs = [math.log(item) for item in abs_returns]
    vol_persistence = min(0.98, max(0.05, _corr(log_abs[:-1], log_abs[1:])))
    vol_residuals = [right - vol_persistence * left for left, right in zip(log_abs[:-1], log_abs[1:])]
    vol_of_vol = min(0.80, max(0.03, statistics.pstdev(vol_residuals)))

    rolling_direction: List[float] = []
    regime_labels: List[str] = []
    for index, item in enumerate(returns):
        recent = returns[max(0, index - 5) : index + 1]
        direction = statistics.fmean(recent) / robust_sigma
        rolling_direction.append(direction)
        magnitude = abs(item) / robust_sigma
        if magnitude >= 1.75:
            regime_labels.append("volatile")
        elif abs(direction) >= 0.45:
            regime_labels.append("directional")
        elif index and item * returns[index - 1] < 0 and magnitude >= 0.55:
            regime_labels.append("reversal")
        else:
            regime_labels.append("balanced")
    regime_weights = {
        label: regime_labels.count(label) / len(regime_labels)
        for label in ("balanced", "directional", "volatile", "reversal")
    }
    regime_persistence = sum(a == b for a, b in zip(regime_labels[:-1], regime_labels[1:])) / max(1, len(regime_labels) - 1)

    imbalances = [item["imbalance"] for item in values[:-1]]
    next_returns = returns
    flow_slope = _slope(imbalances, next_returns)
    flow_corr = _corr(imbalances, next_returns)
    lag_correlations = []
    for lag in range(1, 7):
        lag_correlations.append(_corr(imbalances[:-lag], next_returns[lag:]))
    impact_decay = 0.50
    if abs(lag_correlations[0]) > 1e-6:
        ratios = [abs(item / lag_correlations[0]) for item in lag_correlations[1:] if item * lag_correlations[0] > 0]
        if ratios:
            impact_decay = min(0.95, max(0.05, statistics.median(ratios) ** (1.0 / 3.0)))

    post_jump_ratios = []
    for index, item in enumerate(returns[:-1]):
        if abs(item) >= jump_threshold:
            post_jump_ratios.append(abs(returns[index + 1]) / robust_sigma)
    aftershock_decay = min(0.95, max(0.05, (statistics.fmean(post_jump_ratios) - 0.8) / 3.0)) if post_jump_ratios else 0.50
    ranges = [math.log(item["high"] / item["low"]) for item in values]
    innovations = [(item - median) / robust_sigma for item in returns if abs(item) < jump_threshold]
    ordered_returns = sorted(returns)
    ordered_innovations = sorted(innovations)
    probabilities = [index / 100.0 for index in range(101)]
    path_drawdowns: List[float] = []
    path_worst_returns: List[float] = []
    for start in range(0, len(closes) - 24, 24):
        window = closes[start : start + 25]
        peak = window[0]
        drawdown = 0.0
        for price in window:
            peak = max(peak, price)
            drawdown = max(drawdown, 1.0 - price / peak)
        path_drawdowns.append(drawdown)
        path_worst_returns.append(min(math.log(b / a) for a, b in zip(window, window[1:])))
    return {
        "bar_count": len(values),
        "return_count": len(returns),
        "price_drift": statistics.fmean(returns),
        "price_volatility": statistics.pstdev(returns),
        "robust_volatility": robust_sigma,
        "volatility_persistence": vol_persistence,
        "volatility_of_volatility": vol_of_vol,
        "regime_persistence": regime_persistence,
        "regime_weights": regime_weights,
        "jump_threshold": jump_threshold,
        "jump_intensity": len(jump_sizes) / len(returns),
        "jump_scale": statistics.median(jump_sizes) if jump_sizes else jump_threshold,
        "jump_tail_df": tail_df,
        "jump_aftershock_decay": aftershock_decay,
        "orderflow_slope": flow_slope,
        "orderflow_correlation": flow_corr,
        "impact_decay": impact_decay,
        "median_quote_volume": statistics.median(item["quote_volume"] for item in values),
        "median_trade_count": statistics.median(item["trade_count"] for item in values),
        "median_log_high_low_range": statistics.median(ranges),
        "return_quantiles": {f"q{index:03d}": _quantile_sorted(ordered_returns, probability) for index, probability in enumerate(probabilities)},
        "standardized_innovation_quantiles": [round(_quantile_sorted(ordered_innovations, probability), 8) for probability in probabilities],
        "abs_return_acf": [round(_corr(abs_returns[:-lag], abs_returns[lag:]), 8) for lag in range(1, 13)],
        "path24_max_drawdown_quantiles": {
            label: _quantile(path_drawdowns, probability)
            for label, probability in (("q50", 0.50), ("q90", 0.90), ("q95", 0.95), ("q99", 0.99))
        },
        "path24_worst_return_quantiles": {
            label: _quantile(path_worst_returns, probability)
            for label, probability in (("q01", 0.01), ("q05", 0.05), ("q50", 0.50))
        },
    }


def _pooled_estimate(series: Mapping[str, Sequence[Dict[str, float]]]) -> Dict[str, Any]:
    # Each asset is standardized before pooling so BTC activity does not
    # dominate smaller markets solely through nominal volume or price scale.
    pooled: List[Dict[str, float]] = []
    anchor = 100.0
    for symbol in sorted(series):
        values = series[symbol]
        if not values:
            continue
        scale = anchor / values[0]["close"]
        for item in values:
            normalized = dict(item)
            for key in ("open", "high", "low", "close"):
                normalized[key] *= scale
            pooled.append(normalized)
        anchor = pooled[-1]["close"]
    return _estimate(pooled)


def _ks_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 1.0
    a, b = sorted(left), sorted(right)
    points = sorted(set(a + b))
    i = j = 0
    distance = 0.0
    for point in points:
        while i < len(a) and a[i] <= point:
            i += 1
        while j < len(b) and b[j] <= point:
            j += 1
        distance = max(distance, abs(i / len(a) - j / len(b)))
    return distance


def _holdout_report(fit: Mapping[str, Any], holdout: Mapping[str, Any]) -> Dict[str, Any]:
    fit_quantiles = [fit["return_quantiles"][f"q{index:03d}"] for index in range(101)]
    holdout_quantiles = [holdout["return_quantiles"][f"q{index:03d}"] for index in range(101)]
    normalized_error = statistics.fmean(
        abs(left - right) / max(fit["robust_volatility"], 1e-12)
        for left, right in zip(fit_quantiles, holdout_quantiles)
    )
    return {
        "protocol": "untouched chronological holdout; no agent result or benchmark score enters calibration",
        "fit_return_count": fit["return_count"],
        "holdout_return_count": holdout["return_count"],
        "volatility_ratio": holdout["price_volatility"] / max(fit["price_volatility"], 1e-12),
        "jump_intensity_ratio": holdout["jump_intensity"] / max(fit["jump_intensity"], 1e-12),
        "mean_normalized_quantile_error": normalized_error,
        "abs_return_acf_l1": statistics.fmean(
            abs(a - b) for a, b in zip(fit["abs_return_acf"], holdout["abs_return_acf"])
        ),
        "orderflow_correlation_difference": abs(
            fit["orderflow_correlation"] - holdout["orderflow_correlation"]
        ),
        "interpretation": "descriptive temporal transport audit, not an acceptance test tuned on the holdout",
    }


@dataclass(frozen=True)
class CalibrationProfile:
    schema_version: str
    calibration_id: str
    estimator_version: str
    bar_size_minutes: int
    fit_period: Dict[str, str]
    symbols: List[str]
    sources: List[Dict[str, Any]]
    fit_estimate: Dict[str, Any]
    symbol_estimates: Dict[str, Dict[str, Any]]
    identification: Dict[str, Any]
    protocol_priors: Dict[str, float]
    regime_model: Dict[str, Any] = field(default_factory=dict)
    local_projection: Dict[str, Any] = field(default_factory=dict)
    bootstrap: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibrationProfile":
        return cls(**dict(value))


class CalibrationCatalog:
    def __init__(self, profiles: Iterable[CalibrationProfile]) -> None:
        self._profiles = {item.calibration_id: item for item in profiles}
        if not self._profiles:
            raise ValueError("empty calibration catalog")

    def get(self, calibration_id: str) -> CalibrationProfile:
        try:
            return self._profiles[calibration_id]
        except KeyError as error:
            raise ValueError(f"unknown calibration profile: {calibration_id}") from error


@dataclass(frozen=True)
class CalibrationResult:
    profile: CalibrationProfile
    holdout_validation: Dict[str, Any]
    bootstrap_replicates: List[Dict[str, Any]] = field(default_factory=list)


def fit_binance_calibration(
    archive_paths: Sequence[Path],
    fit_start: str,
    fit_end: str,
    holdout_start: str,
    holdout_end: str,
    bar_size: int = 5,
    hmm_iterations: int = 8,
    bootstrap_replicates: int = 500,
    bootstrap_block_days: int = 7,
    bootstrap_seed: int = 20260317,
) -> CalibrationResult:
    if bar_size < 1:
        raise ValueError("bar_size must be positive")
    fit_paths = [Path(item) for item in archive_paths if fit_start <= _period(Path(item)) <= fit_end]
    holdout_paths = [Path(item) for item in archive_paths if holdout_start <= _period(Path(item)) <= holdout_end]
    if not fit_paths or not holdout_paths:
        raise ValueError("both fit and chronological holdout archives are required")
    if fit_end >= holdout_start:
        raise ValueError("fit period must end before holdout period begins")
    fit_series, fit_sources = _read_series(fit_paths, bar_size)
    holdout_series, holdout_sources = _read_series(holdout_paths, bar_size)
    if set(fit_series) != set(holdout_series):
        raise ValueError("fit and holdout symbol sets must match")
    fit = _pooled_estimate(fit_series)
    holdout = _pooled_estimate(holdout_series)
    advanced = fit_advanced_calibration(
        fit_series,
        fit,
        hmm_iterations=hmm_iterations,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_block_days=bootstrap_block_days,
        bootstrap_seed=bootstrap_seed,
    )
    hmm = advanced["regime_model"]
    fit["regime_weights"] = dict(zip(hmm["state_labels"], hmm["weights"]))
    fit["regime_transition_matrix"] = {
        source: dict(zip(hmm["state_labels"], hmm["transition_matrix"][index]))
        for index, source in enumerate(hmm["state_labels"])
    }
    fit["regime_persistence"] = sum(
        hmm["weights"][index] * hmm["transition_matrix"][index][index]
        for index in range(len(hmm["state_labels"]))
    )
    lp_estimates = advanced["local_projection"]["estimates"]
    fit["orderflow_slope"] = lp_estimates[0]["cumulative_return_response"]
    target_curve = [item["cumulative_return_response"] for item in lp_estimates]
    horizons = [item["horizon_steps"] for item in lp_estimates]
    best_error = float("inf")
    best_impact, best_decay = 0.0002, 0.50
    for decay_index in range(96):
        decay = decay_index / 100.0
        basis = [sum(decay ** step for step in range(horizon + 1)) for horizon in horizons]
        impact = max(0.0, sum(x * y for x, y in zip(basis, target_curve)) / max(sum(x * x for x in basis), 1e-15))
        error = sum((impact * x - y) ** 2 for x, y in zip(basis, target_curve))
        if error < best_error:
            best_error, best_impact, best_decay = error, impact, decay
    fit["orderflow_impact"] = min(0.02, max(0.00001, best_impact))
    fit["impact_decay"] = min(0.95, max(0.0, best_decay))
    advanced["local_projection"]["simulator_indirect_mapping"] = {
        "orderflow_impact": fit["orderflow_impact"],
        "impact_decay": fit["impact_decay"],
        "fit_squared_error": best_error,
        "mapping": "least-squares match of simulator geometric cumulative impulse curve",
    }
    by_symbol = {symbol: _estimate(fit_series[symbol]) for symbol in sorted(fit_series)}
    sources = sorted(fit_sources + holdout_sources, key=lambda item: item["source_object"])
    # The content identity is deliberately fit-only.  Mutating or extending
    # the future holdout cannot alter the calibrated generator.
    identity = {
        "schema": CALIBRATION_SCHEMA,
        "estimator": "hmm-viterbi-lp-block-bootstrap-v3",
        "bar_size": bar_size,
        "hmm_iterations": hmm_iterations,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_block_days": bootstrap_block_days,
        "bootstrap_seed": bootstrap_seed,
        "fit": [fit_start, fit_end],
        "sources": [(item["source_object"], item["raw_zip_sha256"]) for item in fit_sources],
    }
    calibration_id = "bcal-" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    profile = CalibrationProfile(
        schema_version=CALIBRATION_SCHEMA,
        calibration_id=calibration_id,
        estimator_version="hmm-viterbi-lp-block-bootstrap-v3",
        bar_size_minutes=bar_size,
        fit_period={"start": fit_start, "end": fit_end},
        symbols=sorted(fit_series),
        sources=sorted(fit_sources, key=lambda item: item["source_object"]),
        fit_estimate=fit,
        symbol_estimates=by_symbol,
        identification={
            "identified_from_klines": [
                "return location/scale and empirical innovation quantiles",
                "log-absolute-return persistence and volatility-of-volatility proxy",
                "four-state Gaussian HMM occupancy, emissions, and transition matrix",
                "jump frequency/scale/tail-index and post-jump volatility proxy",
                "multi-horizon taker-imbalance local-projection response curve",
                "quote-volume, trade-count, and high-low-range activity proxies",
            ],
            "not_identified_from_klines": [
                "latent fundamental value and fundamental-price coupling",
                "permanent versus transitory event decomposition",
                "bid-ask spread, queue priority, fill probability, and order-book depth",
                "causal counterfactual impact of the benchmark focal agent",
            ],
        },
        protocol_priors={
            "fundamental_volatility_ratio": 0.30,
            "fundamental_price_coupling": 0.16,
            "permanent_event_probability": 0.55,
            "spread_bps": 10.0,
            "fee_bps": 2.0,
            "liquidity_index": 20.0,
        },
        regime_model=hmm,
        local_projection=advanced["local_projection"],
        bootstrap=advanced["bootstrap"],
    )
    holdout_regime = evaluate_regime_model(holdout_series, hmm)
    holdout_lp, _holdout_lp_daily = fit_local_projections(holdout_series)
    holdout_validation = {
        "schema_version": "evolens-binance-locked-holdout-v1",
        "calibration_id": calibration_id,
        "period": {"start": holdout_start, "end": holdout_end},
        "sources": sorted(holdout_sources, key=lambda item: item["source_object"]),
        "estimate": holdout,
        "transport_audit": _holdout_report(fit, holdout),
        "advanced_validation": {
            "regime_model": holdout_regime,
            "local_projection": holdout_lp,
            "rule": "fit transforms and HMM emissions remain frozen; holdout estimates never update generation",
        },
        "status": "DESCRIPTIVE_AUDIT",
    }
    return CalibrationResult(
        profile=profile,
        holdout_validation=holdout_validation,
        bootstrap_replicates=advanced["bootstrap_replicates"],
    )


def write_calibration_profile(path: Path, profile: CalibrationProfile) -> str:
    payload = (json.dumps(profile.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def load_calibration_profile(path: Path) -> CalibrationProfile:
    profile = CalibrationProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
    if profile.schema_version not in SUPPORTED_CALIBRATION_SCHEMAS:
        raise ValueError(f"unsupported calibration schema: {profile.schema_version}")
    return profile


def load_calibration_catalog(dataset_dir: Path, manifest: Mapping[str, Any]) -> Optional[CalibrationCatalog]:
    track = manifest.get("track", {})
    if track.get("kind") != "binance_calibrated_semi_synthetic":
        return None
    relative = track.get("calibration_profile", "resources/calibration_profile.json")
    path = (dataset_dir / relative).resolve()
    root = dataset_dir.resolve()
    if root not in path.parents:
        raise ValueError(f"calibration resource escapes dataset: {relative}")
    expected = manifest.get("sha256", {}).get(relative)
    if not expected or sha256_file(path) != expected:
        raise ValueError(f"calibration profile hash mismatch: {relative}")
    profile = load_calibration_profile(path)
    if profile.calibration_id != track.get("calibration_id"):
        raise ValueError("calibration_id mismatch")
    return CalibrationCatalog([profile])
