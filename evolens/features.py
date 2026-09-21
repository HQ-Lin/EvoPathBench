"""Observable feature summaries used by scripted baselines and diagnostics."""
from __future__ import annotations

import math
import statistics
from typing import Dict, List


def safe_mean(values: List[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def returns(prices: List[float]) -> List[float]:
    result: List[float] = []
    for previous, current in zip(prices, prices[1:]):
        if previous > 0:
            result.append(current / previous - 1.0)
    return result


def observable_features(
    prices: List[float],
    fundamentals: List[float],
    spread_bps: float,
    liquidity: float,
) -> Dict[str, float]:
    if not prices:
        return {
            "trend": 0.0,
            "volatility": 0.0,
            "mean_abs_gap": 0.0,
            "autocorrelation": 0.0,
            "spread_bps": spread_bps,
            "liquidity": liquidity,
        }
    observed_returns = returns(prices)
    trend = prices[-1] / prices[0] - 1.0 if prices[0] else 0.0
    volatility = statistics.pstdev(observed_returns) if len(observed_returns) > 1 else 0.0
    gaps = [abs(price / fundamental - 1.0) for price, fundamental in zip(prices, fundamentals) if fundamental > 0]
    autocorrelation = 0.0
    if len(observed_returns) >= 3:
        left = observed_returns[:-1]
        right = observed_returns[1:]
        left_mean = safe_mean(left)
        right_mean = safe_mean(right)
        numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
        denominator = math.sqrt(
            sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right)
        )
        autocorrelation = numerator / denominator if denominator > 0 else 0.0
    return {
        "trend": trend,
        "volatility": volatility,
        "mean_abs_gap": safe_mean(gaps),
        "autocorrelation": autocorrelation,
        "spread_bps": spread_bps,
        "liquidity": liquidity,
    }


def feature_signature(
    prices: List[float],
    fundamentals: List[float],
    spread_bps: float,
    liquidity: float,
) -> str:
    values = observable_features(prices, fundamentals, spread_bps, liquidity)
    if values["spread_bps"] >= 18 or liquidity <= 8:
        execution = "thin"
    else:
        execution = "liquid"
    if values["mean_abs_gap"] >= 0.012 and values["autocorrelation"] < 0.15:
        regime = "valuation"
    elif abs(values["trend"]) >= 0.018 or values["autocorrelation"] >= 0.20:
        regime = "directional"
    elif values["volatility"] >= 0.018:
        regime = "volatile"
    else:
        regime = "balanced"
    return f"{regime}:{execution}"

