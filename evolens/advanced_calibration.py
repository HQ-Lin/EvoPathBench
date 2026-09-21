"""Advanced fit-only calibration: Gaussian HMM, local projections, and block bootstrap.

The implementation intentionally uses only the Python standard library so the
published benchmark remains rebuildable without a scientific Python stack.
"""
from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Sequence, Tuple


STATE_LABELS = ("balanced", "directional", "volatile", "reversal")
FEATURE_NAMES = ("log_abs_return", "direction_strength", "reversal_indicator", "relative_log_volume")
LP_HORIZONS = (0, 1, 2, 3, 6, 12)


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


def _day(open_time_us: int) -> int:
    return int(open_time_us // 86_400_000_000)


def _feature_sequences(
    series: Mapping[str, Sequence[Mapping[str, float]]],
    frozen_transforms: Mapping[str, Any] = None,
) -> Tuple[Dict[str, List[Tuple[int, Tuple[float, ...]]]], Dict[str, Any]]:
    sequences: Dict[str, List[Tuple[int, Tuple[float, ...]]]] = {}
    transforms: Dict[str, Any] = {"feature_names": list(FEATURE_NAMES), "symbols": {}}
    for symbol in sorted(series):
        values = series[symbol]
        closes = [item["close"] for item in values]
        returns = [math.log(right / left) for left, right in zip(closes, closes[1:])]
        frozen = (frozen_transforms or {}).get("symbols", {}).get(symbol)
        median_return = float(frozen["return_median"]) if frozen else statistics.median(returns)
        sigma = float(frozen["return_robust_sigma"]) if frozen else max(
            1e-10, 1.4826 * statistics.median(abs(item - median_return) for item in returns)
        )
        median_volume = float(frozen["median_quote_volume"]) if frozen else max(
            1e-12, statistics.median(item["quote_volume"] for item in values)
        )
        transforms["symbols"][symbol] = {
            "return_median": median_return,
            "return_robust_sigma": sigma,
            "median_quote_volume": median_volume,
        }
        observations: List[Tuple[int, Tuple[float, ...]]] = []
        for index in range(6, len(values)):
            ret = returns[index - 1]
            recent = returns[index - 6 : index]
            log_abs = math.log(abs(ret - median_return) / sigma + 0.05)
            direction = min(6.0, abs(statistics.fmean(recent) - median_return) / sigma)
            reversal = 1.0 if ret * returns[index - 2] < 0 else 0.0
            relative_volume = min(5.0, max(-5.0, math.log(max(values[index]["quote_volume"], 1e-12) / median_volume)))
            observations.append(
                (
                    int(values[index]["open_time_us"]),
                    (min(5.0, max(-5.0, log_abs)), direction, reversal, relative_volume),
                )
            )
        sequences[symbol] = observations
    return sequences, transforms


def _initial_model(sequences: Mapping[str, Sequence[Tuple[int, Tuple[float, ...]]]]) -> Dict[str, Any]:
    assigned: List[List[Tuple[float, ...]]] = [[] for _ in STATE_LABELS]
    transitions = [[1.0 for _ in STATE_LABELS] for _ in STATE_LABELS]
    weights = [1.0 for _ in STATE_LABELS]
    for observations in sequences.values():
        previous = None
        for _timestamp, obs in observations:
            if obs[0] > math.log(1.75):
                state = 2
            elif obs[1] > 0.45:
                state = 1
            elif obs[2] > 0.5 and obs[0] > math.log(0.55):
                state = 3
            else:
                state = 0
            assigned[state].append(obs)
            if previous is None:
                weights[state] += 1.0
            else:
                transitions[previous][state] += 1.0
            previous = state
    all_observations = [obs for values in assigned for obs in values]
    global_mean = [statistics.fmean(obs[d] for obs in all_observations) for d in range(len(FEATURE_NAMES))]
    global_var = [
        max(0.05, statistics.fmean((obs[d] - global_mean[d]) ** 2 for obs in all_observations))
        for d in range(len(FEATURE_NAMES))
    ]
    means: List[List[float]] = []
    variances: List[List[float]] = []
    for values in assigned:
        if len(values) < 20:
            means.append(list(global_mean))
            variances.append(list(global_var))
        else:
            local_mean = [statistics.fmean(obs[d] for obs in values) for d in range(len(FEATURE_NAMES))]
            means.append(local_mean)
            variances.append(
                [max(0.02, statistics.fmean((obs[d] - local_mean[d]) ** 2 for obs in values)) for d in range(len(FEATURE_NAMES))]
            )
    return {
        "weights": [item / sum(weights) for item in weights],
        "transition": [[item / sum(row) for item in row] for row in transitions],
        "means": means,
        "variances": variances,
    }


def _stationary_distribution(transition: Sequence[Sequence[float]]) -> List[float]:
    weights = [0.25] * 4
    for _ in range(10000):
        updated = [
            sum(weights[source] * transition[source][target] for source in range(4))
            for target in range(4)
        ]
        scale = max(sum(updated), 1e-15)
        updated = [item / scale for item in updated]
        if max(abs(left - right) for left, right in zip(weights, updated)) < 1e-12:
            return updated
        weights = updated
    return weights


def _emissions(
    observations: Sequence[Tuple[int, Tuple[float, ...]]], model: Mapping[str, Any]
) -> Tuple[List[List[float]], List[float]]:
    result: List[List[float]] = []
    offsets: List[float] = []
    for _timestamp, obs in observations:
        logs = []
        for mean, variance in zip(model["means"], model["variances"]):
            logs.append(
                -0.5
                * sum(
                    math.log(2.0 * math.pi * variance[d]) + (obs[d] - mean[d]) ** 2 / variance[d]
                    for d in range(len(obs))
                )
            )
        maximum = max(logs)
        result.append([max(1e-300, math.exp(item - maximum)) for item in logs])
        offsets.append(maximum)
    return result, offsets


def _forward(emissions: Sequence[Sequence[float]], model: Mapping[str, Any]) -> Tuple[List[List[float]], List[float]]:
    alphas: List[List[float]] = []
    scales: List[float] = []
    first = [model["weights"][state] * emissions[0][state] for state in range(4)]
    scale = max(sum(first), 1e-300)
    alphas.append([item / scale for item in first])
    scales.append(scale)
    for emission in emissions[1:]:
        previous = alphas[-1]
        current = [
            emission[state]
            * sum(previous[source] * model["transition"][source][state] for source in range(4))
            for state in range(4)
        ]
        scale = max(sum(current), 1e-300)
        alphas.append([item / scale for item in current])
        scales.append(scale)
    return alphas, scales


def fit_gaussian_hmm(
    series: Mapping[str, Sequence[Mapping[str, float]]], iterations: int = 8
) -> Tuple[Dict[str, Any], Dict[str, List[Tuple[int, int]]]]:
    sequences, transforms = _feature_sequences(series)
    model = _initial_model(sequences)
    history: List[float] = []
    for _iteration in range(iterations):
        weight_sum = [1e-6] * 4
        transition_sum = [[1e-6] * 4 for _ in range(4)]
        gamma_sum = [1e-6] * 4
        value_sum = [[0.0] * len(FEATURE_NAMES) for _ in range(4)]
        square_sum = [[0.0] * len(FEATURE_NAMES) for _ in range(4)]
        log_likelihood = 0.0
        for observations in sequences.values():
            emissions, emission_offsets = _emissions(observations, model)
            alphas, scales = _forward(emissions, model)
            log_likelihood += sum(
                math.log(max(scale, 1e-300)) + offset
                for scale, offset in zip(scales, emission_offsets)
            )
            beta = [1.0] * 4

            def accumulate(index: int, gamma: Sequence[float]) -> None:
                obs = observations[index][1]
                for state in range(4):
                    weight = gamma[state]
                    gamma_sum[state] += weight
                    for dimension in range(len(obs)):
                        value_sum[state][dimension] += weight * obs[dimension]
                        square_sum[state][dimension] += weight * obs[dimension] ** 2

            last_gamma = alphas[-1]
            accumulate(len(observations) - 1, last_gamma)
            for state in range(4):
                weight_sum[state] += alphas[0][state]
            for index in range(len(observations) - 2, -1, -1):
                next_emission = emissions[index + 1]
                denominator = 0.0
                for source in range(4):
                    for target in range(4):
                        denominator += (
                            alphas[index][source]
                            * model["transition"][source][target]
                            * next_emission[target]
                            * beta[target]
                        )
                denominator = max(denominator, 1e-300)
                for source in range(4):
                    for target in range(4):
                        transition_sum[source][target] += (
                            alphas[index][source]
                            * model["transition"][source][target]
                            * next_emission[target]
                            * beta[target]
                            / denominator
                        )
                beta_current = [
                    sum(
                        model["transition"][source][target]
                        * next_emission[target]
                        * beta[target]
                        for target in range(4)
                    )
                    / max(scales[index + 1], 1e-300)
                    for source in range(4)
                ]
                gamma_raw = [alphas[index][state] * beta_current[state] for state in range(4)]
                gamma_scale = max(sum(gamma_raw), 1e-300)
                gamma = [item / gamma_scale for item in gamma_raw]
                accumulate(index, gamma)
                beta = beta_current
        model = {
            "weights": [item / sum(weight_sum) for item in weight_sum],
            "transition": [[item / sum(row) for item in row] for row in transition_sum],
            "means": [[value_sum[s][d] / gamma_sum[s] for d in range(len(FEATURE_NAMES))] for s in range(4)],
            "variances": [
                [
                    max(0.01, square_sum[s][d] / gamma_sum[s] - (value_sum[s][d] / gamma_sum[s]) ** 2)
                    for d in range(len(FEATURE_NAMES))
                ]
                for s in range(4)
            ],
        }
        history.append(log_likelihood)

    # Align otherwise exchangeable HMM labels to stable semantic signatures.
    volatile = max(range(4), key=lambda state: model["means"][state][0])
    remaining = [state for state in range(4) if state != volatile]
    directional = max(remaining, key=lambda state: model["means"][state][1])
    remaining = [state for state in remaining if state != directional]
    reversal = max(remaining, key=lambda state: model["means"][state][2])
    balanced = next(state for state in remaining if state != reversal)
    order = [balanced, directional, volatile, reversal]
    aligned_transition = [
        [model["transition"][source][target] for target in order] for source in order
    ]
    aligned = {
        "schema_version": "evolens-gaussian-hmm-v2",
        "state_labels": list(STATE_LABELS),
        "feature_names": list(FEATURE_NAMES),
        "weights": _stationary_distribution(aligned_transition),
        "initial_state_probabilities": [model["weights"][state] for state in order],
        "transition_matrix": aligned_transition,
        "emission_means": [model["means"][state] for state in order],
        "emission_variances": [model["variances"][state] for state in order],
        "label_alignment_old_indices": order,
        "iterations": iterations,
        "observation_count": sum(len(items) for items in sequences.values()),
        "sequence_count": len(sequences),
        "log_likelihood_history": history,
        "feature_transforms": transforms,
        "claim": "probabilistic statistical regimes; semantic labels are post-fit alignments, not uniquely identified economic market states",
    }
    aligned_model = {
        "weights": aligned["weights"],
        "transition": aligned["transition_matrix"],
        "means": aligned["emission_means"],
        "variances": aligned["emission_variances"],
    }
    decoded: Dict[str, List[Tuple[int, int]]] = {}
    for symbol, observations in sequences.items():
        decoded[symbol] = _viterbi(observations, aligned_model)
    return aligned, decoded


def _viterbi(
    observations: Sequence[Tuple[int, Tuple[float, ...]]], model: Mapping[str, Any]
) -> List[Tuple[int, int]]:
    emissions, _emission_offsets = _emissions(observations, model)
    delta = [math.log(max(model["weights"][state], 1e-300)) + math.log(emissions[0][state]) for state in range(4)]
    back: List[Tuple[int, int, int, int]] = []
    for emission in emissions[1:]:
        pointers = []
        current = []
        for target in range(4):
            candidates = [
                delta[source] + math.log(max(model["transition"][source][target], 1e-300))
                for source in range(4)
            ]
            source = max(range(4), key=lambda item: candidates[item])
            pointers.append(source)
            current.append(candidates[source] + math.log(max(emission[target], 1e-300)))
        back.append(tuple(pointers))
        delta = current
    state = max(range(4), key=lambda item: delta[item])
    states = [state]
    for pointers in reversed(back):
        state = pointers[state]
        states.append(state)
    states.reverse()
    return [(observations[index][0], state) for index, state in enumerate(states)]


def _zero_matrix(size: int) -> List[List[float]]:
    return [[0.0] * size for _ in range(size)]


def _add_outer(matrix: List[List[float]], left: Sequence[float], weight: float = 1.0) -> None:
    for row in range(len(left)):
        for column in range(len(left)):
            matrix[row][column] += weight * left[row] * left[column]


def _solve(matrix: Sequence[Sequence[float]], vector: Sequence[float], ridge: float = 1e-8) -> List[float]:
    size = len(vector)
    augmented = [list(matrix[row]) + [float(vector[row])] for row in range(size)]
    for index in range(size):
        augmented[index][index] += ridge
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        if abs(scale) < 1e-14:
            continue
        augmented[column] = [item / scale for item in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * reference
                for value, reference in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def fit_local_projections(
    series: Mapping[str, Sequence[Mapping[str, float]]],
    horizons: Sequence[int] = LP_HORIZONS,
) -> Tuple[Dict[str, Any], Dict[int, Dict[int, Dict[str, Any]]]]:
    symbols = sorted(series)
    dimension = 7  # imbalance, lag return, abs lag, log volume, three asset fixed effects
    daily_rows: DefaultDict[int, List[Tuple[List[float], Dict[int, float]]]] = defaultdict(list)
    for symbol_index, symbol in enumerate(symbols):
        values = series[symbol]
        closes = [item["close"] for item in values]
        returns = [math.log(right / left) for left, right in zip(closes, closes[1:])]
        median_volume = max(1e-12, statistics.median(item["quote_volume"] for item in values))
        for index in range(2, len(values) - max(horizons) - 1):
            fixed = [1.0 if symbol_index == item else 0.0 for item in range(max(0, len(symbols) - 1))]
            while len(fixed) < 3:
                fixed.append(0.0)
            x = [
                values[index]["imbalance"],
                returns[index - 1],
                abs(returns[index - 1]),
                math.log(max(values[index]["quote_volume"], 1e-12) / median_volume),
            ] + fixed[:3]
            outcomes = {
                int(horizon): sum(returns[index : index + int(horizon) + 1])
                for horizon in horizons
            }
            daily_rows[_day(int(values[index]["open_time_us"]))].append((x, outcomes))

    # Day demeaning supplies a transparent daily common-effect control while
    # preserving a small, auditable normal equation.
    daily_crossproducts: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for day, rows in daily_rows.items():
        mean_x = [statistics.fmean(row[0][d] for row in rows) for d in range(dimension)]
        mean_y = {h: statistics.fmean(row[1][h] for row in rows) for h in horizons}
        horizon_data: Dict[int, Dict[str, Any]] = {}
        for horizon in horizons:
            xtx = _zero_matrix(dimension)
            xty = [0.0] * dimension
            for x, outcomes in rows:
                centered_x = [x[d] - mean_x[d] for d in range(dimension)]
                centered_y = outcomes[horizon] - mean_y[horizon]
                _add_outer(xtx, centered_x)
                for d in range(dimension):
                    xty[d] += centered_x[d] * centered_y
            horizon_data[int(horizon)] = {"xtx": xtx, "xty": xty, "n": len(rows)}
        daily_crossproducts[day] = horizon_data

    estimates = []
    for horizon in horizons:
        xtx = _zero_matrix(dimension)
        xty = [0.0] * dimension
        count = 0
        for data in daily_crossproducts.values():
            item = data[int(horizon)]
            count += item["n"]
            for row in range(dimension):
                xty[row] += item["xty"][row]
                for column in range(dimension):
                    xtx[row][column] += item["xtx"][row][column]
        coefficients = _solve(xtx, xty)
        estimates.append(
            {
                "horizon_steps": int(horizon),
                "horizon_minutes": 5 * (int(horizon) + 1),
                "cumulative_return_response": coefficients[0],
                "coefficients": coefficients,
                "observation_count": count,
            }
        )
    return (
        {
            "schema_version": "evolens-orderflow-local-projection-v1",
            "horizons": [int(item) for item in horizons],
            "regressors": [
                "taker_imbalance",
                "lag_return",
                "absolute_lag_return",
                "relative_log_quote_volume",
                "asset_fixed_effect_1",
                "asset_fixed_effect_2",
                "asset_fixed_effect_3",
            ],
            "common_effect_control": "within-UTC-day demeaning",
            "estimates": estimates,
            "claim": "reduced-form cumulative response; not structural causal market impact",
        },
        daily_crossproducts,
    )


def _daily_market_stats(
    series: Mapping[str, Sequence[Mapping[str, float]]],
    decoded: Mapping[str, Sequence[Tuple[int, int]]],
    robust_sigma: float,
    jump_threshold: float,
) -> Dict[int, Dict[str, Any]]:
    daily: DefaultDict[int, Dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0, "sum": 0.0, "sum2": 0.0, "jump_n": 0,
            "jump_abs": 0.0, "jump_log_excess": 0.0,
            "logpair_n": 0, "log_x": 0.0, "log_y": 0.0,
            "log_x2": 0.0, "log_y2": 0.0, "log_xy": 0.0,
            "state_counts": [0.0] * 4,
            "transition_counts": [[0.0] * 4 for _ in range(4)],
        }
    )
    for symbol, values in series.items():
        closes = [item["close"] for item in values]
        returns = [math.log(right / left) for left, right in zip(closes, closes[1:])]
        for index, ret in enumerate(returns, start=1):
            item = daily[_day(int(values[index]["open_time_us"]))]
            item["n"] += 1
            item["sum"] += ret
            item["sum2"] += ret * ret
            if abs(ret) >= jump_threshold:
                item["jump_n"] += 1
                item["jump_abs"] += abs(ret)
                if abs(ret) > jump_threshold:
                    item["jump_log_excess"] += math.log(abs(ret) / jump_threshold)
            if index >= 2:
                left = math.log(max(abs(returns[index - 2]), robust_sigma * 0.05))
                right = math.log(max(abs(ret), robust_sigma * 0.05))
                item["logpair_n"] += 1
                item["log_x"] += left
                item["log_y"] += right
                item["log_x2"] += left * left
                item["log_y2"] += right * right
                item["log_xy"] += left * right
        states = decoded[symbol]
        for index, (timestamp, state) in enumerate(states):
            item = daily[_day(timestamp)]
            item["state_counts"][state] += 1.0
            if index and _day(states[index - 1][0]) == _day(timestamp):
                item["transition_counts"][states[index - 1][1]][state] += 1.0
    return dict(daily)


def _blocks(days: Sequence[int], block_days: int) -> List[List[int]]:
    ordered = sorted(days)
    runs: List[List[int]] = []
    for day in ordered:
        if not runs or day != runs[-1][-1] + 1:
            runs.append([day])
        else:
            runs[-1].append(day)
    return [
        run[start : start + block_days]
        for run in runs
        for start in range(0, len(run), block_days)
        if len(run[start : start + block_days]) == block_days
    ]


def _percentile_ci(values: Sequence[float]) -> List[float]:
    return [_quantile(values, 0.025), _quantile(values, 0.975)]


def synchronized_block_bootstrap(
    series: Mapping[str, Sequence[Mapping[str, float]]],
    decoded: Mapping[str, Sequence[Tuple[int, int]]],
    lp_daily: Mapping[int, Mapping[int, Mapping[str, Any]]],
    fit_estimate: Mapping[str, Any],
    replicates: int = 500,
    block_days: int = 7,
    seed: int = 20260317,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    market_daily = _daily_market_stats(
        series, decoded, fit_estimate["robust_volatility"], fit_estimate["jump_threshold"]
    )
    common_days = sorted(set(market_daily) & set(lp_daily))
    blocks = _blocks(common_days, block_days)
    effective_block_days = block_days
    if len(blocks) < 2:
        effective_block_days = 1
        blocks = [[day] for day in common_days]
    if len(blocks) < 2:
        raise ValueError("too few synchronized calendar blocks for bootstrap")
    def estimate_from_days(days: Sequence[int], replicate: Any) -> Dict[str, Any]:
        n = sum(market_daily[day]["n"] for day in days)
        total = sum(market_daily[day]["sum"] for day in days)
        total2 = sum(market_daily[day]["sum2"] for day in days)
        drift = total / max(n, 1)
        volatility = math.sqrt(max(0.0, total2 / max(n, 1) - drift * drift))
        jump_n = sum(market_daily[day]["jump_n"] for day in days)
        jump_abs = sum(market_daily[day]["jump_abs"] for day in days)
        jump_log = sum(market_daily[day]["jump_log_excess"] for day in days)
        pair_n = sum(market_daily[day]["logpair_n"] for day in days)
        sx = sum(market_daily[day]["log_x"] for day in days)
        sy = sum(market_daily[day]["log_y"] for day in days)
        sxx = sum(market_daily[day]["log_x2"] for day in days)
        syy = sum(market_daily[day]["log_y2"] for day in days)
        sxy = sum(market_daily[day]["log_xy"] for day in days)
        numerator = sxy - sx * sy / max(pair_n, 1)
        denominator = math.sqrt(max(1e-20, (sxx - sx * sx / max(pair_n, 1)) * (syy - sy * sy / max(pair_n, 1))))
        vol_persistence = min(0.99, max(-0.99, numerator / denominator))
        state_counts = [sum(market_daily[day]["state_counts"][s] for day in days) for s in range(4)]
        transition = []
        for source in range(4):
            counts = [
                sum(market_daily[day]["transition_counts"][source][target] for day in days)
                for target in range(4)
            ]
            transition.append([item / max(sum(counts), 1.0) for item in counts])
        state_total = max(sum(state_counts), 1.0)
        decoded_weights = [item / state_total for item in state_counts]
        # Match the formal HMM point estimator: regime prevalence is the
        # stationary distribution implied by the fitted transition matrix.
        # Decoded occupancy remains in the audit row as a separate diagnostic.
        weights = _stationary_distribution(transition)
        row: Dict[str, Any] = {
            "replicate": replicate,
            "price_drift": drift,
            "price_volatility": volatility,
            "volatility_persistence": vol_persistence,
            "jump_intensity": jump_n / max(n, 1),
            "jump_scale": jump_abs / max(jump_n, 1),
            "jump_tail_df": min(20.0, max(2.1, 1.0 / max(jump_log / max(jump_n, 1), 1e-6))),
            "regime_weights": weights,
            "decoded_regime_weights": decoded_weights,
            "transition_matrix": transition,
            "regime_persistence": sum(weights[state] * transition[state][state] for state in range(4)),
            "local_projection": {},
            "local_projection_coefficients": {},
            "local_projection_observation_count": {},
        }
        for horizon in LP_HORIZONS:
            dimension = len(lp_daily[days[0]][horizon]["xty"])
            xtx = _zero_matrix(dimension)
            xty = [0.0] * dimension
            for day in days:
                item = lp_daily[day][horizon]
                for i in range(dimension):
                    xty[i] += item["xty"][i]
                    for j in range(dimension):
                        xtx[i][j] += item["xtx"][i][j]
            coefficients = _solve(xtx, xty)
            row["local_projection"][str(horizon)] = coefficients[0]
            row["local_projection_coefficients"][str(horizon)] = coefficients
            row["local_projection_observation_count"][str(horizon)] = sum(
                lp_daily[day][horizon]["n"] for day in days
            )
        return row

    # The point estimate and every bootstrap replicate use the same complete
    # synchronized blocks. This avoids attaching an interval from a subtly
    # different estimator to a full-sample HMM/LP coefficient.
    eligible_days = [day for block in blocks for day in block]
    point_estimate = estimate_from_days(eligible_days, "point")
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for replicate in range(replicates):
        selected = [blocks[rng.randrange(len(blocks))] for _ in range(len(blocks))]
        days = [day for block in selected for day in block]
        rows.append(estimate_from_days(days, replicate))

    scalar_keys = (
        "price_drift", "price_volatility", "volatility_persistence", "jump_intensity",
        "jump_scale", "jump_tail_df", "regime_persistence",
    )
    intervals: Dict[str, Any] = {key: _percentile_ci([row[key] for row in rows]) for key in scalar_keys}
    intervals["regime_weights"] = {
        STATE_LABELS[state]: _percentile_ci([row["regime_weights"][state] for row in rows])
        for state in range(4)
    }
    intervals["transition_matrix"] = {
        f"{STATE_LABELS[source]}->{STATE_LABELS[target]}": _percentile_ci(
            [row["transition_matrix"][source][target] for row in rows]
        )
        for source in range(4)
        for target in range(4)
    }
    intervals["local_projection"] = {
        str(horizon): _percentile_ci([row["local_projection"][str(horizon)] for row in rows])
        for horizon in LP_HORIZONS
    }
    summary = {
        "schema_version": "evolens-synchronized-block-bootstrap-v2",
        "replicates": replicates,
        "block_days": effective_block_days,
        "requested_block_days": block_days,
        "synchronized_symbols": sorted(series),
        "calendar_block_count": len(blocks),
        "eligible_calendar_day_count": len(eligible_days),
        "seed": seed,
        "confidence_level": 0.95,
        "interval_method": "percentile",
        "intervals": intervals,
        "point_estimate": point_estimate,
        "hmm_uncertainty_scope": "fixed fitted emissions with Viterbi decoded-state transition/count re-estimation",
        "claim": "blocks preserve within-week dependence and cross-asset common shocks; HMM emission refit uncertainty is not included",
    }
    return summary, rows


def fit_advanced_calibration(
    series: Mapping[str, Sequence[Mapping[str, float]]],
    fit_estimate: Mapping[str, Any],
    hmm_iterations: int = 8,
    bootstrap_replicates: int = 500,
    bootstrap_block_days: int = 7,
    bootstrap_seed: int = 20260317,
) -> Dict[str, Any]:
    hmm, decoded = fit_gaussian_hmm(series, iterations=hmm_iterations)
    lp, lp_daily = fit_local_projections(series)
    bootstrap, replicates = synchronized_block_bootstrap(
        series,
        decoded,
        lp_daily,
        fit_estimate,
        replicates=bootstrap_replicates,
        block_days=bootstrap_block_days,
        seed=bootstrap_seed,
    )
    # Baum-Welch supplies the probabilistic emissions and the transition
    # matrix used for the original Viterbi decode. The benchmark-facing
    # transition estimator is then the hard decoded transition matrix on the
    # exact synchronized-block sample used by the bootstrap. Keeping both
    # avoids presenting uncertainty from one estimator around another.
    point = bootstrap["point_estimate"]
    hmm["baum_welch_transition_matrix"] = hmm["transition_matrix"]
    hmm["baum_welch_stationary_weights"] = hmm["weights"]
    hmm["transition_matrix"] = point["transition_matrix"]
    hmm["weights"] = point["regime_weights"]
    hmm["decoded_occupancy_weights"] = point["decoded_regime_weights"]
    hmm["transition_estimator"] = (
        "Viterbi decoded within-UTC-day transition frequencies on the complete "
        "synchronized blocks used by the bootstrap"
    )
    for estimate in lp["estimates"]:
        horizon = str(estimate["horizon_steps"])
        estimate["cumulative_return_response"] = point["local_projection"][horizon]
        estimate["coefficients"] = point["local_projection_coefficients"][horizon]
        estimate["observation_count"] = point["local_projection_observation_count"][horizon]
        estimate["bootstrap_ci95"] = bootstrap["intervals"]["local_projection"][horizon]
    return {
        "regime_model": hmm,
        "local_projection": lp,
        "bootstrap": bootstrap,
        "bootstrap_replicates": replicates,
    }


def evaluate_regime_model(
    series: Mapping[str, Sequence[Mapping[str, float]]], hmm: Mapping[str, Any]
) -> Dict[str, Any]:
    sequences, _ = _feature_sequences(series, hmm["feature_transforms"])
    model = {
        "weights": hmm["weights"],
        "transition": hmm["transition_matrix"],
        "means": hmm["emission_means"],
        "variances": hmm["emission_variances"],
    }
    state_counts = [0.0] * 4
    transition_counts = [[0.0] * 4 for _ in range(4)]
    for observations in sequences.values():
        decoded = _viterbi(observations, model)
        for index, (_timestamp, state) in enumerate(decoded):
            state_counts[state] += 1.0
            if index:
                transition_counts[decoded[index - 1][1]][state] += 1.0
    total = max(sum(state_counts), 1.0)
    weights = [item / total for item in state_counts]
    transition = [
        [item / max(sum(row), 1.0) for item in row] for row in transition_counts
    ]
    return {
        "state_labels": list(STATE_LABELS),
        "observation_count": int(total),
        "weights": weights,
        "transition_matrix": transition,
        "weighted_diagonal_persistence": sum(weights[state] * transition[state][state] for state in range(4)),
        "protocol": "fit-period emissions and transforms frozen; holdout states decoded without refitting",
    }
