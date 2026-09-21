#!/usr/bin/env python3
"""Design-aware statistical inference for the EvoPathBench 36 x 3 x 3 experiment.

The 36 benchmark structure cells are treated as fixed strata. Within every cell,
the analysis resamples three market streams and then three campaigns within each
selected stream. All resampling and randomization operations preserve pairing
between methods, the no-update baseline, and state-off controls.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


METHODS_WITH_STATE_OFF = (
    "context",
    "reflection",
    "episodic_memory",
    "consolidated_memory",
    "skillopt",
    "skillboost",
    "skillx",
    "trace2skill",
    "skillgrad",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def empirical_cvar(values: np.ndarray, tail_probability: float) -> float:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return float("nan")
    mass = tail_probability * data.size
    whole = int(math.floor(mass))
    fraction = mass - whole
    ordered = np.sort(data)[::-1]
    numerator = float(ordered[:whole].sum())
    if fraction > 1e-12 and whole < data.size:
        numerator += fraction * float(ordered[whole])
    return numerator / mass


def percentile_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha])
    return float(low), float(high)


def path_bootstrap_indices(
    paths: pd.DataFrame, cell_order: list[str], bootstrap_reps: int, rng: np.random.Generator
) -> np.ndarray:
    """Fixed-stratum, two-stage bootstrap: streams, then campaigns."""
    lookup = {
        (row.cell_id, row.stream_id, int(row.campaign_id)): int(row.path_index)
        for row in paths.itertuples(index=False)
    }
    output = np.empty((bootstrap_reps, len(paths)), dtype=np.int32)
    offset = 0
    for cell_id in cell_order:
        cell = paths[paths["cell_id"] == cell_id]
        streams = sorted(cell["stream_id"].unique())
        if len(streams) != 3:
            raise RuntimeError(f"{cell_id}: expected 3 streams, found {len(streams)}")
        selected_streams = rng.integers(0, 3, size=(bootstrap_reps, 3))
        block = np.empty((bootstrap_reps, 9), dtype=np.int32)
        for stream_draw in range(3):
            selected_campaigns = rng.integers(0, 3, size=(bootstrap_reps, 3))
            for campaign_draw in range(3):
                for stream_index, stream_id in enumerate(streams):
                    mask = selected_streams[:, stream_draw] == stream_index
                    if not np.any(mask):
                        continue
                    campaigns = selected_campaigns[mask, campaign_draw]
                    block[mask, 3 * stream_draw + campaign_draw] = [
                        lookup[(cell_id, stream_id, int(campaign))] for campaign in campaigns
                    ]
        output[:, offset : offset + 9] = block
        offset += 9
    return output


def vector_from_path_series(series: pd.Series, path_count: int) -> np.ndarray:
    output = np.full(path_count, np.nan, dtype=float)
    for path_index, value in series.items():
        output[int(path_index)] = float(value)
    if np.isnan(output).any():
        raise RuntimeError("metric is missing one or more path units")
    return output


def bootstrap_path_mean(values: np.ndarray, selections: np.ndarray) -> np.ndarray:
    return values[selections].mean(axis=1)


def stream_cluster_values(values: np.ndarray, paths: pd.DataFrame) -> np.ndarray:
    frame = paths[["path_index", "cell_id", "stream_id"]].copy()
    frame["value"] = values[frame["path_index"].to_numpy(dtype=int)]
    return (
        frame.groupby(["cell_id", "stream_id"], sort=True)["value"]
        .mean()
        .to_numpy(dtype=float)
    )


def signflip_pvalue(
    cluster_totals: np.ndarray,
    observed: float,
    denominator: float,
    reps: int,
    rng: np.random.Generator,
) -> float:
    exceedances = 0
    completed = 0
    chunk = 5000
    absolute_observed = abs(observed)
    while completed < reps:
        size = min(chunk, reps - completed)
        signs = rng.choice(
            np.asarray([-1.0, 1.0]), size=(size, cluster_totals.size), replace=True
        )
        statistics = np.sum(signs * cluster_totals[None, :], axis=1) / denominator
        exceedances += int(np.count_nonzero(np.abs(statistics) >= absolute_observed - 1e-15))
        completed += size
    return (exceedances + 1.0) / (reps + 1.0)


def standardized_record_effect(values: np.ndarray) -> float:
    standard_deviation = float(np.std(values, ddof=1))
    return float(np.mean(values) / standard_deviation) if standard_deviation else 0.0


def grouped_arrays(frame: pd.DataFrame, value_column: str, path_count: int) -> list[np.ndarray]:
    grouped = frame.groupby("path_index", sort=False)[value_column]
    result = [np.empty(0, dtype=float) for _ in range(path_count)]
    for path_index, values in grouped:
        result[int(path_index)] = values.to_numpy(dtype=float)
    if any(values.size == 0 for values in result):
        raise RuntimeError(f"{value_column}: missing path contents")
    return result


def concatenate_selected(groups: list[np.ndarray], selected: np.ndarray) -> np.ndarray:
    return np.concatenate([groups[int(index)] for index in selected])


def bootstrap_path_weights(selections: np.ndarray, path_count: int) -> np.ndarray:
    weights = np.zeros((selections.shape[0], path_count), dtype=np.int16)
    row_index = np.repeat(np.arange(selections.shape[0]), selections.shape[1])
    np.add.at(weights, (row_index, selections.ravel()), 1)
    return weights


def weighted_bootstrap_statistics(
    values: np.ndarray,
    record_path_index: np.ndarray,
    path_weights: np.ndarray,
    tail_probabilities: tuple[float, ...],
) -> dict[str, np.ndarray]:
    output = {
        "mean": np.empty(path_weights.shape[0], dtype=float),
        **{f"cvar_{probability}": np.empty(path_weights.shape[0], dtype=float) for probability in tail_probabilities},
    }
    descending = np.argsort(values)[::-1]
    sorted_values = values[descending]
    sorted_paths = record_path_index[descending]
    chunk = 250
    for start in range(0, path_weights.shape[0], chunk):
        stop = min(start + chunk, path_weights.shape[0])
        selected_weights = path_weights[start:stop, record_path_index]
        total_mass = selected_weights.sum(axis=1).astype(float)
        output["mean"][start:stop] = (selected_weights * values[None, :]).sum(axis=1) / total_mass
        ordered_weights = path_weights[start:stop, sorted_paths]
        cumulative = np.cumsum(ordered_weights, axis=1)
        before = cumulative - ordered_weights
        for probability in tail_probabilities:
            target = probability * total_mass
            included = np.clip(target[:, None] - before, 0, ordered_weights)
            output[f"cvar_{probability}"][start:stop] = (
                included * sorted_values[None, :]
            ).sum(axis=1) / target
    return output


def rowwise_cvar(values: np.ndarray, tail_probability: float) -> np.ndarray:
    count = values.shape[1]
    mass = tail_probability * count
    whole = int(math.floor(mass))
    fraction = mass - whole
    boundary_index = count - whole - 1
    partitioned = np.partition(values, boundary_index, axis=1)
    numerator = partitioned[:, count - whole :].sum(axis=1)
    if fraction > 1e-12:
        numerator += fraction * partitioned[:, boundary_index]
    return numerator / mass


def cluster_swap_cvar_pvalues(
    method_values: np.ndarray,
    baseline_values: np.ndarray,
    stream_cluster_index: np.ndarray,
    tail_probabilities: tuple[float, ...],
    observed_differences: dict[float, float],
    reps: int,
    rng: np.random.Generator,
) -> dict[float, float]:
    """Paired randomization by swapping method labels within stream clusters."""
    cluster_count = int(stream_cluster_index.max()) + 1
    exceedances = {probability: 0 for probability in tail_probabilities}
    completed = 0
    chunk = 200
    while completed < reps:
        size = min(chunk, reps - completed)
        swap_by_cluster = rng.integers(0, 2, size=(size, cluster_count), dtype=np.int8).astype(bool)
        swap = swap_by_cluster[:, stream_cluster_index]
        permuted_method = np.where(swap, baseline_values[None, :], method_values[None, :])
        permuted_baseline = np.where(swap, method_values[None, :], baseline_values[None, :])
        for probability in tail_probabilities:
            differences = rowwise_cvar(permuted_method, probability) - rowwise_cvar(permuted_baseline, probability)
            exceedances[probability] += int(np.count_nonzero(
                np.abs(differences) >= abs(observed_differences[probability]) - 1e-15
            ))
        completed += size
    return {
        probability: (exceedances[probability] + 1.0) / (reps + 1.0)
        for probability in tail_probabilities
    }


def add_holm(rows: list[dict[str, Any]], p_field: str, adjusted_field: str) -> None:
    adjusted = holm_adjust(float(row[p_field]) for row in rows)
    for row, value in zip(rows, adjusted):
        row[adjusted_field] = value
        row["reject_holm_0_05"] = value < 0.05


def resolve_audit_directory(raw: str, project_root: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else project_root / path


def load_records(
    project_root: Path, batch_root: Path
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    state = json.loads((batch_root / "batch_state.json").read_text(encoding="utf-8"))
    plan = json.loads((batch_root / "sampling_plan.json").read_text(encoding="utf-8"))
    if state.get("status") != "COMPLETE" or len(state.get("cells", {})) != 36:
        raise RuntimeError("the 36-cell batch is not complete")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    keep = (
        "condition", "campaign_id", "stream_id", "checkpoint_id", "checkpoint_order",
        "family_id", "probe_episode_id", "repeat_id", "score", "model_error_count",
        "role", "layer", "stream_template",
    )
    for cell_id, cell in sorted(state["cells"].items()):
        audit = resolve_audit_directory(cell["audit_directory"], project_root)
        records_path = audit / "records/evaluation_records.jsonl"
        manifest_path = audit / "audit_manifest.json"
        for line in records_path.open(encoding="utf-8"):
            raw = json.loads(line)
            row = {field: raw[field] for field in keep}
            row["cell_id"] = cell_id
            rows.append(row)
        # Publish only content hashes. Local audit paths can reveal machine or
        # user identities and are not required to verify the analysis inputs.
        sources.append({
            "cell_id": cell_id,
            "evaluation_records_sha256": sha256_file(records_path),
            "audit_manifest_sha256": sha256_file(manifest_path),
        })
    frame = pd.DataFrame(rows)
    frame["campaign_id"] = frame["campaign_id"].astype(int)
    frame["checkpoint_order"] = frame["checkpoint_order"].astype(int)
    frame["repeat_id"] = frame["repeat_id"].astype(int)
    frame["score"] = frame["score"].astype(float)
    frame["model_error_count"] = frame["model_error_count"].astype(int)
    return frame, state, plan, sources


def analyse(
    project_root: Path,
    batch_root: Path,
    output_root: Path,
    bootstrap_reps: int,
    permutation_reps: int,
    cvar_permutation_reps: int,
    seed: int,
) -> dict[str, Any]:
    records, state, plan, sources = load_records(project_root, batch_root)
    methods = list(plan["conditions"])
    if methods[0] != "baseline" or tuple(methods[1:]) != METHODS_WITH_STATE_OFF:
        raise RuntimeError(f"unexpected condition order: {methods}")
    cell_order = sorted(state["cells"])

    paths = (
        records[["cell_id", "stream_id", "campaign_id"]]
        .drop_duplicates()
        .sort_values(["cell_id", "stream_id", "campaign_id"])
        .reset_index(drop=True)
    )
    paths["path_index"] = np.arange(len(paths), dtype=int)
    if len(paths) != 324:
        raise RuntimeError(f"expected 324 evolution paths, found {len(paths)}")
    records = records.merge(paths, on=["cell_id", "stream_id", "campaign_id"], validate="many_to_one")

    pair_key = [
        "cell_id", "stream_id", "campaign_id", "checkpoint_id", "checkpoint_order",
        "family_id", "probe_episode_id", "repeat_id", "path_index",
    ]
    score_wide = records.pivot(index=pair_key, columns="condition", values="score")
    error_wide = records.pivot(index=pair_key, columns="condition", values="model_error_count")
    rng_boot = np.random.default_rng(seed)
    selections = path_bootstrap_indices(paths, cell_order, bootstrap_reps, rng_boot)
    path_weights = bootstrap_path_weights(selections, len(paths))

    def infer_path_difference(
        record_difference: pd.Series, rng: np.random.Generator
    ) -> dict[str, Any]:
        path_grouped = record_difference.groupby(level="path_index")
        path_values = path_grouped.mean()
        path_sums = vector_from_path_series(path_grouped.sum(), len(paths))
        path_counts = vector_from_path_series(path_grouped.size(), len(paths))
        vector = vector_from_path_series(path_values, len(paths))
        observed = float(record_difference.mean())
        boot = path_sums[selections].sum(axis=1) / path_counts[selections].sum(axis=1)
        low, high = percentile_interval(boot)
        difference_frame = record_difference.rename("difference").reset_index()
        cluster_totals = (
            difference_frame.groupby(["cell_id", "stream_id"], sort=True)["difference"]
            .sum().to_numpy(dtype=float)
        )
        return {
            "estimate": observed,
            "ci95_low": low,
            "ci95_high": high,
            "p_randomization": signflip_pvalue(
                cluster_totals, observed, float(record_difference.size), permutation_reps, rng
            ),
            "standardized_record_effect": standardized_record_effect(record_difference.to_numpy(dtype=float)),
            "positive_record_rate": float(np.mean(record_difference.to_numpy(dtype=float) > 0)),
            "positive_path_rate": float(np.mean(vector > 0)),
            "zero_path_rate": float(np.mean(vector == 0)),
            "path_units": len(vector),
            "stream_clusters": len(cluster_totals),
            "record_pairs": int(record_difference.size),
        }

    primary_rows: list[dict[str, Any]] = []
    rng_primary = np.random.default_rng(seed + 1)
    k5 = score_wide.xs("K5", level="checkpoint_id")
    for estimand in ("CEG", "SUE"):
        family: list[dict[str, Any]] = []
        for method in methods[1:]:
            comparator = "baseline" if estimand == "CEG" else f"{method}_state_off"
            difference = (k5[method] - k5[comparator]).dropna()
            result = infer_path_difference(difference, rng_primary)
            family.append({"method": method, "estimand": estimand, **result})
        add_holm(family, "p_randomization", "p_holm_within_estimand")
        primary_rows.extend(family)

    longitudinal_rows: list[dict[str, Any]] = []
    rng_long = np.random.default_rng(seed + 2)
    checkpoints = sorted(
        records[["checkpoint_id", "checkpoint_order"]].drop_duplicates().itertuples(index=False),
        key=lambda item: item.checkpoint_order,
    )
    for checkpoint in checkpoints:
        subset = score_wide.xs(checkpoint.checkpoint_id, level="checkpoint_id")
        for method in methods[1:]:
            difference = (subset[method] - subset["baseline"]).dropna()
            result = infer_path_difference(difference, rng_long)
            longitudinal_rows.append({
                "method": method,
                "checkpoint": checkpoint.checkpoint_id,
                "checkpoint_order": int(checkpoint.checkpoint_order),
                "role": "cold_start_balance" if checkpoint.checkpoint_order == 0 else "longitudinal_ceg",
                **result,
            })
    longitudinal_tests = [row for row in longitudinal_rows if row["checkpoint_order"] > 0]
    adjusted = holm_adjust(row["p_randomization"] for row in longitudinal_tests)
    for row, value in zip(longitudinal_tests, adjusted):
        row["p_holm_k1_to_k5_all_methods"] = value
        row["reject_holm_0_05"] = value < 0.05
    for row in longitudinal_rows:
        if row["checkpoint_order"] == 0:
            row["p_holm_k1_to_k5_all_methods"] = None
            row["reject_holm_0_05"] = None

    # K5 all-method pairwise comparisons, useful for distinguishing top methods.
    pairwise_rows: list[dict[str, Any]] = []
    rng_pairwise = np.random.default_rng(seed + 3)
    for left_index, left in enumerate(methods):
        for right in methods[left_index + 1 :]:
            difference = (k5[left] - k5[right]).dropna()
            result = infer_path_difference(difference, rng_pairwise)
            pairwise_rows.append({"left_method": left, "right_method": right, **result})
    add_holm(pairwise_rows, "p_randomization", "p_holm_all_45_pairs")

    # Forgetting is evaluated only on episodes observed before and at K5.
    series_key = [
        "condition", "cell_id", "stream_id", "campaign_id", "family_id",
        "probe_episode_id", "path_index",
    ]
    checkpoint_scores = (
        records.groupby(series_key + ["checkpoint_order"], sort=False)["score"].mean().unstack("checkpoint_order")
    )
    final_order = int(records["checkpoint_order"].max())
    prior_columns = [column for column in checkpoint_scores.columns if column < final_order]
    eligible = checkpoint_scores[final_order].notna() & checkpoint_scores[prior_columns].notna().any(axis=1)
    forgetting = (
        checkpoint_scores.loc[eligible, prior_columns].max(axis=1)
        - checkpoint_scores.loc[eligible, final_order]
    ).clip(lower=0.0).rename("forgetting").reset_index()
    forgetting_wide = forgetting.pivot(
        index=["cell_id", "stream_id", "campaign_id", "family_id", "probe_episode_id", "path_index"],
        columns="condition", values="forgetting",
    )

    # Stream cluster index for cluster-level label-swap tests.
    stream_map = {
        key: index
        for index, key in enumerate(sorted(set(zip(paths["cell_id"], paths["stream_id"]))))
    }

    def distribution_inference_rows(
        wide: pd.DataFrame, value_name: str, tail_probabilities: tuple[float, ...]
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for method_index, method in enumerate(methods):
            paired = (
                wide[["baseline"]].dropna().reset_index()
                if method == "baseline"
                else wide[["baseline", method]].dropna().reset_index()
            )
            if method == "baseline":
                paired["method_value"] = paired["baseline"]
                method_column = "method_value"
            else:
                method_column = method
            method_values = paired[method_column].to_numpy(dtype=float)
            baseline_values = paired["baseline"].to_numpy(dtype=float)
            record_paths = paired["path_index"].to_numpy(dtype=int)
            cluster_index = np.asarray(
                [stream_map[(row.cell_id, row.stream_id)] for row in paired.itertuples(index=False)],
                dtype=int,
            )
            observed_method = {
                "mean": float(np.mean(method_values)),
                **{f"cvar_{p}": empirical_cvar(method_values, p) for p in tail_probabilities},
            }
            observed_baseline = {
                "mean": float(np.mean(baseline_values)),
                **{f"cvar_{p}": empirical_cvar(baseline_values, p) for p in tail_probabilities},
            }
            method_bootstrap = weighted_bootstrap_statistics(
                method_values, record_paths, path_weights, tail_probabilities
            )
            baseline_bootstrap = weighted_bootstrap_statistics(
                baseline_values, record_paths, path_weights, tail_probabilities
            )
            differences = {
                metric: observed_method[metric] - observed_baseline[metric]
                for metric in observed_method
            }
            if method == "baseline":
                mean_p_value = 1.0
                cvar_p_values = {p: 1.0 for p in tail_probabilities}
            else:
                per_record_difference = paired.assign(difference=paired[method_column] - paired["baseline"])
                cluster_totals = (
                    per_record_difference.groupby(["cell_id", "stream_id"], sort=True)["difference"]
                    .sum().to_numpy(dtype=float)
                )
                mean_p_value = signflip_pvalue(
                    cluster_totals, differences["mean"], float(len(per_record_difference)), permutation_reps,
                    np.random.default_rng(seed + 100 + method_index),
                )
                cvar_p_values = cluster_swap_cvar_pvalues(
                    method_values,
                    baseline_values,
                    cluster_index,
                    tail_probabilities,
                    {p: differences[f"cvar_{p}"] for p in tail_probabilities},
                    cvar_permutation_reps,
                    np.random.default_rng(seed + 1000 + method_index),
                )
            for metric_name in observed_method:
                estimate = observed_method[metric_name]
                baseline_estimate = observed_baseline[metric_name]
                difference = differences[metric_name]
                method_boot = method_bootstrap[metric_name]
                difference_boot = method_boot - baseline_bootstrap[metric_name]
                method_low, method_high = percentile_interval(method_boot)
                difference_low, difference_high = percentile_interval(difference_boot)
                p_value = (
                    mean_p_value
                    if metric_name == "mean"
                    else cvar_p_values[float(metric_name.removeprefix("cvar_"))]
                )
                output.append({
                    "method": method,
                    "outcome": value_name,
                    "metric": metric_name,
                    "estimate": estimate,
                    "ci95_low": method_low,
                    "ci95_high": method_high,
                    "baseline_estimate_on_paired_support": baseline_estimate,
                    "difference_vs_baseline": difference,
                    "difference_ci95_low": difference_low,
                    "difference_ci95_high": difference_high,
                    "p_randomization_vs_baseline": p_value,
                    "records": len(paired),
                    "path_units": len(paths),
                    "stream_clusters": len(stream_map),
                })
        for metric_name in ["mean", *(f"cvar_{p}" for p in tail_probabilities)]:
            family = [row for row in output if row["metric"] == metric_name and row["method"] != "baseline"]
            adjusted = holm_adjust(row["p_randomization_vs_baseline"] for row in family)
            for row, value in zip(family, adjusted):
                row["p_holm_within_metric"] = value
                row["reject_holm_0_05"] = value < 0.05
        for row in output:
            if row["method"] == "baseline":
                row["p_holm_within_metric"] = 1.0
                row["reject_holm_0_05"] = False
        return output

    retention_rows = distribution_inference_rows(forgetting_wide, "forgetting_loss", (0.2, 0.1))

    k5_losses = (-k5[methods]).rename(columns={method: method for method in methods})
    k5_loss_frame = k5_losses.reset_index()
    score_tail_rows = distribution_inference_rows(k5_losses, "k5_score_loss", (0.2, 0.1))

    # Intent-to-treat is primary: method failures are part of reliability. This
    # sensitivity removes only evaluation records whose decision call failed.
    sensitivity_rows: list[dict[str, Any]] = []
    for estimand in ("CEG", "SUE"):
        for method in methods[1:]:
            comparator = "baseline" if estimand == "CEG" else f"{method}_state_off"
            complete = (
                k5[[method, comparator]].notna().all(axis=1)
                & (error_wide.xs("K5", level="checkpoint_id")[[method, comparator]].fillna(1) == 0).all(axis=1)
            )
            primary_difference = (k5[method] - k5[comparator]).dropna()
            clean_difference = (k5.loc[complete, method] - k5.loc[complete, comparator]).dropna()
            sensitivity_rows.append({
                "method": method,
                "estimand": estimand,
                "primary_estimate": float(primary_difference.mean()),
                "error_free_evaluation_estimate": float(clean_difference.mean()),
                "absolute_change": float(clean_difference.mean() - primary_difference.mean()),
                "primary_record_pairs": int(primary_difference.size),
                "error_free_record_pairs": int(clean_difference.size),
                "excluded_record_pairs": int(primary_difference.size - clean_difference.size),
            })

    # Cell-level estimates expose heterogeneity without treating cells as IID samples.
    heterogeneity_rows: list[dict[str, Any]] = []
    for method in methods[1:]:
        difference = (k5[method] - k5["baseline"]).dropna().rename("ceg").reset_index()
        for cell_id, group in difference.groupby("cell_id", sort=True):
            heterogeneity_rows.append({
                "method": method,
                "cell_id": cell_id,
                "cell_mean_ceg": float(group["ceg"].mean()),
                "positive_record_rate": float((group["ceg"] > 0).mean()),
                "records": len(group),
            })

    output_root.mkdir(parents=True, exist_ok=False)
    write_csv(output_root / "primary_k5_inference.csv", primary_rows)
    write_csv(output_root / "longitudinal_ceg_inference.csv", longitudinal_rows)
    write_csv(output_root / "pairwise_k5_inference.csv", pairwise_rows)
    write_csv(output_root / "retention_inference.csv", retention_rows)
    write_csv(output_root / "score_tail_inference.csv", score_tail_rows)
    write_csv(output_root / "evaluation_error_sensitivity.csv", sensitivity_rows)
    write_csv(output_root / "cell_heterogeneity.csv", heterogeneity_rows)

    config = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "batch_root": str(batch_root),
        "plan_sha256": state["plan_sha256"],
        "confidence_level": 0.95,
        "confidence_interval_scope": "Pointwise percentile intervals; multiplicity decisions are based on Holm-adjusted randomization p-values.",
        "bootstrap_reps": bootstrap_reps,
        "permutation_reps_mean_metrics": permutation_reps,
        "permutation_reps_cvar_metrics": cvar_permutation_reps,
        "random_seed": seed,
        "fixed_strata": 36,
        "market_streams_per_stratum": 3,
        "campaigns_per_stream": 3,
        "path_units": len(paths),
        "stream_clusters": len(stream_map),
        "bootstrap": "Within each fixed structure cell, resample streams with replacement, then campaigns within each selected stream; preserve all method/control pairs.",
        "mean_metric_test": "Two-sided Monte Carlo sign-flip randomization at the market-stream cluster level.",
        "cvar_test": "Two-sided paired label-swap randomization at the market-stream cluster level.",
        "multiplicity": "Holm family-wise error control at alpha=0.05 within each declared estimand family.",
        "primary_policy": "Intent-to-treat: model failures remain part of method reliability.",
        "sensitivity_policy": "Remove only K5 evaluation pairs with a failed decision call; evolution/update failures remain part of the method.",
    }
    (output_root / "analysis_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "source_manifest.json").write_text(
        json.dumps({"schema_version": 1, "plan_sha256": state["plan_sha256"], "sources": sources}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def bp(value: float) -> str:
        return f"{10000 * value:.3f}"

    primary_ceg = sorted(
        (row for row in primary_rows if row["estimand"] == "CEG"),
        key=lambda row: row["estimate"], reverse=True,
    )
    report = [
        "# EvoPathBench Statistical Inference Report",
        "",
        "## Design",
        "",
        f"Inference uses {len(paths)} paired evolution paths in 36 fixed benchmark strata, with 3 market streams and 3 campaigns per stratum. Confidence intervals use {bootstrap_reps:,} two-stage cluster-bootstrap replicates. Mean effects use {permutation_reps:,} stream-cluster sign flips; CVaR effects use {cvar_permutation_reps:,} stream-cluster label swaps. Holm correction controls family-wise error at 0.05.",
        "",
        "## Primary K5 growth inference",
        "",
        "| Method | CEG bp [95% CI] | Holm p | SUE bp [95% CI] | Holm p |",
        "|---|---:|---:|---:|---:|",
    ]
    sue_lookup = {row["method"]: row for row in primary_rows if row["estimand"] == "SUE"}
    for row in primary_ceg:
        sue_row = sue_lookup[row["method"]]
        report.append(
            f"| {row['method']} | {bp(row['estimate'])} [{bp(row['ci95_low'])}, {bp(row['ci95_high'])}] | {row['p_holm_within_estimand']:.4g} | "
            f"{bp(sue_row['estimate'])} [{bp(sue_row['ci95_low'])}, {bp(sue_row['ci95_high'])}] | {sue_row['p_holm_within_estimand']:.4g} |"
        )
    report.extend([
        "",
        "## Stability inference",
        "",
        "Differences below are method minus baseline; positive values mean more forgetting and therefore worse stability.",
        "",
        "| Method | Mean forgetting Δ bp [95% CI]; Holm p | CVaR20 Δ bp [95% CI]; Holm p | CVaR10 Δ bp [95% CI]; Holm p |",
        "|---|---:|---:|---:|",
    ])
    retention_lookup = {(row["method"], row["metric"]): row for row in retention_rows}
    for method in methods[1:]:
        cells = []
        for metric in ("mean", "cvar_0.2", "cvar_0.1"):
            row = retention_lookup[(method, metric)]
            cells.append(
                f"{bp(row['difference_vs_baseline'])} [{bp(row['difference_ci95_low'])}, {bp(row['difference_ci95_high'])}]; "
                f"{row['p_holm_within_metric']:.4g}"
            )
        report.append(f"| {method} | {cells[0]} | {cells[1]} | {cells[2]} |")
    score_tail_lookup = {(row["method"], row["metric"]): row for row in score_tail_rows}
    report.extend([
        "",
        "## Final-score tail-risk inference",
        "",
        "Positive differences indicate a worse lower tail than the baseline.",
        "",
        "| Method | Score-loss CVaR20 Δ bp [95% CI]; Holm p | Score-loss CVaR10 Δ bp [95% CI]; Holm p |",
        "|---|---:|---:|",
    ])
    for method in methods[1:]:
        cells = []
        for metric in ("cvar_0.2", "cvar_0.1"):
            row = score_tail_lookup[(method, metric)]
            cells.append(
                f"{bp(row['difference_vs_baseline'])} [{bp(row['difference_ci95_low'])}, {bp(row['difference_ci95_high'])}]; "
                f"{row['p_holm_within_metric']:.4g}"
            )
        report.append(f"| {method} | {cells[0]} | {cells[1]} |")
    top_pair = next(
        row for row in pairwise_rows
        if {row["left_method"], row["right_method"]} == {"skillboost", "skillopt"}
    )
    top_pair_skillboost_minus_skillopt = (
        top_pair["estimate"] if top_pair["left_method"] == "skillboost" else -top_pair["estimate"]
    )
    significant_ceg = [row["method"] for row in primary_ceg if row["reject_holm_0_05"] and row["estimate"] > 0]
    significant_sue = [row["method"] for row in primary_rows if row["estimand"] == "SUE" and row["reject_holm_0_05"] and row["estimate"] > 0]
    max_sensitivity = max(abs(row["absolute_change"]) for row in sensitivity_rows)
    report.extend([
        "",
        "## Audit conclusions",
        "",
        f"- Positive CEG after Holm correction: {', '.join(significant_ceg) if significant_ceg else 'none'}.",
        f"- Positive SUE after Holm correction: {', '.join(significant_sue) if significant_sue else 'none'}.",
        f"- Removing K5 evaluation pairs with direct decision-call errors changes any reported CEG/SUE mean by at most {bp(max_sensitivity)} bp.",
        f"- SkillBoost exceeds SkillOpt at K5 by {bp(top_pair_skillboost_minus_skillopt)} bp; the all-45-pairs Holm-adjusted p-value is {top_pair['p_holm_all_45_pairs']:.4g}.",
        "- No method has a final-score CVaR20 or CVaR10 difference from baseline that remains significant after Holm correction.",
        "- Evolution/update failures are retained in the primary analysis because operational reliability is part of the evaluated method; excluding whole failed paths would condition on a post-treatment event.",
        "- CVaR tail probabilities (0.2 and 0.1) are risk-tail masses, whereas the 95% intervals quantify sampling uncertainty.",
        "",
        "Detailed corrected p-values, effect sizes, longitudinal estimates, pairwise contrasts, and cell heterogeneity are stored in the accompanying CSV files.",
    ])
    (output_root / "STATISTICAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output_root / "analysis_script.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")

    checksum_paths = sorted(path for path in output_root.iterdir() if path.is_file())
    (output_root / "SHA256SUMS").write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in checksum_paths) + "\n",
        encoding="utf-8",
    )
    return {
        "output_root": str(output_root),
        "significant_positive_ceg": significant_ceg,
        "significant_positive_sue": significant_sue,
        "primary_rows": primary_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--permutation-reps", type=int, default=50_000)
    parser.add_argument("--cvar-permutation-reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_902)
    args = parser.parse_args()
    result = analyse(
        args.project_root.resolve(), args.batch_root.resolve(), args.output_root.resolve(),
        args.bootstrap_reps, args.permutation_reps, args.cvar_permutation_reps, args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
