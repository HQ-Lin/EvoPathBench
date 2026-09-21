#!/usr/bin/env python3
"""Estimate the three EvoPathBench evaluation dimensions from audited records.

The analysis keeps the benchmark structure explicit:
* accumulation streams provide held-out near and transfer probes;
* interference streams provide matched K2-to-K5 retention anchors;
* reversal streams provide hidden rule-change probes at K3, K4, and K5.

All effects are paired with the no-update baseline. Confidence intervals use the
same fixed-stratum, stream-then-campaign bootstrap as the primary analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from statistical_analysis import (
    METHODS_WITH_STATE_OFF,
    add_holm,
    bootstrap_path_weights,
    cluster_swap_cvar_pvalues,
    empirical_cvar,
    holm_adjust,
    load_records,
    path_bootstrap_indices,
    percentile_interval,
    sha256_file,
    signflip_pvalue,
    standardized_record_effect,
    vector_from_path_series,
    weighted_bootstrap_statistics,
    write_csv,
)


METHODS = ("baseline", *METHODS_WITH_STATE_OFF)


def path_universe(records: pd.DataFrame, template: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = records[records["stream_template"] == template].copy()
    paths = (
        subset[["cell_id", "stream_id", "campaign_id"]]
        .drop_duplicates()
        .sort_values(["cell_id", "stream_id", "campaign_id"])
        .reset_index(drop=True)
    )
    paths["path_index"] = np.arange(len(paths), dtype=int)
    expected = 12 * 3 * 3
    if len(paths) != expected:
        raise RuntimeError(f"{template}: expected {expected} paths, found {len(paths)}")
    subset = subset.merge(
        paths, on=["cell_id", "stream_id", "campaign_id"], validate="many_to_one"
    )
    return subset, paths


def infer_paired_mean(
    difference: pd.Series,
    paths: pd.DataFrame,
    selections: np.ndarray,
    permutation_reps: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    difference = difference.dropna()
    path_grouped = difference.groupby(level="path_index")
    path_sums = vector_from_path_series(path_grouped.sum(), len(paths))
    path_counts = vector_from_path_series(path_grouped.size(), len(paths))
    path_means = vector_from_path_series(path_grouped.mean(), len(paths))
    estimate = float(difference.mean())
    bootstrap = path_sums[selections].sum(axis=1) / path_counts[selections].sum(axis=1)
    low, high = percentile_interval(bootstrap)
    frame = difference.rename("difference").reset_index()
    cluster_totals = (
        frame.groupby(["cell_id", "stream_id"], sort=True)["difference"]
        .sum()
        .to_numpy(dtype=float)
    )
    return {
        "estimate": estimate,
        "ci95_low": low,
        "ci95_high": high,
        "p_randomization": signflip_pvalue(
            cluster_totals, estimate, float(len(difference)), permutation_reps, rng
        ),
        "standardized_record_effect": standardized_record_effect(
            difference.to_numpy(dtype=float)
        ),
        "positive_path_rate": float(np.mean(path_means > 0)),
        "record_pairs": int(len(difference)),
        "path_units": int(len(paths)),
        "stream_clusters": int(len(cluster_totals)),
    }


def score_wide(records: pd.DataFrame) -> pd.DataFrame:
    key = [
        "cell_id",
        "stream_id",
        "campaign_id",
        "checkpoint_id",
        "checkpoint_order",
        "family_id",
        "probe_episode_id",
        "repeat_id",
        "path_index",
    ]
    return records.pivot(index=key, columns="condition", values="score")


def mean_metric_rows(
    records: pd.DataFrame,
    paths: pd.DataFrame,
    selections: np.ndarray,
    permutation_reps: int,
    seed: int,
) -> list[dict[str, Any]]:
    wide = score_wide(records)
    output: list[dict[str, Any]] = []
    specifications = (
        ("held_out", "CEG_near", "accumulation", "K5", ("probe_near",), "baseline"),
        (
            "held_out",
            "CEG_transfer",
            "accumulation",
            "K5",
            ("probe_transfer",),
            "baseline",
        ),
    )
    for metric_index, (block, metric, template, checkpoint, roles, comparator_kind) in enumerate(
        specifications
    ):
        selected = records[
            (records["stream_template"] == template)
            & (records["checkpoint_id"] == checkpoint)
            & records["role"].isin(roles)
        ]
        selected_key = list(wide.index.names)
        support = selected[selected_key].drop_duplicates()
        local = wide.reset_index().merge(support, on=selected_key).set_index(selected_key)
        family: list[dict[str, Any]] = []
        for method_index, method in enumerate(METHODS_WITH_STATE_OFF):
            comparator = "baseline" if comparator_kind == "baseline" else f"{method}_state_off"
            result = infer_paired_mean(
                local[method] - local[comparator],
                paths,
                selections,
                permutation_reps,
                np.random.default_rng(seed + metric_index * 100 + method_index),
            )
            family.append({"block": block, "metric": metric, "method": method, **result})
        add_holm(family, "p_randomization", "p_holm_within_metric")
        output.extend(family)

    # State mediation is evaluated on both held-out probe types together.
    selected = records[
        (records["checkpoint_id"] == "K5")
        & records["role"].isin(("probe_near", "probe_transfer"))
    ]
    selected_key = list(wide.index.names)
    support = selected[selected_key].drop_duplicates()
    local = wide.reset_index().merge(support, on=selected_key).set_index(selected_key)
    family = []
    for method_index, method in enumerate(METHODS_WITH_STATE_OFF):
        result = infer_paired_mean(
            local[method] - local[f"{method}_state_off"],
            paths,
            selections,
            permutation_reps,
            np.random.default_rng(seed + 1000 + method_index),
        )
        family.append(
            {"block": "held_out", "metric": "SUE_heldout", "method": method, **result}
        )
    add_holm(family, "p_randomization", "p_holm_within_metric")
    output.extend(family)
    return output


def retention_rows(
    records: pd.DataFrame,
    paths: pd.DataFrame,
    selections: np.ndarray,
    permutation_reps: int,
    cvar_permutation_reps: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected = records[
        (records["role"] == "retention_anchor")
        & records["checkpoint_id"].isin(("K2", "K5"))
    ]
    key = [
        "cell_id",
        "stream_id",
        "campaign_id",
        "family_id",
        "probe_episode_id",
        "repeat_id",
        "path_index",
    ]
    wide = selected.pivot(index=key, columns=["condition", "checkpoint_id"], values="score")
    losses = pd.DataFrame(index=wide.index)
    for method in METHODS:
        losses[method] = (wide[(method, "K2")] - wide[(method, "K5")]).clip(lower=0.0)
    losses = losses.dropna(subset=list(METHODS))
    if len(losses) != len(paths):
        raise RuntimeError(
            f"retention: expected one common anchor per path ({len(paths)}), found {len(losses)}"
        )

    path_weights = bootstrap_path_weights(selections, len(paths))
    record_paths = losses.index.get_level_values("path_index").to_numpy(dtype=int)
    stream_keys = sorted(set(zip(paths["cell_id"], paths["stream_id"])))
    stream_map = {key: index for index, key in enumerate(stream_keys)}
    cluster_index = np.asarray(
        [stream_map[(row.cell_id, row.stream_id)] for row in losses.reset_index().itertuples(index=False)],
        dtype=int,
    )
    baseline_values = losses["baseline"].to_numpy(dtype=float)
    baseline_boot = weighted_bootstrap_statistics(
        baseline_values, record_paths, path_weights, (0.2, 0.1)
    )
    baseline_stats = {
        "mean": float(np.mean(baseline_values)),
        "cvar_0.2": empirical_cvar(baseline_values, 0.2),
        "cvar_0.1": empirical_cvar(baseline_values, 0.1),
    }
    output: list[dict[str, Any]] = []
    for method_index, method in enumerate(METHODS_WITH_STATE_OFF):
        values = losses[method].to_numpy(dtype=float)
        method_boot = weighted_bootstrap_statistics(values, record_paths, path_weights, (0.2, 0.1))
        observed = {
            "mean": float(np.mean(values)),
            "cvar_0.2": empirical_cvar(values, 0.2),
            "cvar_0.1": empirical_cvar(values, 0.1),
        }
        differences = {name: observed[name] - baseline_stats[name] for name in observed}
        difference_frame = losses.reset_index().assign(difference=values - baseline_values)
        cluster_totals = (
            difference_frame.groupby(["cell_id", "stream_id"], sort=True)["difference"]
            .sum()
            .to_numpy(dtype=float)
        )
        mean_p = signflip_pvalue(
            cluster_totals,
            differences["mean"],
            float(len(values)),
            permutation_reps,
            np.random.default_rng(seed + method_index),
        )
        cvar_p = cluster_swap_cvar_pvalues(
            values,
            baseline_values,
            cluster_index,
            (0.2, 0.1),
            {0.2: differences["cvar_0.2"], 0.1: differences["cvar_0.1"]},
            cvar_permutation_reps,
            np.random.default_rng(seed + 100 + method_index),
        )
        for name in ("mean", "cvar_0.2", "cvar_0.1"):
            bootstrap = method_boot[name] - baseline_boot[name]
            low, high = percentile_interval(bootstrap)
            probability = None if name == "mean" else float(name.removeprefix("cvar_"))
            output.append(
                {
                    "block": "retention",
                    "metric": f"Delta_RL_{name}",
                    "method": method,
                    "estimate": differences[name],
                    "ci95_low": low,
                    "ci95_high": high,
                    "p_randomization": mean_p if probability is None else cvar_p[probability],
                    "record_pairs": int(len(values)),
                    "path_units": int(len(paths)),
                    "stream_clusters": int(len(stream_keys)),
                }
            )
    for metric in ("Delta_RL_mean", "Delta_RL_cvar_0.2", "Delta_RL_cvar_0.1"):
        family = [row for row in output if row["metric"] == metric]
        adjusted = holm_adjust(row["p_randomization"] for row in family)
        for row, value in zip(family, adjusted):
            row["p_holm_within_metric"] = value
            row["reject_holm_0_05"] = value < 0.05
    return output


def revision_rows(
    records: pd.DataFrame,
    paths: pd.DataFrame,
    selections: np.ndarray,
    permutation_reps: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected = records[
        (records["role"] == "probe_update")
        & records["checkpoint_id"].isin(("K3", "K4", "K5"))
    ]
    key = [
        "cell_id",
        "stream_id",
        "campaign_id",
        "family_id",
        "probe_episode_id",
        "repeat_id",
        "path_index",
    ]
    wide = selected.pivot(index=key, columns=["condition", "checkpoint_id"], values="score")
    output: list[dict[str, Any]] = []
    for metric_index, (metric, post) in enumerate((("REV_1", "K4"), ("REV_2", "K5"))):
        family: list[dict[str, Any]] = []
        baseline_change = wide[("baseline", post)] - wide[("baseline", "K3")]
        for method_index, method in enumerate(METHODS_WITH_STATE_OFF):
            method_change = wide[(method, post)] - wide[(method, "K3")]
            result = infer_paired_mean(
                method_change - baseline_change,
                paths,
                selections,
                permutation_reps,
                np.random.default_rng(seed + metric_index * 100 + method_index),
            )
            family.append({"block": "revision", "metric": metric, "method": method, **result})
        add_holm(family, "p_randomization", "p_holm_within_metric")
        output.extend(family)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--permutation-reps", type=int, default=50_000)
    parser.add_argument("--cvar-permutation-reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    records, state, plan, sources = load_records(
        args.project_root.resolve(), args.batch_root.resolve()
    )
    all_rows: list[dict[str, Any]] = []
    templates: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for template_index, template in enumerate(("accumulation", "interference", "reversal")):
        subset, paths = path_universe(records, template)
        selections = path_bootstrap_indices(
            paths,
            sorted(paths["cell_id"].unique()),
            args.bootstrap_reps,
            np.random.default_rng(args.seed + template_index),
        )
        templates[template] = (subset, paths)
        if template == "accumulation":
            all_rows.extend(
                mean_metric_rows(
                    subset, paths, selections, args.permutation_reps, args.seed + 1000
                )
            )
        elif template == "interference":
            all_rows.extend(
                retention_rows(
                    subset,
                    paths,
                    selections,
                    args.permutation_reps,
                    args.cvar_permutation_reps,
                    args.seed + 2000,
                )
            )
        else:
            all_rows.extend(
                revision_rows(
                    subset, paths, selections, args.permutation_reps, args.seed + 3000
                )
            )

    # The no-update baseline defines every reported contrast.
    metric_order = (
        ("held_out", "CEG_near"),
        ("held_out", "CEG_transfer"),
        ("held_out", "SUE_heldout"),
        ("retention", "Delta_RL_mean"),
        ("retention", "Delta_RL_cvar_0.2"),
        ("retention", "Delta_RL_cvar_0.1"),
        ("revision", "REV_1"),
        ("revision", "REV_2"),
    )
    baseline_rows = [
        {
            "block": block,
            "metric": metric,
            "method": "baseline",
            "estimate": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "p_randomization": 1.0,
            "p_holm_within_metric": 1.0,
            "reject_holm_0_05": False,
        }
        for block, metric in metric_order
    ]
    rows = baseline_rows + all_rows
    args.output_root.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_root / "evaluation_dimension_metrics.csv", rows)
    metadata = {
        "schema_version": 1,
        "plan_sha256": state["plan_sha256"],
        "model": plan["model"],
        "bootstrap_reps": args.bootstrap_reps,
        "permutation_reps": args.permutation_reps,
        "cvar_permutation_reps": args.cvar_permutation_reps,
        "seed": args.seed,
        "metric_definitions": {
            "CEG_near": "K5 method-minus-baseline score on accumulation probe_near episodes.",
            "CEG_transfer": "K5 method-minus-baseline score on accumulation probe_transfer episodes.",
            "SUE_heldout": "K5 state-on minus state-off score over accumulation probe_near and probe_transfer episodes.",
            "Delta_RL_mean": "Method-minus-baseline mean positive K2-to-K5 loss on matched interference retention anchors.",
            "Delta_RL_cvar_0.2": "Method-minus-baseline CVaR of positive retention loss in the worst 20% tail.",
            "Delta_RL_cvar_0.1": "Method-minus-baseline CVaR of positive retention loss in the worst 10% tail.",
            "REV_1": "Difference-in-differences on hidden reversal probes from K3 to K4.",
            "REV_2": "Difference-in-differences on hidden reversal probes from K3 to K5.",
        },
        "source_count": len(sources),
    }
    (args.output_root / "analysis_config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_root / "source_manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "plan_sha256": state["plan_sha256"], "sources": sources},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checksum_files = sorted(path for path in args.output_root.iterdir() if path.is_file())
    (args.output_root / "SHA256SUMS").write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in checksum_files) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(args.output_root), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
