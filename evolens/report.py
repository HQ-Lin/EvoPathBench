"""Human-readable report rendering for benchmark run artifacts."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> List[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(_fmt(value) for value in row) + " |" for row in rows)
    return lines


def render_markdown_report(summary: Dict[str, Any], config: Dict[str, Any]) -> str:
    manifest = config.get("dataset_manifest") or {}
    hashes = manifest.get("sha256", {})
    lines: List[str] = [
        "# EvoPathBench Run Report",
        "",
        "> Scripted-baseline and small-sample outputs are harness diagnostics, not scientific conclusions or investment evidence.",
        "",
        "## Run",
        "",
        f"- Training conditions: {', '.join(config.get('conditions', []))}",
        f"- Campaigns: {config.get('campaigns', 'unknown')}",
        f"- Execution repeats: {config.get('repeats', 'unknown')}",
        f"- Streams: {len(config.get('streams', []))}",
        f"- Evaluation records: {summary.get('record_count', 0)}",
        f"- Benchmark version: {config.get('benchmark_version', 'unknown')}",
        f"- Dataset schema: {manifest.get('schema_version', 'unknown')}",
        f"- Episodes SHA-256: {hashes.get('episodes.jsonl', 'not recorded')}",
        f"- Streams SHA-256: {hashes.get('streams.jsonl', 'not recorded')}",
        "",
        "## Checkpoint summary",
        "",
    ]
    lines.extend(
        _table(
            ["stream template", "condition", "checkpoint", "n", "mean score", "drawdown", "violation rate"],
            (
                (
                    row.get("stream_template", "unspecified"),
                    row["condition"],
                    row["checkpoint_id"],
                    row["n"],
                    row["mean_score"],
                    row["mean_max_drawdown"],
                    row["violation_rate"],
                )
                for row in summary.get("primary", [])
            ),
        )
    )
    lines.extend(["", "## Paired CEG by checkpoint", ""])
    lines.extend(
        _table(
            ["stream template", "condition", "checkpoint", "family", "pairs", "method - baseline", "positive-pair rate"],
            (
                (
                    row.get("stream_template", "unspecified"),
                    row["condition"],
                    row["checkpoint_id"],
                    row["family_id"],
                    row["n_pairs"],
                    row["mean_ceg"],
                    row["positive_pair_rate"],
                )
                for row in summary.get("paired_ceg", [])
            ),
        )
    )
    lines.extend(["", "## State utilization by checkpoint", ""])
    lines.extend(
        _table(
            ["stream template", "condition", "checkpoint", "family", "pairs", "state-on - state-off", "positive-pair rate"],
            (
                (
                    row.get("stream_template", "unspecified"),
                    row["condition"],
                    row["checkpoint_id"],
                    row["family_id"],
                    row["n_pairs"],
                    row["mean_sue"],
                    row["positive_pair_rate"],
                )
                for row in summary.get("state_utilization_effect", [])
            ),
        )
    )
    lines.extend(["", "## ACQUIRE mechanism effects", ""])
    mechanism_sue = {
        (row.get("stream_template", "unspecified"), row["condition"], row["checkpoint_id"], row["mechanism"]): row
        for row in summary.get("mechanism_sue", [])
    }
    lines.extend(
        _table(
            ["stream template", "condition", "checkpoint", "mechanism", "CEG pairs", "CEG", "SUE pairs", "SUE"],
            (
                (
                    row.get("stream_template", "unspecified"),
                    row["condition"],
                    row["checkpoint_id"],
                    row["mechanism"],
                    row["n_pairs"],
                    row["mean_ceg"],
                    mechanism_sue.get(
                        (row.get("stream_template", "unspecified"), row["condition"], row["checkpoint_id"], row["mechanism"]), {}
                    ).get("n_pairs", 0),
                    mechanism_sue.get(
                        (row.get("stream_template", "unspecified"), row["condition"], row["checkpoint_id"], row["mechanism"]), {}
                    ).get("mean_sue", 0.0),
                )
                for row in summary.get("mechanism_ceg", [])
                if row["checkpoint_id"] != "K0"
            ),
        )
    )
    lines.extend(["", "## Capability erosion", ""])
    lines.extend(
        _table(
            [
                "stream template",
                "condition",
                "family",
                "paths",
                "mean forgetting",
                "worst-20% CVaR",
                "worst-10% CVaR",
            ],
            (
                (
                    row.get("stream_template", "unspecified"),
                    row["condition"],
                    row["family_id"],
                    row["n_paths"],
                    row["mean_forgetting"],
                    row["cvar_tail_0_2_forgetting"],
                    row["cvar_tail_0_1_forgetting"],
                )
                for row in summary.get("forgetting", [])
            ),
        )
    )
    lines.extend(["", "## Mechanism audit", ""])
    lines.extend(
        _table(
            [
                "stream template",
                "condition",
                "writes",
                "commits",
                "probe hit rate",
                "exact retrievals",
                "scope transfers",
                "probe applications",
                "model errors/calls",
                "tokens in/out/reasoning",
            ],
            (
                (
                    row.get("stream_template", "unspecified"),
                    row["condition"],
                    row["writes"],
                    row["committed_writes"],
                    row["probe_retrieval_hit_rate"],
                    row["probe_exact_retrievals"],
                    row["probe_scope_transfer_retrievals"],
                    row["probe_applications"],
                    (
                        f"{row.get('training_model_errors', 0) + row.get('probe_model_errors', 0)}/"
                        f"{row.get('training_model_calls', 0) + row.get('probe_model_calls', 0)}"
                    ),
                    (
                        f"{row.get('input_tokens', 0)}/{row.get('output_tokens', 0)}/"
                        f"{row.get('reasoning_tokens', 0)}"
                    ),
                )
                for row in summary.get("mechanism", [])
            ),
        )
    )
    lines.extend(["", "## Persistent artifact dynamics", ""])
    lines.extend(
        _table(
            ["stream template", "condition", "checkpoint", "states", "active", "pending", "superseded"],
            (
                (
                    row.get("stream_template", "unspecified"),
                    row["condition"],
                    row["checkpoint_id"],
                    row["n_states"],
                    row["mean_active_artifacts"],
                    row["mean_pending_hypotheses"],
                    row["mean_superseded_artifacts"],
                )
                for row in summary.get("artifact_state", [])
            ),
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- CEG compares each method with baseline on paired campaigns; it does not by itself identify which stored artifact caused the effect.",
            "- SUE compares state access on/off at the identical frozen checkpoint; a zero SUE means stored state did not change probe scores.",
            "- ACQUIRE labels identify the intended diagnostic contrast; module-level causal claims still require matching ablations and traces.",
            "- CVaR is a tail-severity estimate, not a confidence level or a per-run guarantee.",
            "- Inspect raw records, cluster dependence, failures, and sample size before making a research claim.",
            "",
        ]
    )
    return "\n".join(lines)
