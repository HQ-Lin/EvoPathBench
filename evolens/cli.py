"""Command-line entry point for generation, auditing, and evaluation."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from . import __version__
from .audit import audit_dataset
from .audit_artifacts import (
    create_audit_run,
    finalize_audit_run,
    model_call_ledger,
    state_transition_ledger,
    write_json,
    write_jsonl,
)
from .dataset import FAMILIES, LAYERS, ScenarioGenerator, load_dataset, write_dataset
from .calibration import fit_binance_calibration, load_calibration_catalog
from .calibrated_dataset import generate_binance_calibrated_dataset
from .agent import (
    DEFAULT_MODEL_URL,
    ModelCompletionClient,
    ModelConfig,
    ModelTradingAgent,
    estimate_model_calls,
    require_api_key,
)
from .market import MarketSimulator
from .market_data import download_binance_monthly_klines, freeze_binance_klines, load_replay_catalog
from .historical_dataset import generate_binance_replay_dataset
from .runner import BenchmarkRunner, write_run
from .metrics import summarize
from .report import render_markdown_report
from .streams import StreamGenerator, validate_streams
from .trajectory_store import TrajectoryStore, trajectory_id, trajectory_key


def _csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_state_off_conditions(
    conditions: List[str], value: Optional[str], disabled: bool
) -> List[str]:
    if len(conditions) != len(set(conditions)):
        raise ValueError("conditions must not contain duplicates")
    if disabled:
        return []
    requested = _csv(value)
    stateless = {"baseline", "fixed_expert", "random"}
    if requested is None:
        return [condition for condition in conditions if condition not in stateless]
    if len(requested) != len(set(requested)):
        raise ValueError("state-off conditions must not contain duplicates")
    unselected = sorted(set(requested) - set(conditions))
    if unselected:
        raise ValueError(f"state-off conditions were not selected for evaluation: {unselected}")
    invalid = sorted(set(requested) & stateless)
    if invalid:
        raise ValueError(f"stateless conditions cannot receive state-off ablations: {invalid}")
    return requested


def command_generate(args: argparse.Namespace) -> None:
    families = _csv(args.families) or list(FAMILIES)
    layers = _csv(args.layers) or list(LAYERS)
    templates = _csv(args.templates) or ["accumulation", "interference", "reversal"]
    scenarios = ScenarioGenerator(
        seed=args.seed,
        horizon=args.horizon,
        price_process_version=args.price_process_version,
    ).generate(
        instances_per_role=args.instances_per_role,
        families=families,
        layers=layers,
    )
    streams = StreamGenerator(seed=args.stream_seed).generate(scenarios, templates=templates)
    validate_streams(
        streams,
        [scenario.episode_id for scenario in scenarios],
        [scenario.episode_id for scenario in scenarios if scenario.hidden],
    )
    manifest = write_dataset(
        Path(args.output),
        scenarios,
        streams,
        {
            "seed": args.seed,
            "stream_seed": args.stream_seed,
            "horizon": args.horizon,
            "instances_per_role": args.instances_per_role,
            "families": families,
            "layers": layers,
            "templates": templates,
            "price_process_version": args.price_process_version,
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def command_freeze_binance(args: argparse.Namespace) -> None:
    manifest = freeze_binance_klines(
        [Path(item) for item in args.archive],
        Path(args.output),
        require_official_checksums=not args.allow_missing_checksum,
        require_acquisition_metadata=not args.allow_missing_acquisition_metadata,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def command_download_binance(args: argparse.Namespace) -> None:
    manifest = download_binance_monthly_klines(
        _csv(args.symbols) or [],
        _csv(args.months) or [],
        args.interval,
        Path(args.output),
        workers=args.workers,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def command_generate_binance(args: argparse.Namespace) -> None:
    manifest = generate_binance_replay_dataset(
        Path(args.snapshot),
        Path(args.output),
        horizon=args.horizon,
        stride=args.stride,
        bar_size=args.bar_size,
        instances_per_role=args.instances_per_role,
        seed=args.seed,
        stream_seed=args.stream_seed,
        templates=_csv(args.templates) or ["accumulation", "interference", "reversal", "chronological"],
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def command_generate_binance_calibrated(args: argparse.Namespace) -> None:
    archives = sorted(Path(args.archive_dir).glob("*-1m-????-??.zip"))
    result = fit_binance_calibration(
        archives,
        fit_start=args.fit_start,
        fit_end=args.fit_end,
        holdout_start=args.holdout_start,
        holdout_end=args.holdout_end,
        bar_size=args.bar_size,
        hmm_iterations=args.hmm_iterations,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_block_days=args.bootstrap_block_days,
        bootstrap_seed=args.bootstrap_seed,
    )
    manifest = generate_binance_calibrated_dataset(
        result.profile,
        result.holdout_validation,
        Path(args.output),
        instances_per_role=args.instances_per_role,
        seed=args.seed,
        stream_seed=args.stream_seed,
        horizon=args.horizon,
        bootstrap_replicates=result.bootstrap_replicates,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def _dataset_market(dataset_path: Path, manifest: dict) -> MarketSimulator:
    return MarketSimulator(
        load_replay_catalog(dataset_path, manifest),
        load_calibration_catalog(dataset_path, manifest),
    )


def command_validate(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset)
    scenarios, streams, manifest = load_dataset(dataset_path)
    audit = audit_dataset(
        scenarios,
        streams,
        sample_size=args.sample_size,
        market_simulator=_dataset_market(dataset_path, manifest),
    )
    result = {"manifest": manifest, "audit": audit}
    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def command_evaluate(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset)
    scenarios, streams, manifest = load_dataset(dataset_path)
    templates = set(_csv(args.templates) or [])
    layers = set(_csv(args.layers) or [])
    requested_stream_ids = set(_csv(args.stream_ids) or [])
    selected = [
        stream
        for stream in streams
        if (not templates or stream.template in templates)
        and (not layers or stream.layer in layers)
        and (not requested_stream_ids or stream.stream_id in requested_stream_ids)
    ]
    missing_stream_ids = requested_stream_ids - {stream.stream_id for stream in selected}
    if missing_stream_ids:
        raise ValueError(f"unknown or filtered stream ids: {sorted(missing_stream_ids)}")
    conditions = _csv(args.conditions) or [
        "baseline",
        "reflection",
        "context",
        "episodic_memory",
        "consolidated_memory",
    ]
    state_off_conditions = _resolve_state_off_conditions(
        conditions, args.state_off_conditions, args.no_state_off
    )
    result = BenchmarkRunner(
        scenarios,
        selected,
        dataset_manifest=manifest,
        market_simulator=_dataset_market(dataset_path, manifest),
    ).run(
        conditions=conditions,
        campaigns=args.campaigns,
        repeats=args.repeats,
        max_streams=args.max_streams,
        state_off_conditions=state_off_conditions,
        max_concurrency=args.max_concurrency,
    )
    paths = write_run(Path(args.output), result)
    compact = {
        "config": result["config"],
        "records": len(result["evaluation_records"]),
        "training_records": len(result["training_records"]),
        "paths": paths,
        "paired_ceg": result["summary"]["paired_ceg"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))


def _model_config(args: argparse.Namespace) -> ModelConfig:
    if args.base_url != DEFAULT_MODEL_URL and not args.allow_custom_base_url:
        raise ValueError(
            "refusing to send EVOPATHBENCH_API_KEY to a custom host; pass --allow-custom-base-url "
            "only for a trusted OpenAI-compatible endpoint"
        )
    return ModelConfig(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        top_p=args.top_p,
        api_seed=args.api_seed,
        decision_max_tokens=args.decision_max_tokens,
        reflection_max_tokens=args.reflection_max_tokens,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
        decision_interval=args.decision_interval,
        context_episodes=args.context_episodes,
        context_steps=args.context_steps,
        skillopt_edit_budget=args.skillopt_edit_budget,
        skill_document_max_chars=args.skill_document_max_chars,
        skillboost_candidate_count=args.skillboost_candidate_count,
        skillboost_max_case_regression=args.skillboost_max_case_regression,
        skillboost_max_slice_regression=args.skillboost_max_slice_regression,
        skill_validation_variants=args.skill_validation_variants,
        skillx_max_items_per_level=args.skillx_max_items_per_level,
        trace2skill_analyst_count=args.trace2skill_analyst_count,
        skillgrad_max_patterns=args.skillgrad_max_patterns,
        skillgrad_max_references=args.skillgrad_max_references,
        save_raw_responses=args.save_raw_responses,
        enable_thinking=False,
        thinking_budget=None,
        max_consecutive_failures=args.max_consecutive_failures,
    )


def _model_usage(result: dict) -> dict:
    conditioned_events = []
    parse_errors = 0
    for record in result["training_records"]:
        conditioned_events.extend(
            (record.get("condition", "unknown"), event)
            for event in record.get("agent_trace", [])
            if event.get("event") == "model_call"
        )
        parse_errors += sum(
            1 for event in record.get("agent_trace", []) if event.get("event") == "parse_error"
        )
    for record in result["evaluation_records"]:
        conditioned_events.extend(
            (record.condition, event)
            for event in record.agent_trace
            if event.get("event") == "model_call"
        )
        parse_errors += sum(1 for event in record.agent_trace if event.get("event") == "parse_error")
    token_fields = ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens")

    def attribution(event: dict) -> str:
        context = event.get("audit_context")
        if isinstance(context, dict) and context.get("token_attribution"):
            return str(context["token_attribution"])
        return "self_evolution_reflection" if event.get("phase") == "reflection" else "base_decision"

    def aggregate(items) -> dict:
        items = list(items)
        values = {field: 0 for field in token_fields}
        for _condition, event in items:
            usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
            event_values = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": int(usage.get("reasoning_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0) or input_tokens + output_tokens,
            }
            for field in token_fields:
                values[field] += event_values[field]
        return {
            "logical_calls": len(items),
            "failed_calls": sum(1 for _condition, event in items if not event.get("ok")),
            **values,
        }

    overall = aggregate(conditioned_events)
    phases = sorted({str(event.get("phase", "unknown")) for _, event in conditioned_events})
    attributions = sorted({attribution(event) for _, event in conditioned_events})
    conditions = sorted({condition for condition, _ in conditioned_events})
    by_phase = {
        phase: aggregate(
            (condition, event)
            for condition, event in conditioned_events
            if event.get("phase", "unknown") == phase
        )
        for phase in phases
    }
    by_attribution = {
        label: aggregate(
            (condition, event)
            for condition, event in conditioned_events
            if attribution(event) == label
        )
        for label in attributions
    }
    by_condition = {
        condition: aggregate(
            (item_condition, event)
            for item_condition, event in conditioned_events
            if item_condition == condition
        )
        for condition in conditions
    }
    direct = aggregate(
        (condition, event)
        for condition, event in conditioned_events
        if attribution(event) in {
            "self_evolution_reflection",
            "self_evolution_skill_optimizer",
        }
    )
    memory = by_attribution.get("memory_conditioned_decision", aggregate([]))
    context = by_attribution.get("context_conditioned_decision", aggregate([]))
    skill = by_attribution.get("skill_conditioned_decision", aggregate([]))
    validation = by_attribution.get("skill_validation_decision", aggregate([]))
    gross = {
        "logical_calls": (
            direct["logical_calls"]
            + memory["logical_calls"]
            + context["logical_calls"]
            + skill["logical_calls"]
            + validation["logical_calls"]
        ),
        "failed_calls": (
            direct["failed_calls"]
            + memory["failed_calls"]
            + context["failed_calls"]
            + skill["failed_calls"]
            + validation["failed_calls"]
        ),
        **{
            field: direct[field] + memory[field] + context[field] + skill[field] + validation[field]
            for field in token_fields
        },
    }
    return {
        **overall,
        "failed_calls": overall["failed_calls"] + parse_errors,
        "parse_errors": parse_errors,
        "by_phase": by_phase,
        "by_token_attribution": by_attribution,
        "by_condition": by_condition,
        "self_evolution": {
            "direct_update": direct,
            "direct_reflection_update": direct,
            "memory_conditioned_decisions": memory,
            "context_conditioned_decisions": context,
            "skill_conditioned_decisions": skill,
            "candidate_validation_decisions": validation,
            "gross_evolution_path": gross,
            "semantics": {
                "direct_update": "Tokens billed for explicit reflection or skill-optimizer update calls.",
                "direct_reflection_update": (
                    "Backward-compatible alias of direct_update; includes skill-optimizer updates."
                ),
                "memory_conditioned_decisions": (
                    "Full tokens on decisions that received structured persistent memory; not marginal overhead."
                ),
                "context_conditioned_decisions": (
                    "Full tokens on decisions that received the bounded sliding episode window."
                ),
                "skill_conditioned_decisions": (
                    "Full tokens on deployment decisions that received a persistent skill document."
                ),
                "candidate_validation_decisions": (
                    "Tokens spent on paired internal validation rollouts used only for skill acceptance."
                ),
                "gross_evolution_path": (
                    "Direct updates plus memory-, context-, skill-, and validation-conditioned decision tokens."
                ),
                "reasoning_tokens": "Subset of output_tokens when reported by the configured model provider.",
            },
        },
    }


def _paired_memory_token_overhead(result: dict) -> List[dict]:
    """Pair state-on/off probes to estimate the marginal persistent-memory prompt cost."""
    records = result.get("evaluation_records", [])
    by_key = {}
    for record in records:
        key = (
            record.campaign_id,
            record.stream_id,
            record.checkpoint_id,
            record.probe_episode_id,
            record.repeat_id,
            record.execution_seed,
        )
        by_key[(record.condition, key)] = record
    deltas = {}
    for record in records:
        if not record.condition.endswith("_state_off"):
            continue
        on_condition = record.condition[: -len("_state_off")]
        key = (
            record.campaign_id,
            record.stream_id,
            record.checkpoint_id,
            record.probe_episode_id,
            record.repeat_id,
            record.execution_seed,
        )
        state_on = by_key.get((on_condition, key))
        if state_on is None:
            continue
        deltas.setdefault(on_condition, []).append(
            {
                "input_tokens": state_on.input_tokens - record.input_tokens,
                "output_tokens": state_on.output_tokens - record.output_tokens,
                "reasoning_tokens": state_on.reasoning_tokens - record.reasoning_tokens,
                "total_tokens": (
                    state_on.input_tokens
                    + state_on.output_tokens
                    - record.input_tokens
                    - record.output_tokens
                ),
            }
        )
    rows = []
    for condition, values in sorted(deltas.items()):
        n_pairs = len(values)
        row = {"condition": condition, "n_pairs": n_pairs}
        for field in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"):
            total = sum(item[field] for item in values)
            row[f"total_delta_{field}"] = total
            row[f"mean_delta_{field}"] = total / n_pairs if n_pairs else 0.0
        rows.append(row)
    return rows


def _semantic_resume_descriptor(value: dict) -> dict:
    """Return the scientific compatibility key for trajectory resume."""
    normalized = dict(value)
    normalized.pop("max_concurrency", None)
    return normalized


def _open_resumable_audit_run(
    output_root: Path,
    *,
    resume: bool,
    model: str,
    conditions: List[str],
    descriptor: dict,
    dataset_manifest: dict,
    run_name: Optional[str],
):
    """Resume the newest exact protocol match, or create a fresh audit run."""
    if resume and output_root.exists():
        candidates = []
        dataset_hashes = dataset_manifest.get("sha256", {})
        for child in output_root.iterdir():
            manifest_path = child / "audit_manifest.json"
            if not child.is_dir() or not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("status") not in {"RUNNING", "FAILED"}:
                continue
            if (
                manifest.get("run_kind") != "evaluation"
                or manifest.get("model") != model
                or manifest.get("conditions") != conditions
                or _semantic_resume_descriptor(manifest.get("descriptor") or {})
                != _semantic_resume_descriptor(descriptor)
                or (manifest.get("dataset") or {}).get("sha256", {}) != dataset_hashes
            ):
                continue
            candidates.append((str(manifest.get("created_at_utc", "")), child, manifest))
        if candidates:
            _created, run_dir, manifest = sorted(candidates)[-1]
            previous = {
                "status": manifest.get("status"),
                "completed_at_utc": manifest.get("completed_at_utc"),
                "failure": manifest.get("failure"),
            }
            manifest = dict(manifest)
            manifest["status"] = "RUNNING"
            manifest["completed_at_utc"] = None
            manifest["failure"] = None
            resume_history = list(manifest.get("resume_history", []))
            resume_history.append(previous)
            manifest["resume_history"] = resume_history
            runtime_history = list(manifest.get("runtime_concurrency_history", []))
            runtime_history.append(
                {
                    "changed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "previous_max_concurrency": (manifest.get("descriptor") or {}).get(
                        "max_concurrency"
                    ),
                    "resumed_max_concurrency": descriptor.get("max_concurrency"),
                }
            )
            manifest["runtime_concurrency_history"] = runtime_history
            write_json(run_dir / "audit_manifest.json", manifest)
            for stale_name in ("AUDIT_README.md", "integrity.json", "SHA256SUMS"):
                stale = run_dir / stale_name
                if stale.exists():
                    stale.unlink()
            return run_dir, manifest, True
    run_dir, manifest = create_audit_run(
        output_root,
        run_kind="evaluation",
        model=model,
        conditions=conditions,
        descriptor=descriptor,
        dataset_manifest=dataset_manifest,
        run_name=run_name,
    )
    return run_dir, manifest, False


def command_evaluate_model(args: argparse.Namespace) -> None:
    if not 1 <= args.max_concurrency <= 64:
        raise ValueError("max_concurrency must be between 1 and 64")
    if args.max_calls < 1 or args.max_http_attempts < 1:
        raise ValueError("call budgets must be positive")
    if not 0.0 <= args.max_error_rate <= 1.0:
        raise ValueError("--max-error-rate must be between 0 and 1")
    dataset_path = Path(args.dataset)
    scenarios, streams, manifest = load_dataset(dataset_path)
    templates = set(_csv(args.templates) or [])
    layers = set(_csv(args.layers) or [])
    requested_stream_ids = set(_csv(args.stream_ids) or [])
    selected = [
        stream
        for stream in streams
        if (not templates or stream.template in templates)
        and (not layers or stream.layer in layers)
        and (not requested_stream_ids or stream.stream_id in requested_stream_ids)
    ]
    missing_stream_ids = requested_stream_ids - {stream.stream_id for stream in selected}
    if missing_stream_ids:
        raise ValueError(f"unknown or filtered stream ids: {sorted(missing_stream_ids)}")
    if args.max_streams is not None:
        selected = selected[: args.max_streams]
    if not selected:
        raise ValueError("no streams selected")
    conditions = _csv(args.conditions) or ["episodic_memory"]
    unsupported = sorted(set(conditions) - ModelTradingAgent.CONDITIONS)
    if unsupported:
        raise ValueError(f"unsupported model-agent conditions: {unsupported}")
    state_off_conditions = _resolve_state_off_conditions(
        conditions, args.state_off_conditions, args.no_state_off
    )
    config = _model_config(args)
    estimate = estimate_model_calls(
        scenarios,
        selected,
        conditions,
        config,
        args.campaigns,
        args.repeats,
        state_off_conditions,
    )
    preview = {
        "provider": "openai_compatible",
        "model": config.model,
        "api_key_env": config.api_key_env,
        "streams": [stream.stream_id for stream in selected],
        "estimate": estimate,
        "max_calls": args.max_calls,
        "max_http_attempts": args.max_http_attempts,
        "max_concurrency": args.max_concurrency,
        "thinking_enabled": False,
        "output_root": str(Path(args.output_root)),
        "run_name": args.run_name,
    }
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if estimate["logical_calls"] > args.max_calls:
        raise RuntimeError(
            f"estimated {estimate['logical_calls']} logical calls exceeds --max-calls={args.max_calls}; "
            "inspect with --dry-run, reduce the experiment, or explicitly raise the limit"
        )
    if estimate["max_http_attempts"] > args.max_http_attempts:
        raise RuntimeError(
            f"estimated {estimate['max_http_attempts']} HTTP attempts exceeds "
            f"--max-http-attempts={args.max_http_attempts}; lower retries/calls or explicitly raise the limit"
        )
    require_api_key(config)

    descriptor = {
        "dataset_path": str(Path(args.dataset).resolve()),
        "streams": [stream.stream_id for stream in selected],
        "campaigns": args.campaigns,
        "repeats": args.repeats,
        "conditions": conditions,
        "state_off_conditions": state_off_conditions,
        "max_concurrency": args.max_concurrency,
        "preflight_estimate": estimate,
        "budgets": {
            "max_calls": args.max_calls,
            "max_http_attempts": args.max_http_attempts,
            "max_error_rate": args.max_error_rate,
        },
        "remote_settings": config.public_dict(),
    }
    run_dir, audit_manifest, resumed = _open_resumable_audit_run(
        Path(args.output_root),
        resume=args.resume,
        model=config.model,
        conditions=conditions,
        descriptor=descriptor,
        dataset_manifest=manifest,
        run_name=args.run_name,
    )

    expected_trajectory_keys = [
        trajectory_key(condition, campaign_id, stream.stream_id)
        for condition in conditions
        for campaign_id in range(args.campaigns)
        for stream in selected
    ]
    trajectory_store = TrajectoryStore(
        run_dir,
        expected_trajectory_keys,
        api_key_env=config.api_key_env,
    )
    if resumed:
        print(
            json.dumps(
                {
                    "event": "resume",
                    "audit_directory": str(run_dir),
                    "completed_trajectories": len(trajectory_store.completed_ids),
                    "total_trajectories": len(expected_trajectory_keys),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    completion_client = ModelCompletionClient(
        config, max_connections=args.max_concurrency
    )

    def complete(messages, call_config, max_tokens, phase=None):
        return completion_client(
            messages, call_config, max_tokens, phase=phase
        )

    def factory(condition: str, seed: int) -> ModelTradingAgent:
        return ModelTradingAgent(
            condition=condition,
            seed=seed,
            config=config,
            completion_fn=complete,
        )

    try:
        runner = BenchmarkRunner(
            scenarios,
            selected,
            agent_factory=factory,
            dataset_manifest=manifest,
            market_simulator=_dataset_market(dataset_path, manifest),
        )
        for key in expected_trajectory_keys:
            item_id = trajectory_id(key)
            if item_id in trajectory_store.completed_ids:
                continue
            trajectory_store.begin(key)
            print(
                json.dumps(
                    {
                        "event": "trajectory_started",
                        "trajectory_id": item_id,
                        "trajectory_key": key,
                        "progress": trajectory_store.snapshot(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            try:
                condition = str(key["condition"])
                trajectory_result = runner.run(
                    conditions=[condition],
                    campaigns=1,
                    campaign_offset=int(key["campaign_id"]),
                    repeats=args.repeats,
                    stream_ids=[str(key["stream_id"])],
                    state_off_conditions=(
                        [condition] if condition in state_off_conditions else []
                    ),
                    capture_agent_traces=True,
                    max_concurrency=args.max_concurrency,
                )
                trajectory_usage = _model_usage(trajectory_result)
                trajectory_store.complete(key, trajectory_result, trajectory_usage)
                print(
                    json.dumps(
                        {
                            "event": "trajectory_completed",
                            "trajectory_id": item_id,
                            "trajectory_key": key,
                            "progress": trajectory_store.snapshot(),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except BaseException as trajectory_failure:
                trajectory_store.fail(key, trajectory_failure)
                raise

        combined_config = {
            "conditions": list(conditions),
            "state_off_conditions": list(state_off_conditions),
            "benchmark_version": __version__,
            "dataset_manifest": manifest,
            "campaigns": args.campaigns,
            "campaign_offset": 0,
            "repeats": args.repeats,
            "capture_agent_traces": True,
            "max_concurrency": args.max_concurrency,
            "streams": [stream.stream_id for stream in selected],
            "stream_templates": {
                stream.stream_id: stream.template for stream in selected
            },
        }
        result = trajectory_store.materialize(combined_config)
        result["summary"] = summarize(
            result["evaluation_records"],
            result["training_records"],
            result["state_records"],
        )
        usage = _model_usage(result)
        usage["self_evolution"]["paired_memory_prompt_overhead"] = (
            _paired_memory_token_overhead(result)
        )
        usage["self_evolution"]["semantics"]["paired_memory_prompt_overhead"] = (
            "State-on minus state-off token usage on paired frozen probes; input-token delta "
            "is the preferred estimate of marginal persistent-memory prompt cost."
        )
        error_rate = (
            usage["failed_calls"] / usage["logical_calls"]
            if usage["logical_calls"]
            else 1.0
        )
        valid = bool(usage["logical_calls"]) and error_rate <= args.max_error_rate
        result["config"]["remote_agent"] = {
            "provider": "openai_compatible",
            "settings": config.public_dict(),
            "preflight_estimate": estimate,
            "max_calls": args.max_calls,
            "max_http_attempts": args.max_http_attempts,
            "max_concurrency": args.max_concurrency,
            "audit_run_id": audit_manifest["run_id"],
        }
        result["config"]["validity"] = {
            "valid": valid,
            "model_error_rate": error_rate,
            "max_error_rate": args.max_error_rate,
        }
        result["summary"]["model_usage"] = usage
        paths = write_run(run_dir, result, audit_layout=True)
        call_ledger_path = run_dir / "ledgers" / "model_calls.jsonl"
        state_ledger_path = run_dir / "ledgers" / "state_transitions.jsonl"
        call_count = write_jsonl(call_ledger_path, model_call_ledger(result))
        transition_count = write_jsonl(state_ledger_path, state_transition_ledger(result))
        paths.update(
            {
                "model_call_ledger": str(call_ledger_path),
                "state_transition_ledger": str(state_ledger_path),
            }
        )
        terminal_status = "COMPLETE" if valid else "INVALID"
        finalize_audit_run(
            run_dir,
            audit_manifest,
            terminal_status,
            result_metadata={
                "valid": valid,
                "evaluation_records": len(result["evaluation_records"]),
                "training_records": len(result["training_records"]),
                "state_records": len(result["state_records"]),
                "model_call_ledger_rows": call_count,
                "state_transition_ledger_rows": transition_count,
                "trajectory_progress": trajectory_store.snapshot(),
                "model_usage": usage,
            },
        )
    except Exception as exc:
        finalize_audit_run(run_dir, audit_manifest, "FAILED", failure=exc)
        raise
    finally:
        completion_client.close()
    print(
        json.dumps(
            {
                "config": result["config"],
                "records": len(result["evaluation_records"]),
                "training_records": len(result["training_records"]),
                "model_usage": usage,
                "valid": valid,
                "paths": paths,
                "audit_run_id": audit_manifest["run_id"],
                "audit_directory": str(run_dir),
                "trajectory_progress": trajectory_store.snapshot(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if not valid:
        raise RuntimeError(
            f"remote run marked invalid: model error rate {error_rate:.3%} exceeds "
            f"--max-error-rate={args.max_error_rate:.3%}"
        )


def command_model_smoke(args: argparse.Namespace) -> None:
    config = _model_config(args)
    require_api_key(config)
    scenarios = ScenarioGenerator(seed=args.seed, horizon=args.horizon).generate(
        instances_per_role=1,
        families=["trend"],
        layers=["exogenous"],
    )
    scenario = next(item for item in scenarios if item.role == "learn_near")
    descriptor = {
        "synthetic_smoke": {
            "generator_seed": args.seed,
            "execution_seed": args.seed,
            "horizon": args.horizon,
            "family": "trend",
            "layer": "exogenous",
            "condition": args.condition,
        },
        "remote_settings": config.public_dict(),
        "expected_calls": (
            (args.horizon + args.decision_interval - 1) // args.decision_interval
            + (
                0
                if args.condition in {"baseline", "context"}
                else 1 + config.skillboost_candidate_count
                if args.condition == "skillboost"
                else 2
                if args.condition == "skillx"
                else config.trace2skill_analyst_count + 1
                if args.condition == "trace2skill"
                else 3
                if args.condition == "skillgrad"
                else 1
            )
        ),
    }
    run_dir, audit_manifest = create_audit_run(
        Path(args.output_root),
        run_kind="smoke",
        model=config.model,
        conditions=[args.condition],
        descriptor=descriptor,
        dataset_manifest=None,
        run_name=args.run_name,
    )
    agent = ModelTradingAgent(condition=args.condition, seed=args.seed, config=config)
    initial_state_hash = agent.state_hash()
    try:
        result = MarketSimulator().run(scenario, agent, execution_seed=args.seed)
        agent.end_episode(result, allow_state_update=True, expose_feedback=True)
        result.agent_trace = agent.get_trace()
        calls = [event for event in result.agent_trace if event.get("event") == "model_call"]
        usage = _model_usage(
            {
                "training_records": [
                    {"condition": args.condition, "agent_trace": result.agent_trace}
                ],
                "evaluation_records": [],
            }
        )
        usage["self_evolution"]["paired_memory_prompt_overhead"] = []
        usage["self_evolution"]["semantics"]["paired_memory_prompt_overhead"] = (
            "Unavailable in single-episode smoke runs because no state-off pair is executed."
        )
        final_state_hash = agent.state_hash()
        ok = bool(calls) and usage["failed_calls"] == 0
        config_record = {
            "provider": "openai_compatible",
            "settings": config.public_dict(),
            "audit_run_id": audit_manifest["run_id"],
            "usage": usage,
        }
        summary = {
            "ok": ok,
            "condition": args.condition,
            "horizon": args.horizon,
            "decision_interval": args.decision_interval,
            "score": result.score,
            "return_pct": result.return_pct,
            "max_drawdown": result.max_drawdown,
            "violation_count": len(result.violations),
            "usage": usage,
            "initial_state_hash": initial_state_hash,
            "final_state_hash": final_state_hash,
            "state_changed": initial_state_hash != final_state_hash,
        }
        write_json(run_dir / "records" / "smoke_result.json", result.to_dict())
        write_json(run_dir / "records" / "final_state.json", agent.state_snapshot())
        write_json(run_dir / "run_config.json", config_record)
        write_json(run_dir / "summary.json", summary)
        synthetic_result = {
            "training_records": [
                {
                    "condition": args.condition,
                    "campaign_id": 0,
                    "stream_id": "smoke-single-episode",
                    "event_index": 0,
                    "episode_id": result.episode_id,
                    "execution_seed": args.seed,
                    "state_hash_before": initial_state_hash,
                    "state_hash_after": final_state_hash,
                    "score": result.score,
                    "return_pct": result.return_pct,
                    "agent_trace": result.agent_trace,
                }
            ],
            "evaluation_records": [],
        }
        call_count = write_jsonl(
            run_dir / "ledgers" / "model_calls.jsonl",
            model_call_ledger(synthetic_result),
        )
        transition_count = write_jsonl(
            run_dir / "ledgers" / "state_transitions.jsonl",
            state_transition_ledger(synthetic_result),
        )
        finalize_audit_run(
            run_dir,
            audit_manifest,
            "COMPLETE" if ok else "INVALID",
            result_metadata={
                **summary,
                "model_call_ledger_rows": call_count,
                "state_transition_ledger_rows": transition_count,
            },
        )
    except Exception as exc:
        finalize_audit_run(run_dir, audit_manifest, "FAILED", failure=exc)
        raise
    compact = {
        "ok": ok,
        "condition": args.condition,
        "horizon": args.horizon,
        "decision_interval": args.decision_interval,
        "usage": usage,
        "state_hash": final_state_hash,
        "audit_run_id": audit_manifest["run_id"],
        "audit_directory": str(run_dir),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))
    if not compact["ok"]:
        raise RuntimeError(
            "model smoke test was marked INVALID; inspect audit_manifest.json and ledgers/model_calls.jsonl"
        )


def command_report(args: argparse.Namespace) -> None:
    run_dir = Path(args.run)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    report = render_markdown_report(summary, config)
    output = Path(args.output) if args.output else run_dir / "report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(output)}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evolens", description="EvoPathBench benchmark tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a task pool and longitudinal streams")
    generate.add_argument("--output", required=True)
    generate.add_argument("--seed", type=int, default=7)
    generate.add_argument("--stream-seed", type=int, default=17)
    generate.add_argument("--horizon", type=int, default=24)
    generate.add_argument("--instances-per-role", type=int, default=5)
    generate.add_argument("--price-process-version", type=int, choices=(1, 2), default=2)
    generate.add_argument("--families", help="comma-separated family ids")
    generate.add_argument("--layers", help="comma-separated exogenous,endogenous")
    generate.add_argument("--templates", help="comma-separated accumulation,interference,reversal")
    generate.set_defaults(func=command_generate)

    download_binance = subparsers.add_parser(
        "download-binance",
        help="download checksummed monthly Binance Spot archives for a frozen offline release",
    )
    download_binance.add_argument("--symbols", required=True, help="comma-separated symbols, e.g. BTCUSDT,ETHUSDT")
    download_binance.add_argument("--months", required=True, help="comma-separated YYYY-MM periods")
    download_binance.add_argument("--interval", default="1m")
    download_binance.add_argument("--output", required=True)
    download_binance.add_argument("--workers", type=int, default=4, help="bounded concurrent downloads (1-8)")
    download_binance.set_defaults(func=command_download_binance)

    freeze_binance = subparsers.add_parser(
        "freeze-binance",
        help="canonicalize downloaded official Binance Spot kline archives into an immutable snapshot",
    )
    freeze_binance.add_argument("--archive", action="append", required=True, help="official monthly kline ZIP; repeat")
    freeze_binance.add_argument("--output", required=True)
    freeze_binance.add_argument(
        "--allow-missing-checksum",
        action="store_true",
        help="development fixtures only; formal releases must verify adjacent .CHECKSUM files",
    )
    freeze_binance.add_argument(
        "--allow-missing-acquisition-metadata",
        action="store_true",
        help="development fixtures only; formal releases require downloader .SOURCE.json sidecars",
    )
    freeze_binance.set_defaults(func=command_freeze_binance)

    generate_binance = subparsers.add_parser(
        "generate-binance-replay",
        help="build a longitudinal historical replay benchmark from a frozen Binance snapshot",
    )
    generate_binance.add_argument("--snapshot", required=True)
    generate_binance.add_argument("--output", required=True)
    generate_binance.add_argument("--seed", type=int, default=7)
    generate_binance.add_argument("--stream-seed", type=int, default=17)
    generate_binance.add_argument("--horizon", type=int, default=24)
    generate_binance.add_argument("--stride", type=int, default=25)
    generate_binance.add_argument(
        "--bar-size", type=int, default=5, help="number of source 1m bars per causal decision bar"
    )
    generate_binance.add_argument("--instances-per-role", type=int, default=5)
    generate_binance.add_argument(
        "--templates", help="comma-separated accumulation,interference,reversal,chronological"
    )
    generate_binance.set_defaults(func=command_generate_binance)

    generate_calibrated = subparsers.add_parser(
        "generate-binance-calibrated",
        help="fit a frozen Binance profile and build the large semi-synthetic longitudinal track",
    )
    generate_calibrated.add_argument("--archive-dir", required=True)
    generate_calibrated.add_argument("--output", required=True)
    generate_calibrated.add_argument("--fit-start", default="2021-01")
    generate_calibrated.add_argument("--fit-end", default="2023-12")
    generate_calibrated.add_argument("--holdout-start", default="2024-01")
    generate_calibrated.add_argument("--holdout-end", default="2025-12")
    generate_calibrated.add_argument("--bar-size", type=int, default=5)
    generate_calibrated.add_argument("--horizon", type=int, default=24)
    generate_calibrated.add_argument("--instances-per-role", type=int, default=50)
    generate_calibrated.add_argument("--hmm-iterations", type=int, default=8)
    generate_calibrated.add_argument("--bootstrap-replicates", type=int, default=500)
    generate_calibrated.add_argument("--bootstrap-block-days", type=int, default=7)
    generate_calibrated.add_argument("--bootstrap-seed", type=int, default=20260317)
    generate_calibrated.add_argument("--seed", type=int, default=7)
    generate_calibrated.add_argument("--stream-seed", type=int, default=17)
    generate_calibrated.set_defaults(func=command_generate_binance_calibrated)

    validate = subparsers.add_parser("validate", help="validate hashes, structure, determinism, and conservation")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--sample-size", type=int, default=24)
    validate.add_argument("--output")
    validate.set_defaults(func=command_validate)

    evaluate = subparsers.add_parser("evaluate", help="run scripted longitudinal baselines")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--conditions", help="comma-separated benchmark conditions")
    evaluate.add_argument("--templates", help="optional comma-separated stream templates")
    evaluate.add_argument("--layers", help="optional comma-separated market layers")
    evaluate.add_argument("--stream-ids", help="optional comma-separated exact stream ids")
    evaluate.add_argument("--campaigns", type=int, default=2)
    evaluate.add_argument("--repeats", type=int, default=2)
    evaluate.add_argument("--max-streams", type=int)
    evaluate.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="parallel read-only rollout jobs (1-64); learning updates remain sequential",
    )
    evaluate.add_argument(
        "--state-off-conditions",
        help="selected conditions receiving state-read ablations; defaults to every selected stateful condition",
    )
    evaluate.add_argument("--no-state-off", action="store_true")
    evaluate.set_defaults(func=command_evaluate)

    remote = subparsers.add_parser(
        "evaluate-model",
        help="run a budget-guarded longitudinal evaluation with a remote model agent",
    )
    remote.add_argument("--dataset", required=True)
    remote.add_argument("--output-root", default="runs/model_experiments")
    remote.add_argument("--run-name", help="optional audit-safe label appended to the generated run id")
    remote.add_argument("--conditions", default="episodic_memory")
    remote.add_argument("--templates")
    remote.add_argument("--layers")
    remote.add_argument("--stream-ids", help="optional comma-separated exact stream ids")
    remote.add_argument("--campaigns", type=int, default=1)
    remote.add_argument("--repeats", type=int, default=1)
    remote.add_argument("--max-streams", type=int, default=1)
    remote.add_argument(
        "--max-concurrency",
        type=int,
        default=20,
        help="parallel frozen-probe and candidate-validation rollouts (1-64)",
    )
    remote.add_argument(
        "--state-off-conditions",
        help="selected conditions receiving state-read ablations; defaults to every selected stateful condition",
    )
    remote.add_argument("--no-state-off", action="store_true")
    remote.add_argument("--max-calls", type=int, default=500)
    remote.add_argument("--max-http-attempts", type=int, default=2000)
    remote.add_argument("--max-error-rate", type=float, default=0.05)
    remote.add_argument(
        "--resume",
        action="store_true",
        help="resume the newest exact protocol match from atomically persisted trajectories",
    )
    remote.add_argument("--dry-run", action="store_true")
    # The paper track has horizon=24, so interval=8 yields exactly three
    # policy-model decisions at steps 0, 8, and 16 in every episode.
    _add_model_arguments(remote, decision_interval=8)
    remote.set_defaults(func=command_evaluate_model)

    smoke = subparsers.add_parser(
        "model-smoke",
        help="make a bounded set of remote calls through one synthetic episode",
    )
    smoke.add_argument("--output-root", default="runs/model_experiments")
    smoke.add_argument("--run-name", help="optional audit-safe label appended to the generated run id")
    smoke.add_argument("--condition", choices=sorted(ModelTradingAgent.CONDITIONS), default="episodic_memory")
    smoke.add_argument("--horizon", type=int, default=6)
    smoke.add_argument("--seed", type=int, default=23)
    _add_model_arguments(smoke, decision_interval=2)
    smoke.set_defaults(func=command_model_smoke)

    report = subparsers.add_parser("report", help="render a Markdown report from run artifacts")
    report.add_argument("--run", required=True)
    report.add_argument("--output")
    report.set_defaults(func=command_report)
    return parser


def _add_model_arguments(parser: argparse.ArgumentParser, decision_interval: int) -> None:
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--base-url", default=DEFAULT_MODEL_URL)
    parser.add_argument("--allow-custom-base-url", action="store_true")
    parser.add_argument("--api-key-env", default="EVOPATHBENCH_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--api-seed", type=int)
    parser.add_argument("--decision-max-tokens", type=int, default=700)
    parser.add_argument("--reflection-max-tokens", type=int, default=1000)
    parser.add_argument("--decision-interval", type=int, default=decision_interval)
    parser.add_argument(
        "--context-episodes",
        type=int,
        default=5,
        help="number of completed learning episodes exposed by the Context baseline",
    )
    parser.add_argument(
        "--context-steps",
        type=int,
        default=3,
        help="maximum retained decision steps per episode in Context",
    )
    parser.add_argument(
        "--skillopt-edit-budget",
        type=int,
        default=2,
        help="maximum number of bounded text edits in each SkillOpt update",
    )
    parser.add_argument(
        "--skill-document-max-chars",
        type=int,
        default=6000,
        help="hard audit and prompt-size cap for persistent skill documents",
    )
    parser.add_argument(
        "--skillboost-candidate-count",
        type=int,
        default=4,
        help="number of prior-guided SkillBoost candidates generated per update",
    )
    parser.add_argument(
        "--skillboost-max-case-regression",
        type=float,
        default=0.25,
        help="maximum paired validation-case regression rate for SkillBoost acceptance",
    )
    parser.add_argument(
        "--skillboost-max-slice-regression",
        type=float,
        default=0.0,
        help="maximum mean-score loss allowed on any protected family/layer slice",
    )
    parser.add_argument(
        "--skill-validation-variants",
        type=int,
        default=1,
        help="deterministic internal validation variants added per arrived learning event",
    )
    parser.add_argument(
        "--skillx-max-items-per-level",
        type=int,
        default=6,
        help="maximum retained SkillX skills in each hierarchy level",
    )
    parser.add_argument(
        "--trace2skill-analyst-count",
        type=int,
        default=2,
        help="independent frozen-snapshot Trace2Skill patch proposals per update",
    )
    parser.add_argument(
        "--skillgrad-max-patterns",
        type=int,
        default=12,
        help="maximum persistent SkillGrad momentum patterns",
    )
    parser.add_argument(
        "--skillgrad-max-references",
        type=int,
        default=6,
        help="maximum conditional L3 references in the SkillGrad package",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    parser.add_argument("--save-raw-responses", action="store_true")


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
