"""End-to-end longitudinal benchmark runner."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .agents import BaseTradingAgent, make_agent
from . import __version__
from .dataset import scenario_index
from .market import MarketSimulator
from .metrics import summarize
from .models import EvaluationRecord, ScenarioSpec, StreamSpec


def _stable_seed(*parts: Any) -> int:
    joined = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(joined.encode("utf-8")).hexdigest()[:8], 16)


class BenchmarkRunner:
    def __init__(
        self,
        scenarios: Sequence[ScenarioSpec],
        streams: Sequence[StreamSpec],
        agent_factory: Optional[Callable[[str, int], BaseTradingAgent]] = None,
        dataset_manifest: Optional[Dict[str, Any]] = None,
        market_simulator: Optional[MarketSimulator] = None,
    ) -> None:
        self.scenarios = scenario_index(scenarios)
        self.streams = list(streams)
        self.market = market_simulator or MarketSimulator()
        self.agent_factory = agent_factory or make_agent
        self.dataset_manifest = dataset_manifest
        self.track_kind = (
            (dataset_manifest or {}).get("track", {}).get("kind")
            or (dataset_manifest or {}).get("schema_version", "unspecified")
        )

    @staticmethod
    def _parallel_map(
        function: Callable[[Any], Any], items: Sequence[Any], max_concurrency: int
    ) -> List[Any]:
        """Run independent read-only jobs with deterministic output ordering."""
        if not items:
            return []
        if max_concurrency == 1 or len(items) == 1:
            return [function(item) for item in items]
        executor = ThreadPoolExecutor(
            max_workers=min(max_concurrency, len(items)),
            thread_name_prefix="evolens-rollout",
        )
        results: List[Any] = [None] * len(items)
        pending: Dict[Any, int] = {}
        next_index = 0

        def submit_one(index: int) -> None:
            pending[executor.submit(function, items[index])] = index

        try:
            initial = min(max_concurrency, len(items))
            for next_index in range(initial):
                submit_one(next_index)
            next_index = initial
            while pending:
                completed, _not_done = wait(
                    tuple(pending), return_when=FIRST_COMPLETED
                )
                completed_indices = sorted(
                    ((pending.pop(future), future) for future in completed),
                    key=lambda pair: pair[0],
                )
                # Resolve the complete batch before replenishing it. If any
                # worker raised a fatal error, no additional jobs are started.
                for index, future in completed_indices:
                    results[index] = future.result()
                for _index, _future in completed_indices:
                    if next_index >= len(items):
                        break
                    submit_one(next_index)
                    next_index += 1
        except BaseException:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        executor.shutdown(wait=True)
        return results

    @staticmethod
    def _skill_validation_scenario(
        scenario: ScenarioSpec,
        campaign_id: int,
        stream_id: str,
        event_index: int,
        variant_index: int,
    ) -> ScenarioSpec:
        """Create a paired internal validation case from an arrived learning task.

        Only simulator seeds and the starting-price nuisance factor change.  The
        case is not a benchmark probe and its opaque ID is never put in a model
        prompt or final evaluation record.
        """
        seed = _stable_seed(
            "skill-validation", campaign_id, stream_id, event_index, variant_index
        )
        scale = 0.98 + (seed % 401) / 10_000.0
        opaque_id = hashlib.sha256(
            f"{campaign_id}|{stream_id}|{event_index}|{variant_index}".encode("utf-8")
        ).hexdigest()[:16]
        return replace(
            scenario,
            episode_id=f"internal-skill-validation-{opaque_id}",
            role="skill_validation",
            variant=f"optimizer_validation_{variant_index}",
            split="internal_validation",
            hidden=True,
            initial_price=round(scenario.initial_price * scale, 4),
            observation_mapping_seed=_stable_seed(seed, "mapping"),
            environment_seed=_stable_seed(seed, "environment"),
        )

    def _bind_skill_candidate_validator(
        self,
        agent: BaseTradingAgent,
        validation_bank: Sequence[ScenarioSpec],
        campaign_id: int,
        stream_id: str,
        event_index: int,
        max_concurrency: int,
    ) -> None:
        if not getattr(agent, "supports_candidate_validation", False):
            return
        bank = tuple(validation_bank)
        validation_protocol = (
            "arrived-historical-window-v1"
            if bank and all(item.market_source.kind == "binance_replay" for item in bank)
            else "arrived-task-seed-variant-v1"
        )

        def validator(candidates: Sequence[tuple[str, str]]) -> Dict[str, Any]:
            reports: Dict[str, Any] = {}
            combined_trace: List[Dict[str, Any]] = []
            jobs: List[Tuple[str, str, str, BaseTradingAgent, ScenarioSpec, int]] = []
            for candidate_id, skill_document in candidates:
                for case_index, scenario in enumerate(bank):
                    case_id = hashlib.sha256(
                        f"{campaign_id}|{stream_id}|{event_index}|{case_index}".encode("utf-8")
                    ).hexdigest()[:12]
                    candidate_agent = agent.clone(read_only=True)
                    candidate_agent.set_state_access(True)
                    candidate_agent.set_skill_document(skill_document)
                    candidate_agent.set_validation_audit_context(
                        {
                            "candidate_id": candidate_id,
                            "validation_case_id": case_id,
                            "validation_protocol": validation_protocol,
                        }
                    )
                    execution_seed = _stable_seed(
                        "skill-validation-execution",
                        campaign_id,
                        stream_id,
                        event_index,
                        case_index,
                    )
                    slice_id = hashlib.sha256(
                        f"{scenario.family_id}|{scenario.layer}".encode("utf-8")
                    ).hexdigest()[:10]
                    jobs.append(
                        (
                            candidate_id,
                            case_id,
                            slice_id,
                            candidate_agent,
                            scenario,
                            execution_seed,
                        )
                    )

            def run_validation_case(
                job: Tuple[str, str, str, BaseTradingAgent, ScenarioSpec, int]
            ) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
                candidate_id, case_id, slice_id, candidate_agent, scenario, execution_seed = job
                result = self.market.run(scenario, candidate_agent, execution_seed)
                traced_events: List[Dict[str, Any]] = []
                for trace_event in result.agent_trace:
                    traced = copy.deepcopy(trace_event)
                    traced["validation_candidate_id"] = candidate_id
                    traced["validation_case_id"] = case_id
                    traced_events.append(traced)
                return (
                    candidate_id,
                    {"case_id": case_id, "slice_id": slice_id, "score": result.score},
                    traced_events,
                )

            case_rows: Dict[str, List[Dict[str, Any]]] = {
                candidate_id: [] for candidate_id, _document in candidates
            }
            for candidate_id, case, traced_events in self._parallel_map(
                run_validation_case, jobs, max_concurrency
            ):
                case_rows[candidate_id].append(case)
                combined_trace.extend(traced_events)
            for candidate_id, _skill_document in candidates:
                cases = case_rows[candidate_id]
                reports[candidate_id] = {
                    "mean_score": mean(row["score"] for row in cases) if cases else 0.0,
                    "cases": cases,
                }
            return {
                "reports": reports,
                "agent_trace": combined_trace,
                "validation_protocol": validation_protocol,
            }

        agent.set_candidate_validator(validator)

    def _evaluate_probe(self, job: Dict[str, Any]) -> EvaluationRecord:
        """Execute one frozen probe episode; jobs are independent by construction."""
        result = self.market.run(
            job["scenario"],
            job["agent"],
            execution_seed=job["execution_seed"],
        )
        retrieval_events = [
            event for event in result.agent_trace if event.get("event") == "retrieve"
        ]
        application_events = [
            event for event in result.agent_trace if event.get("event") == "apply"
        ]
        action_events = [
            event for event in result.agent_trace if event.get("event") == "action"
        ]
        model_events = [
            event for event in result.agent_trace if event.get("event") == "model_call"
        ]
        parse_error_events = [
            event for event in result.agent_trace if event.get("event") == "parse_error"
        ]
        scenario = job["scenario"]
        return EvaluationRecord(
            condition=job["condition"],
            campaign_id=job["campaign_id"],
            stream_id=job["stream_id"],
            checkpoint_id=job["checkpoint_id"],
            checkpoint_order=job["checkpoint_order"],
            probe_episode_id=job["probe_episode_id"],
            family_id=scenario.family_id,
            role=scenario.role,
            layer=scenario.layer,
            repeat_id=job["repeat_id"],
            score=result.score,
            return_pct=result.return_pct,
            final_wealth=result.final_wealth,
            max_drawdown=result.max_drawdown,
            turnover=result.turnover,
            violation_count=len(result.violations),
            fee_paid=result.fee_paid,
            state_hash=job["state_hash"],
            strategy_mode=result.strategy_mode,
            stream_template=job["stream_template"],
            track_kind=self.track_kind,
            retrieval_count=len(retrieval_events),
            retrieval_hit_count=sum(1 for event in retrieval_events if event.get("hit")),
            exact_retrieval_count=sum(
                1 for event in retrieval_events if event.get("match_type") == "exact"
            ),
            scope_transfer_retrieval_count=sum(
                1
                for event in retrieval_events
                if event.get("match_type") == "scope_transfer"
            ),
            application_count=len(application_events),
            action_count=len(action_events),
            execution_seed=job["execution_seed"],
            mechanism_ids=sorted(
                set(scenario.evolution_mechanisms) | set(job["checkpoint_mechanisms"])
            ),
            model_call_count=len(model_events),
            model_error_count=sum(1 for event in model_events if event.get("error"))
            + len(parse_error_events),
            input_tokens=sum(
                int(event.get("usage", {}).get("input_tokens", 0) or 0)
                for event in model_events
            ),
            output_tokens=sum(
                int(event.get("usage", {}).get("output_tokens", 0) or 0)
                for event in model_events
            ),
            reasoning_tokens=sum(
                int(event.get("usage", {}).get("reasoning_tokens", 0) or 0)
                for event in model_events
            ),
            agent_trace=result.agent_trace if job["capture_agent_traces"] else [],
        )

    def run(
        self,
        conditions: Sequence[str],
        campaigns: int = 2,
        campaign_offset: int = 0,
        repeats: int = 2,
        stream_ids: Optional[Sequence[str]] = None,
        max_streams: Optional[int] = None,
        state_off_conditions: Optional[Sequence[str]] = None,
        capture_agent_traces: bool = False,
        max_concurrency: int = 1,
    ) -> Dict[str, Any]:
        if not conditions:
            raise ValueError("at least one evaluation condition is required")
        if len(conditions) != len(set(conditions)):
            raise ValueError("evaluation conditions must not contain duplicates")
        if not 1 <= max_concurrency <= 64:
            raise ValueError("max_concurrency must be between 1 and 64")
        if state_off_conditions is None:
            default_stateful = {
                "reflection",
                "context",
                "episodic_memory",
                "consolidated_memory",
                "skillopt",
                "skillboost",
                "skillx",
                "trace2skill",
                "skillgrad",
            }
            state_off_conditions = tuple(
                condition for condition in conditions if condition in default_stateful
            )
        if len(state_off_conditions) != len(set(state_off_conditions)):
            raise ValueError("state-off conditions must not contain duplicates")
        unselected_state_off = sorted(set(state_off_conditions) - set(conditions))
        if unselected_state_off:
            raise ValueError(
                f"state-off conditions were not selected: {unselected_state_off}"
            )
        selected = [stream for stream in self.streams if stream_ids is None or stream.stream_id in set(stream_ids)]
        if max_streams is not None:
            selected = selected[:max_streams]
        if not selected:
            raise ValueError("no streams selected")
        if campaigns < 1 or repeats < 1:
            raise ValueError("campaigns and repeats must be positive")
        if campaign_offset < 0:
            raise ValueError("campaign_offset must be non-negative")
        evaluation_records: List[EvaluationRecord] = []
        training_records: List[Dict[str, Any]] = []
        state_records: List[Dict[str, Any]] = []
        for condition in conditions:
            for local_campaign_id in range(campaigns):
                campaign_id = campaign_offset + local_campaign_id
                for stream in selected:
                    agent_seed = _stable_seed("agent", campaign_id, stream.stream_id)
                    agent = self.agent_factory(condition, agent_seed)
                    skill_validation_bank: List[ScenarioSpec] = []
                    for event_count in range(len(stream.events) + 1):
                        for checkpoint_order, checkpoint in enumerate(stream.checkpoints):
                            if checkpoint.after_event != event_count:
                                continue
                            state_hash = agent.state_hash()
                            state_records.append(
                                {
                                    "condition": condition,
                                    "campaign_id": campaign_id,
                                    "stream_id": stream.stream_id,
                                    "stream_template": stream.template,
                                    "track_kind": self.track_kind,
                                    "layer": stream.layer,
                                    "checkpoint_id": checkpoint.checkpoint_id,
                                    "checkpoint_order": checkpoint_order,
                                    "state_hash": state_hash,
                                    "state": agent.state_snapshot(),
                                }
                            )
                            frozen = agent.clone(read_only=True)
                            evaluation_variants = [(condition, frozen)]
                            if condition in state_off_conditions and frozen.supports_state_ablation:
                                state_off = frozen.clone(read_only=True)
                                state_off.set_state_access(False)
                                off_condition = f"{condition}_state_off"
                                evaluation_variants.append((off_condition, state_off))
                                state_records.append(
                                    {
                                        "condition": off_condition,
                                        "source_condition": condition,
                                        "campaign_id": campaign_id,
                                        "stream_id": stream.stream_id,
                                        "stream_template": stream.template,
                                        "track_kind": self.track_kind,
                                        "layer": stream.layer,
                                        "checkpoint_id": checkpoint.checkpoint_id,
                                        "checkpoint_order": checkpoint_order,
                                        "state_hash": state_hash,
                                        "state_access": False,
                                        "state": agent.state_snapshot(),
                                    }
                                )
                            probe_jobs: List[Dict[str, Any]] = []
                            for probe_id in checkpoint.probe_episode_ids:
                                scenario = self.scenarios[probe_id]
                                for repeat_id in range(repeats):
                                    for evaluation_condition, evaluation_agent in evaluation_variants:
                                        execution_seed = _stable_seed(
                                            "probe",
                                            campaign_id,
                                            stream.stream_id,
                                            checkpoint.checkpoint_id,
                                            probe_id,
                                            repeat_id,
                                        )
                                        probe_jobs.append(
                                            {
                                                "condition": evaluation_condition,
                                                "campaign_id": campaign_id,
                                                "stream_id": stream.stream_id,
                                                "stream_template": stream.template,
                                                "checkpoint_id": checkpoint.checkpoint_id,
                                                "checkpoint_order": checkpoint_order,
                                                "checkpoint_mechanisms": checkpoint.target_mechanisms,
                                                "probe_episode_id": probe_id,
                                                "repeat_id": repeat_id,
                                                "scenario": scenario,
                                                "agent": evaluation_agent.clone(read_only=True),
                                                "execution_seed": execution_seed,
                                                "state_hash": state_hash,
                                                "capture_agent_traces": capture_agent_traces,
                                            }
                                        )
                            evaluation_records.extend(
                                self._parallel_map(
                                    self._evaluate_probe, probe_jobs, max_concurrency
                                )
                            )
                        if event_count == len(stream.events):
                            continue
                        event = stream.events[event_count]
                        scenario = self.scenarios[event.episode_id]
                        if getattr(agent, "supports_candidate_validation", False):
                            variants = int(getattr(agent, "candidate_validation_variants", 1))
                            if scenario.market_source.kind == "binance_replay":
                                # A seed change cannot create another historical
                                # tape. Use one distinct, already-arrived window
                                # per event and accumulate a real replay bank.
                                skill_validation_bank.append(
                                    self._skill_validation_scenario(
                                        scenario,
                                        campaign_id,
                                        stream.stream_id,
                                        event_count,
                                        0,
                                    )
                                )
                            else:
                                skill_validation_bank.extend(
                                    self._skill_validation_scenario(
                                        scenario,
                                        campaign_id,
                                        stream.stream_id,
                                        event_count,
                                        variant_index,
                                    )
                                    for variant_index in range(variants)
                                )
                            self._bind_skill_candidate_validator(
                                agent,
                                skill_validation_bank,
                                campaign_id,
                                stream.stream_id,
                                event_count,
                                max_concurrency,
                            )
                        execution_seed = _stable_seed("learn", campaign_id, stream.stream_id, event_count)
                        state_hash_before = agent.state_hash()
                        result = self.market.run(scenario, agent, execution_seed=execution_seed)
                        agent.end_episode(result, event.allow_state_update, event.expose_feedback)
                        training_records.append(
                            {
                                "condition": condition,
                                "campaign_id": campaign_id,
                                "stream_id": stream.stream_id,
                                "stream_template": stream.template,
                                "track_kind": self.track_kind,
                                "layer": scenario.layer,
                                "event_index": event_count,
                                "episode_id": scenario.episode_id,
                                "family_id": scenario.family_id,
                                "role": scenario.role,
                                "score": result.score,
                                "return_pct": result.return_pct,
                                "violations": result.violations,
                                "feature_signature": result.feature_signature,
                                "execution_seed": execution_seed,
                                "state_hash_before": state_hash_before,
                                "state_hash_after": agent.state_hash(),
                                "agent_trace": agent.get_trace(),
                            }
                        )
        return {
            "config": {
                "conditions": list(conditions),
                "state_off_conditions": list(state_off_conditions),
                "benchmark_version": __version__,
                "dataset_manifest": self.dataset_manifest,
                "campaigns": campaigns,
                "campaign_offset": campaign_offset,
                "repeats": repeats,
                "capture_agent_traces": capture_agent_traces,
                "max_concurrency": max_concurrency,
                "streams": [stream.stream_id for stream in selected],
                "stream_templates": {stream.stream_id: stream.template for stream in selected},
            },
            "evaluation_records": evaluation_records,
            "training_records": training_records,
            "state_records": state_records,
            "summary": summarize(evaluation_records, training_records, state_records),
        }


def write_run(
    output_dir: Path,
    result: Dict[str, Any],
    audit_layout: bool = False,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_raw = bool(
        result.get("config", {})
        .get("remote_agent", {})
        .get("settings", {})
        .get("save_raw_responses", False)
    )
    if save_raw or audit_layout:
        output_dir.chmod(0o700)
    key_env = (
        result.get("config", {})
        .get("remote_agent", {})
        .get("settings", {})
        .get("api_key_env")
    )
    secret = os.getenv(key_env, "") if key_env else ""

    def checked_json(value: Any, *, indent: Optional[int] = None) -> str:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )
        if secret and secret in serialized:
            raise RuntimeError("secret-leak guard rejected a run artifact")
        return serialized
    records_dir = output_dir / "records" if audit_layout else output_dir
    records_dir.mkdir(parents=True, exist_ok=True)
    if audit_layout:
        records_dir.chmod(0o700)
    paths = {
        "evaluation_records": records_dir / "evaluation_records.jsonl",
        "training_records": records_dir / "training_records.jsonl",
        "state_records": records_dir / "state_records.jsonl",
        "summary": output_dir / "summary.json",
        "run_config": output_dir / "run_config.json",
    }
    with paths["evaluation_records"].open("w", encoding="utf-8") as handle:
        for record in result["evaluation_records"]:
            handle.write(checked_json(record.to_dict()) + "\n")
    with paths["training_records"].open("w", encoding="utf-8") as handle:
        for record in result["training_records"]:
            handle.write(checked_json(record) + "\n")
    with paths["state_records"].open("w", encoding="utf-8") as handle:
        for record in result["state_records"]:
            handle.write(checked_json(record) + "\n")
    paths["summary"].write_text(
        checked_json(result["summary"], indent=2) + "\n",
        encoding="utf-8",
    )
    paths["run_config"].write_text(
        checked_json(result["config"], indent=2) + "\n",
        encoding="utf-8",
    )
    if save_raw or audit_layout:
        for path in paths.values():
            path.chmod(0o600)
    return {key: str(path) for key, path in paths.items()}
