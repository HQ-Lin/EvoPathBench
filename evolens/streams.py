"""Longitudinal stream compiler for accumulation, interference, and reversal."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence, Tuple

from .models import CheckpointSpec, ScenarioSpec, StreamEvent, StreamSpec
from .mechanisms import EVOLUTION_MECHANISMS, TEMPLATE_MECHANISMS


def _group_scenarios(
    scenarios: Iterable[ScenarioSpec],
) -> DefaultDict[Tuple[str, str, str], List[ScenarioSpec]]:
    grouped: DefaultDict[Tuple[str, str, str], List[ScenarioSpec]] = defaultdict(list)
    for scenario in scenarios:
        grouped[(scenario.family_id, scenario.layer, scenario.role)].append(scenario)
    for values in grouped.values():
        values.sort(key=lambda item: item.episode_id)
    return grouped


def _probes(
    grouped: Dict[Tuple[str, str, str], List[ScenarioSpec]],
    families: Sequence[str],
    layer: str,
    include_stress: bool = True,
) -> List[str]:
    roles = ["probe_near", "probe_transfer", "retention_anchor", "shortcut_control"]
    if include_stress:
        roles.append("stress")
    result: List[str] = []
    for family in families:
        for role in roles:
            values = grouped.get((family, layer, role), [])
            if values:
                result.append(values[0].episode_id)
    return result


class StreamGenerator:
    def __init__(self, seed: int = 17) -> None:
        self.seed = seed

    def generate(
        self,
        scenarios: Sequence[ScenarioSpec],
        templates: Sequence[str] = ("accumulation", "interference", "reversal"),
    ) -> List[StreamSpec]:
        grouped = _group_scenarios(scenarios)
        families = sorted({scenario.family_id for scenario in scenarios})
        layers = sorted({scenario.layer for scenario in scenarios})
        streams: List[StreamSpec] = []
        for layer in layers:
            for focal_index, focal in enumerate(families):
                if "accumulation" in templates:
                    streams.append(self._accumulation(grouped, focal, layer, focal_index))
                if "interference" in templates and len(families) >= 4:
                    distractors = [families[(focal_index + offset) % len(families)] for offset in (1, 2, 3)]
                    streams.append(self._interference(grouped, focal, distractors, layer, focal_index))
                if "reversal" in templates:
                    streams.append(self._reversal(grouped, focal, layer, focal_index))
        return streams

    def _stream_seed(self, focal_index: int, template_index: int, layer: str) -> int:
        return self.seed * 1009 + focal_index * 97 + template_index * 13 + (1 if layer == "endogenous" else 0)

    def _learn_items(
        self,
        grouped: Dict[Tuple[str, str, str], List[ScenarioSpec]],
        family: str,
        layer: str,
        count: int,
        seed: int,
    ) -> List[ScenarioSpec]:
        values = list(grouped[(family, layer, "learn_near")])
        if len(values) < count:
            raise ValueError(f"need {count} learn episodes for {family}/{layer}, found {len(values)}")
        rng = random.Random(seed)
        rng.shuffle(values)
        return values[:count]

    def _accumulation(
        self,
        grouped: Dict[Tuple[str, str, str], List[ScenarioSpec]],
        focal: str,
        layer: str,
        focal_index: int,
    ) -> StreamSpec:
        seed = self._stream_seed(focal_index, 1, layer)
        learn = self._learn_items(grouped, focal, layer, 5, seed)
        events = [StreamEvent(item.episode_id, True, True) for item in learn]
        probes = _probes(grouped, [focal], layer)
        checkpoints = [
            CheckpointSpec("K0", 0, "cold", probes, []),
            CheckpointSpec(
                "K1",
                1,
                "after_first_experience",
                probes,
                ["outcome_attribution", "experience_compression", "memory_invocation"],
            ),
            CheckpointSpec(
                "K2",
                2,
                "after_second_evidence",
                probes,
                ["evidence_consolidation", "scope_qualification", "memory_invocation"],
            ),
            CheckpointSpec(
                "K3",
                3,
                "after_third_evidence",
                probes,
                ["evidence_consolidation", "scope_qualification", "memory_invocation"],
            ),
            CheckpointSpec(
                "K4",
                4,
                "after_fourth_evidence",
                probes,
                ["evidence_consolidation", "capability_endurance", "memory_invocation"],
            ),
            CheckpointSpec(
                "K5",
                5,
                "after_fifth_evidence",
                probes,
                ["capability_endurance", "risk_calibration", "memory_invocation"],
            ),
        ]
        return StreamSpec(
            stream_id=f"accumulation-{layer}-{focal}",
            template="accumulation",
            layer=layer,
            focal_family=focal,
            events=events,
            checkpoints=checkpoints,
            stream_seed=seed,
            target_mechanisms=list(TEMPLATE_MECHANISMS["accumulation"]),
        )

    def _interference(
        self,
        grouped: Dict[Tuple[str, str, str], List[ScenarioSpec]],
        focal: str,
        distractors: Sequence[str],
        layer: str,
        focal_index: int,
    ) -> StreamSpec:
        seed = self._stream_seed(focal_index, 2, layer)
        focal_items = self._learn_items(grouped, focal, layer, 2, seed)
        distractor_items = [self._learn_items(grouped, family, layer, 1, seed + index + 1)[0] for index, family in enumerate(distractors)]
        learn = focal_items + distractor_items
        events = [StreamEvent(item.episode_id, True, True) for item in learn]
        focal_probes = _probes(grouped, [focal], layer)
        full_probes = _probes(grouped, [focal] + list(distractors), layer, include_stress=False)
        checkpoints = [
            CheckpointSpec("K0", 0, "cold", focal_probes, []),
            CheckpointSpec(
                "K1",
                1,
                "after_first_focal_experience",
                focal_probes,
                ["experience_compression", "memory_invocation"],
            ),
            CheckpointSpec(
                "K2",
                2,
                "after_focal_consolidation",
                focal_probes,
                ["experience_compression", "evidence_consolidation", "memory_invocation"],
            ),
            CheckpointSpec(
                "K3",
                3,
                "after_first_interference",
                focal_probes,
                ["capability_endurance", "scope_qualification", "memory_invocation"],
            ),
            CheckpointSpec(
                "K4",
                4,
                "after_second_interference",
                focal_probes,
                ["capability_endurance", "scope_qualification", "memory_invocation"],
            ),
            CheckpointSpec(
                "K5",
                5,
                "after_third_interference",
                full_probes,
                ["capability_endurance", "scope_qualification", "memory_invocation"],
            ),
        ]
        return StreamSpec(
            stream_id=f"interference-{layer}-{focal}",
            template="interference",
            layer=layer,
            focal_family=focal,
            events=events,
            checkpoints=checkpoints,
            stream_seed=seed,
            target_mechanisms=list(TEMPLATE_MECHANISMS["interference"]),
        )

    def _reversal(
        self,
        grouped: Dict[Tuple[str, str, str], List[ScenarioSpec]],
        focal: str,
        layer: str,
        focal_index: int,
    ) -> StreamSpec:
        seed = self._stream_seed(focal_index, 3, layer)
        learn = self._learn_items(grouped, focal, layer, 3, seed)
        updates = list(grouped[(focal, layer, "update")])
        if len(updates) < 2:
            raise ValueError(f"need 2 update episodes for {focal}/{layer}, found {len(updates)}")
        rng = random.Random(seed + 101)
        rng.shuffle(updates)
        selected_updates = updates[:2]
        hidden_updates = list(grouped[(focal, layer, "probe_update")])
        if not hidden_updates:
            raise ValueError(f"missing hidden update probe for {focal}/{layer}")
        hidden_update = hidden_updates[seed % len(hidden_updates)]
        events = [StreamEvent(item.episode_id, True, True) for item in learn]
        events.extend(StreamEvent(item.episode_id, True, True) for item in selected_updates)
        standard_probes = _probes(grouped, [focal], layer)
        update_probes = standard_probes + [hidden_update.episode_id]
        checkpoints = [
            CheckpointSpec("K0", 0, "cold", update_probes, []),
            CheckpointSpec(
                "K1",
                1,
                "after_first_original_experience",
                update_probes,
                ["experience_compression", "memory_invocation"],
            ),
            CheckpointSpec(
                "K2",
                2,
                "after_second_original_experience",
                update_probes,
                ["evidence_consolidation", "memory_invocation", "risk_calibration"],
            ),
            CheckpointSpec(
                "K3",
                3,
                "before_reversal",
                update_probes,
                ["capability_endurance", "memory_invocation", "risk_calibration"],
            ),
            CheckpointSpec(
                "K4",
                4,
                "after_first_counterevidence",
                update_probes,
                [
                    "outcome_attribution",
                    "conflict_revision",
                    "memory_invocation",
                    "capability_endurance",
                    "risk_calibration",
                ],
            ),
            CheckpointSpec(
                "K5",
                5,
                "after_confirmed_revision",
                update_probes,
                [
                    "outcome_attribution",
                    "conflict_revision",
                    "memory_invocation",
                    "capability_endurance",
                    "risk_calibration",
                ],
            ),
        ]
        return StreamSpec(
            stream_id=f"reversal-{layer}-{focal}",
            template="reversal",
            layer=layer,
            focal_family=focal,
            events=events,
            checkpoints=checkpoints,
            stream_seed=seed,
            target_mechanisms=list(TEMPLATE_MECHANISMS["reversal"]),
        )


def validate_streams(
    streams: Sequence[StreamSpec],
    scenario_ids: Iterable[str],
    hidden_scenario_ids: Iterable[str] = (),
) -> Dict[str, Any]:
    valid_ids = set(scenario_ids)
    hidden_ids = set(hidden_scenario_ids)
    seen = set()
    event_count = 0
    probe_count = 0
    covered_mechanisms = set()
    for stream in streams:
        if stream.stream_id in seen:
            raise ValueError(f"duplicate stream_id: {stream.stream_id}")
        seen.add(stream.stream_id)
        expected_mechanisms = set(TEMPLATE_MECHANISMS.get(stream.template, ()))
        if set(stream.target_mechanisms) != expected_mechanisms:
            raise ValueError(f"mechanism targets do not match template in {stream.stream_id}")
        covered_mechanisms.update(stream.target_mechanisms)
        previous = -1
        for event in stream.events:
            if event.episode_id not in valid_ids:
                raise ValueError(f"unknown stream event: {event.episode_id}")
            event_count += 1
        for checkpoint in stream.checkpoints:
            if checkpoint.after_event < previous or checkpoint.after_event > len(stream.events):
                raise ValueError(f"invalid checkpoint order in {stream.stream_id}")
            previous = checkpoint.after_event
            unknown_targets = set(checkpoint.target_mechanisms) - set(EVOLUTION_MECHANISMS)
            if unknown_targets:
                raise ValueError(
                    f"unknown checkpoint mechanisms in {stream.stream_id}: {sorted(unknown_targets)}"
                )
            for episode_id in checkpoint.probe_episode_ids:
                if episode_id not in valid_ids:
                    raise ValueError(f"unknown probe: {episode_id}")
                if hidden_ids and episode_id not in hidden_ids:
                    raise ValueError(f"non-hidden episode used as probe: {episode_id}")
                probe_count += 1
    missing = set(EVOLUTION_MECHANISMS) - covered_mechanisms
    return {
        "stream_count": len(streams),
        "event_count": event_count,
        "probe_count": probe_count,
        "mechanism_count": len(covered_mechanisms),
        "covered_mechanisms": sorted(covered_mechanisms),
        "missing_mechanisms": sorted(missing),
    }
