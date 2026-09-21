"""Scripted baselines and the adapter contract for future LLM agents."""
from __future__ import annotations

import copy
import hashlib
import json
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .features import feature_signature
from .models import EpisodeResult, Observation, OrderIntent, ScenarioSpec
from .persistence import StrategyEntry, StrategyStore

MODES: Tuple[str, ...] = ("momentum", "mean_reversion", "fundamental", "conservative")


class BaseTradingAgent:
    """Minimal adapter contract. LLM adapters should implement the same methods."""

    current_mode: str
    supports_state_ablation: bool = False

    def begin_episode(self, scenario: ScenarioSpec, execution_seed: int) -> None:
        raise NotImplementedError

    def act(self, observation: Observation) -> List[OrderIntent]:
        raise NotImplementedError

    def end_episode(self, result: EpisodeResult, allow_state_update: bool, expose_feedback: bool) -> None:
        raise NotImplementedError

    def get_trace(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def state_hash(self) -> str:
        raise NotImplementedError

    def state_snapshot(self) -> Dict[str, Any]:
        raise NotImplementedError

    def set_state_access(self, enabled: bool) -> None:
        """Toggle reads from persistent state for paired checkpoint ablations."""
        if not self.supports_state_ablation:
            raise NotImplementedError("agent does not implement state-access ablation")

    def clone(self, read_only: bool = True) -> "BaseTradingAgent":
        clone = copy.deepcopy(self)
        clone.read_only = read_only
        return clone


class StaticTradingAgent(BaseTradingAgent):
    def __init__(self, mode: str = "conservative", seed: int = 0) -> None:
        if mode not in MODES and mode != "random":
            raise ValueError(f"unsupported strategy mode: {mode}")
        self.fixed_mode = mode
        self.current_mode = mode
        self.seed = seed
        self.read_only = True
        self._trace: List[Dict[str, Any]] = []
        self._rng = random.Random(seed)

    def begin_episode(self, scenario: ScenarioSpec, execution_seed: int) -> None:
        self.current_mode = self.fixed_mode
        self._trace = []
        self._rng = random.Random(self.seed ^ execution_seed)

    def _target_position(self, observation: Observation) -> int:
        contract = observation.risk_contract
        base = contract.initial_position
        high = min(contract.max_abs_position, base + 15)
        low = 0 if not contract.allow_short else max(-contract.max_abs_position, base - 25)
        if observation.max_drawdown_so_far >= 0.80 * contract.max_drawdown:
            return min(base, observation.position)
        if self.current_mode == "random":
            return self._rng.choice((low, base, high))
        if self.current_mode == "momentum":
            if len(observation.prices) < 3:
                return base
            move = observation.prices[-1] / observation.prices[-3] - 1.0
            if move > 0.002:
                return high
            if move < -0.002:
                return low
            return base
        gap = observation.price / max(observation.public_signal, 1e-9) - 1.0
        threshold = 0.008 if self.current_mode == "mean_reversion" else 0.004
        if self.current_mode in {"mean_reversion", "fundamental"}:
            if gap < -threshold:
                return high
            if gap > threshold:
                return low
            return base
        # Conservative: only reduce large deviations and otherwise stay close to the initial allocation.
        if gap > 0.02:
            return max(low, base - 5)
        if gap < -0.02 and observation.max_drawdown_so_far < 0.5 * contract.max_drawdown:
            return min(high, base + 5)
        return base

    def act(self, observation: Observation) -> List[OrderIntent]:
        target = self._target_position(observation)
        delta = target - observation.position
        if delta == 0:
            return []
        quantity = min(abs(delta), observation.risk_contract.max_order_size)
        side = "buy" if delta > 0 else "sell"
        self._trace.append(
            {
                "event": "action",
                "step": observation.step,
                "mode": self.current_mode,
                "side": side,
                "quantity": quantity,
            }
        )
        return [OrderIntent(side=side, quantity=quantity, tag=self.current_mode)]

    def end_episode(self, result: EpisodeResult, allow_state_update: bool, expose_feedback: bool) -> None:
        return None

    def get_trace(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._trace)

    def state_hash(self) -> str:
        return hashlib.sha256(f"static:{self.fixed_mode}".encode("utf-8")).hexdigest()

    def state_snapshot(self) -> Dict[str, Any]:
        return {"type": "static", "mode": self.fixed_mode, "state_hash": self.state_hash()}


class OracleFamilyAgent(StaticTradingAgent):
    """Diagnostic oracle that may read family labels. Never use it as a benchmarked agent."""

    FAMILY_MODE = {
        "trend": "momentum",
        "mean_reversion": "mean_reversion",
        "liquidity": "conservative",
        "event_jump": "fundamental",
        "opponent_reflexivity": "fundamental",
        "risk_contract": "conservative",
    }

    def begin_episode(self, scenario: ScenarioSpec, execution_seed: int) -> None:
        self.fixed_mode = self.FAMILY_MODE.get(scenario.family_id, "conservative")
        super().begin_episode(scenario, execution_seed)


class AdaptiveTradingAgent(StaticTradingAgent):
    """A cheap stateful baseline used to validate the longitudinal harness.

    It is intentionally simple and is not proposed as a self-evolution algorithm.
    """

    CONDITIONS = {
        "baseline",
        "reflection",
        "context",
        "episodic_memory",
        "consolidated_memory",
    }
    supports_state_ablation = True

    def __init__(
        self,
        condition: str,
        seed: int = 0,
        max_entries: int = 12,
        context_episodes: int = 5,
    ) -> None:
        if condition not in self.CONDITIONS:
            raise ValueError(f"unsupported condition: {condition}")
        super().__init__(mode="momentum", seed=seed)
        if context_episodes < 1:
            raise ValueError("context_episodes must be positive")
        self.condition = condition
        self.supports_state_ablation = condition != "baseline"
        self.store = StrategyStore(max_entries=max_entries)
        self.reflection: List[Dict[str, Any]] = []
        self.context_episodes = context_episodes
        self.context: List[Dict[str, Any]] = []
        self.read_only = False
        self.state_access_enabled = True
        self._retrieved = False
        self._retrieved_rule: Optional[str] = None
        self._retrieval_match: str = "miss"

    def begin_episode(self, scenario: ScenarioSpec, execution_seed: int) -> None:
        super().begin_episode(scenario, execution_seed)
        self.current_mode = "momentum"
        self._retrieved = False
        self._retrieved_rule = None
        self._retrieval_match = "miss"

    def _retrieve_if_ready(self, observation: Observation) -> None:
        if self._retrieved or len(observation.prices) < 4:
            return
        signature = feature_signature(
            observation.prices,
            observation.fundamentals,
            observation.spread_bps,
            observation.liquidity,
        )
        selected_mode: Optional[str] = None
        rule_id: Optional[str] = None
        match_type = "miss"
        if not self.state_access_enabled:
            self._retrieved = True
            self._trace.append(
                {
                    "event": "retrieve",
                    "signature": signature,
                    "hit": False,
                    "disabled": True,
                    "rule_id": None,
                    "mechanism": "memory_invocation",
                }
            )
            return
        if self.condition in {"episodic_memory", "consolidated_memory"}:
            entry, match_type = self.store.retrieve_with_match(signature)
            if entry is not None:
                selected_mode = entry.mode
                rule_id = entry.rule_id
        elif self.condition == "reflection":
            candidates = [item for item in self.reflection if item["signature"] == signature]
            if candidates:
                best = max(candidates, key=lambda item: (item["score"], item["episode_id"]))
                selected_mode = best["mode"]
                rule_id = f"raw:{best['episode_id']}"
                match_type = "exact"
        elif self.condition == "context":
            candidates = [item for item in self.context if item["signature"] == signature]
            if candidates:
                latest = candidates[-1]
                selected_mode = latest["mode"]
                rule_id = f"window:{latest['episode_id']}"
                match_type = "exact"
        self._retrieved = True
        hit = selected_mode is not None
        self._trace.append(
            {
                "event": "retrieve",
                "signature": signature,
                "hit": hit,
                "rule_id": rule_id,
                "match_type": match_type,
                "mechanism": "memory_invocation",
                "mechanisms": ["memory_invocation"]
                + (["scope_qualification"] if match_type == "scope_transfer" else []),
            }
        )
        if selected_mode is not None:
            previous = self.current_mode
            self.current_mode = selected_mode
            self._retrieved_rule = rule_id
            self._retrieval_match = match_type
            self._trace.append(
                {
                    "event": "apply",
                    "rule_id": rule_id,
                    "from_mode": previous,
                    "to_mode": selected_mode,
                    "match_type": match_type,
                    "mechanism": "memory_invocation",
                    "mechanisms": ["memory_invocation"]
                    + (["scope_qualification"] if match_type == "scope_transfer" else []),
                }
            )

    def set_state_access(self, enabled: bool) -> None:
        self.state_access_enabled = bool(enabled)

    def act(self, observation: Observation) -> List[OrderIntent]:
        self._retrieve_if_ready(observation)
        return super().act(observation)

    @staticmethod
    def _next_mode(mode: str) -> str:
        index = MODES.index(mode) if mode in MODES else 0
        return MODES[(index + 1) % len(MODES)]

    def end_episode(self, result: EpisodeResult, allow_state_update: bool, expose_feedback: bool) -> None:
        if not allow_state_update or not expose_feedback or self.read_only or self.condition == "baseline":
            return
        if self.condition in {"episodic_memory", "consolidated_memory"} and self._retrieved_rule is not None:
            attributed = self.store.record_application_outcome(
                self._retrieved_rule,
                result.episode_id,
                result.score,
            )
            self._trace.append(
                {
                    "event": "attribute",
                    "mechanism": "outcome_attribution",
                    "rule_id": self._retrieved_rule,
                    "match_type": self._retrieval_match,
                    "score": result.score,
                    "recorded": attributed,
                }
            )
        candidate_mode = self.current_mode if result.score >= 0.0 else self._next_mode(self.current_mode)
        evidence_score = result.score if candidate_mode == self.current_mode else min(0.0, result.score) / 2.0
        if self.condition == "context":
            self.context.append(
                {
                    "episode_id": result.episode_id,
                    "signature": result.feature_signature,
                    "mode": candidate_mode,
                    "score": evidence_score,
                }
            )
            self.context = self.context[-self.context_episodes :]
            self._trace.append(
                {
                    "event": "write",
                    "artifact": "context",
                    "signature": result.feature_signature,
                    "mode": candidate_mode,
                    "committed": True,
                    "history_episode_count": len(self.context),
                    "mechanism": "context_injection",
                }
            )
            return
        if self.condition == "reflection":
            self.reflection.append(
                {
                    "episode_id": result.episode_id,
                    "signature": result.feature_signature,
                    "mode": candidate_mode,
                    "score": evidence_score,
                }
            )
            if len(self.reflection) > self.store.max_entries:
                self.reflection = self.reflection[-self.store.max_entries :]
            self._trace.append(
                {
                    "event": "write",
                    "artifact": "reflection",
                    "signature": result.feature_signature,
                    "mode": candidate_mode,
                    "committed": True,
                    "mechanism": "experience_compression",
                }
            )
            return
        gate = 2 if self.condition == "consolidated_memory" else 1
        target_store = self.store
        existing = target_store.retrieve(result.feature_signature)
        if existing is None:
            mechanism = "experience_compression"
        elif existing.mode == candidate_mode:
            mechanism = "evidence_consolidation"
        else:
            mechanism = "conflict_revision"
        committed, entry = target_store.propose(
            result.feature_signature,
            candidate_mode,
            result.episode_id,
            evidence_score,
            gate_evidence=gate,
        )
        self._trace.append(
            {
                "event": "write",
                "artifact": "strategy",
                "signature": result.feature_signature,
                "mode": candidate_mode,
                "committed": committed,
                "rule_id": entry.rule_id if entry is not None else None,
                "mechanism": mechanism,
                "gate_evidence": gate,
            }
        )

    def state_snapshot(self) -> Dict[str, Any]:
        return {
            "type": "adaptive",
            "condition": self.condition,
            "store": self.store.to_dict(),
            "reflection": copy.deepcopy(self.reflection),
            "context_episodes": self.context_episodes,
            "context": copy.deepcopy(self.context),
            "mechanism_state": {
                "active_artifacts": len(self.store.active_entries()),
                "pending_hypotheses": len(self.store.pending),
                "superseded_artifacts": sum(
                    1 for entry in self.store.entries if entry.status == "invalidated"
                ),
            },
            "state_hash": self.state_hash(),
        }

    def state_hash(self) -> str:
        payload = {
            "condition": self.condition,
            "store": self.store.to_dict(),
            "reflection": self.reflection,
            "context_episodes": self.context_episodes,
            "context": self.context,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def make_agent(condition: str, seed: int = 0) -> BaseTradingAgent:
    if condition == "fixed_expert":
        return OracleFamilyAgent(mode="conservative", seed=seed)
    if condition == "random":
        return StaticTradingAgent(mode="random", seed=seed)
    return AdaptiveTradingAgent(condition=condition, seed=seed)
