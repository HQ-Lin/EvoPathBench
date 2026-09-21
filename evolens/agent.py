"""OpenAI-compatible trading agent with auditable, versioned self-evolution state."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .agents import BaseTradingAgent
from .features import feature_signature
from .models import EpisodeResult, Observation, OrderIntent, ScenarioSpec, StreamSpec
from .persistence import StrategyEntry, StrategyStore
from .skill_baselines import (
    apply_skillopt_edits,
    clean_skill_document,
    select_skillboost_candidate,
)
from .self_evolution_methods import (
    SKILL_CONDITIONS,
    empty_skillgrad_state,
    empty_skillx_library,
    render_skillgrad_package,
    render_skillx_library,
    sanitize_skillgrad_momentum,
    sanitize_skillgrad_package,
    sanitize_skillx_library,
    sanitize_trace2skill_document,
    sanitize_trace2skill_proposal,
)


DEFAULT_MODEL_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


@dataclass(frozen=True)
class ModelConfig:
    """Public invocation settings. The API key itself is deliberately absent."""

    model: str = "qwen3.6-plus"
    base_url: str = DEFAULT_MODEL_URL
    api_key_env: str = "EVOPATHBENCH_API_KEY"
    temperature: float = 0.1
    top_p: Optional[float] = None
    api_seed: Optional[int] = None
    decision_max_tokens: int = 700
    reflection_max_tokens: int = 1000
    timeout_seconds: float = 120.0
    max_retries: int = 3
    decision_interval: int = 1
    context_episodes: int = 5
    context_steps: int = 3
    skillopt_edit_budget: int = 2
    skill_document_max_chars: int = 6000
    skillboost_candidate_count: int = 4
    skillboost_max_case_regression: float = 0.25
    skillboost_max_slice_regression: float = 0.0
    skill_validation_variants: int = 1
    skillx_max_items_per_level: int = 6
    trace2skill_analyst_count: int = 2
    skillgrad_max_patterns: int = 12
    skillgrad_max_references: int = 6
    save_raw_responses: bool = False
    enable_thinking: bool = False
    thinking_budget: Optional[int] = None
    max_consecutive_failures: int = 5

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if not self.base_url.startswith("https://"):
            raise ValueError("base_url must use HTTPS")
        if not self.api_key_env:
            raise ValueError("api_key_env must be non-empty")
        if self.decision_interval < 1:
            raise ValueError("decision_interval must be positive")
        if self.context_episodes < 1:
            raise ValueError("context_episodes must be positive")
        if self.context_steps < 1:
            raise ValueError("context_steps must be positive")
        if self.skillopt_edit_budget < 1:
            raise ValueError("skillopt_edit_budget must be positive")
        if self.skill_document_max_chars < 256:
            raise ValueError("skill_document_max_chars must be at least 256")
        if self.skillboost_candidate_count < 2:
            raise ValueError("skillboost_candidate_count must be at least 2")
        if not 0.0 <= self.skillboost_max_case_regression <= 1.0:
            raise ValueError("skillboost_max_case_regression must be between 0 and 1")
        if self.skillboost_max_slice_regression < 0.0:
            raise ValueError("skillboost_max_slice_regression cannot be negative")
        if self.skill_validation_variants < 1:
            raise ValueError("skill_validation_variants must be positive")
        if self.skillx_max_items_per_level < 1:
            raise ValueError("skillx_max_items_per_level must be positive")
        if self.trace2skill_analyst_count < 2:
            raise ValueError("trace2skill_analyst_count must be at least 2")
        if self.skillgrad_max_patterns < 1:
            raise ValueError("skillgrad_max_patterns must be positive")
        if self.skillgrad_max_references < 1:
            raise ValueError("skillgrad_max_references must be positive")
        if self.decision_max_tokens < 1 or self.reflection_max_tokens < 1:
            raise ValueError("token limits must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.enable_thinking and (self.thinking_budget is None or self.thinking_budget < 1):
            raise ValueError("thinking_budget must be positive when thinking is enabled")
        if self.max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be positive")

    def public_dict(self) -> Dict[str, Any]:
        """Return serializable settings without reading or materializing a secret."""
        return asdict(self)


def require_api_key(config: ModelConfig) -> None:
    """Fail before an evaluation if its configured environment variable is absent."""
    if not os.getenv(config.api_key_env):
        raise RuntimeError(
            f"model API key is missing; set the {config.api_key_env} environment variable"
        )


class ModelAuthenticationError(RuntimeError):
    """Non-recoverable credential failure that should invalidate the run."""


class ModelRequestError(RuntimeError):
    """Non-recoverable request/configuration failure that should invalidate the run."""


class ModelTransientError(RuntimeError):
    """A retryable request that exhausted its bounded attempt budget."""

    def __init__(self, message: str, attempts: Sequence[Dict[str, Any]]) -> None:
        super().__init__(message)
        self.attempts = list(attempts)


class ModelCircuitOpenError(RuntimeError):
    """Too many consecutive remote failures to continue a scientifically valid run."""


def _redact(text: Any, api_key_env: str = "EVOPATHBENCH_API_KEY") -> str:
    value = str(text)
    secret = os.getenv(api_key_env)
    if secret:
        value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"(?i)(authorization\s*:\s*bearer\s+)[^\s,;]+", r"\1[REDACTED]", value)
    value = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+\-/=]+", r"\1[REDACTED]", value)
    return value


def _normalize_usage(value: Any) -> Dict[str, int]:
    usage = value if isinstance(value, dict) else {}

    def integer(*names: str) -> int:
        for name in names:
            candidate = usage.get(name)
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                return max(0, int(candidate))
        return 0

    input_tokens = integer("input_tokens", "prompt_tokens")
    output_tokens = integer("output_tokens", "completion_tokens")
    total_tokens = integer("total_tokens") or input_tokens + output_tokens
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        details = {}
    reasoning_value = details.get("reasoning_tokens", usage.get("reasoning_tokens", 0))
    reasoning_tokens = (
        max(0, int(reasoning_value))
        if isinstance(reasoning_value, (int, float)) and not isinstance(reasoning_value, bool)
        else 0
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract one JSON object from a fenced or prose-surrounded model response."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty model response")
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    decoder = json.JSONDecoder()
    # Try the complete response first. A valid JSON string can itself contain
    # Markdown fences (notably generated skill documents); selecting an inner
    # fence first would discard the outer JSON object.
    candidates = [text.strip()]
    if fenced:
        candidates.append(fenced.group(1).strip())
    for candidate in candidates:
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                value, _end = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("model response does not contain a JSON object")


def _call_model_with_client(
    client: Any,
    messages: List[Dict[str, str]],
    config: ModelConfig,
    max_tokens: int,
    phase: Optional[str] = None,
) -> Dict[str, Any]:
    """Call an OpenAI-compatible endpoint through an existing synchronous client.

    Only transient transport failures, HTTP 429, and selected 5xx responses are
    retried. A valid but malformed model response is handled by the agent and is
    never retried here.
    """
    del phase  # Kept in the callable contract for test doubles and tracing.
    import httpx

    require_api_key(config)
    api_key = os.environ[config.api_key_env]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if config.top_p is not None:
        payload["top_p"] = config.top_p
    if config.api_seed is not None:
        payload["seed"] = config.api_seed

    retryable_statuses = {408, 429, 500, 502, 503, 504}
    started = time.monotonic()
    attempts: List[Dict[str, Any]] = []
    last_error: Optional[Exception] = None
    for attempt in range(config.max_retries + 1):
        attempt_started = time.monotonic()
        try:
            response = client.post(config.base_url, json=payload, headers=headers)
            status = response.status_code
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "http_status": status,
                    "elapsed_seconds": round(time.monotonic() - attempt_started, 4),
                }
            )
            if status in {401, 403}:
                raise ModelAuthenticationError(
                    f"model-provider authentication failed with HTTP {status}; check {config.api_key_env}"
                )
            if 400 <= status < 500 and status not in retryable_statuses:
                raise ModelRequestError(
                    f"the model provider rejected the request with HTTP {status}; check model and request settings"
                )
            if status not in retryable_statuses:
                response.raise_for_status()
            if status in retryable_statuses:
                last_error = RuntimeError(f"model-provider transient HTTP {status}")
                if attempt >= config.max_retries:
                    break
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after is not None else 0.0
                except ValueError:
                    delay = 0.0
                delay = min(8.0, max(delay, 0.5 * (2**attempt)))
                time.sleep(delay + random.random() * 0.1)
                continue
            data = response.json()
            choices = data.get("choices") or []
            if not choices or not isinstance(choices[0], dict):
                raise ValueError("model-provider response has no choices[0]")
            message = choices[0].get("message") or {}
            content = message.get("content") or ""
            return {
                "content": content,
                "usage": data.get("usage", {}),
                "request_id": data.get("request_id") or data.get("id") or response.headers.get("x-request-id"),
                "response_model": data.get("model"),
                "finish_reason": choices[0].get("finish_reason"),
                "elapsed_seconds": round(time.monotonic() - started, 4),
                "attempts": attempts,
            }
        except (httpx.NetworkError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
            last_error = exc
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "error_type": type(exc).__name__,
                    "elapsed_seconds": round(time.monotonic() - attempt_started, 4),
                }
            )
            if attempt >= config.max_retries:
                break
            time.sleep(min(8.0, 0.5 * (2**attempt)) + random.random() * 0.1)
        except Exception:
            raise
    raise ModelTransientError(
        f"model-provider request failed after {len(attempts)} attempt(s): "
        f"{_redact(last_error or 'unknown transient error', config.api_key_env)}",
        attempts,
    )


def call_model(
    messages: List[Dict[str, str]],
    config: ModelConfig,
    max_tokens: int,
    phase: Optional[str] = None,
) -> Dict[str, Any]:
    """Make one model call with an isolated client.

    This remains the safe standalone default and the public seam used by tests.
    Full benchmark runs use :class:`ModelCompletionClient` to reuse TLS
    connections across concurrent rollout jobs.
    """
    import httpx

    timeout = httpx.Timeout(
        config.timeout_seconds,
        connect=min(10.0, config.timeout_seconds),
    )
    with httpx.Client(timeout=timeout) as client:
        return _call_model_with_client(
            client, messages, config, max_tokens, phase=phase
        )


class ModelCompletionClient:
    """Thread-safe completion callable backed by one bounded HTTP connection pool."""

    def __init__(self, config: ModelConfig, max_connections: int = 20) -> None:
        if not 1 <= max_connections <= 64:
            raise ValueError("max_connections must be between 1 and 64")
        import httpx

        require_api_key(config)
        timeout = httpx.Timeout(
            config.timeout_seconds,
            connect=min(10.0, config.timeout_seconds),
        )
        self._client = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
        )
        self._closed = False

    def __deepcopy__(self, memo: Dict[int, Any]) -> "ModelCompletionClient":
        # The transport is shared infrastructure, not agent state. Agent clones
        # must reuse it instead of trying to copy HTTPX locks and connections.
        del memo
        return self

    def __call__(
        self,
        messages: List[Dict[str, str]],
        config: ModelConfig,
        max_tokens: int,
        phase: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._closed:
            raise RuntimeError("model completion client is closed")
        return _call_model_with_client(
            self._client, messages, config, max_tokens, phase=phase
        )

    def close(self) -> None:
        if not self._closed:
            self._client.close()
            self._closed = True


CompletionFn = Callable[..., Dict[str, Any]]


def estimate_model_calls(
    scenarios: Sequence[ScenarioSpec],
    streams: Sequence[StreamSpec],
    conditions: Sequence[str],
    config: ModelConfig,
    campaigns: int,
    repeats: int,
    state_off_conditions: Sequence[str],
) -> Dict[str, Any]:
    """Statically count logical model calls before any paid request is sent."""
    scenario_by_id = {scenario.episode_id: scenario for scenario in scenarios}
    state_off = set(state_off_conditions)
    by_condition: Dict[str, Dict[str, int]] = {}
    total_decisions = 0
    total_reflections = 0
    total_skill_optimizers = 0
    total_validation_decisions = 0
    for condition in conditions:
        decisions = 0
        reflections = 0
        skill_optimizers = 0
        validation_decisions = 0
        probe_variants = 2 if condition in state_off else 1
        for stream in streams:
            validation_bank_decisions = 0
            for event in stream.events:
                scenario = scenario_by_id[event.episode_id]
                episode_decisions = math.ceil(scenario.horizon / config.decision_interval)
                decisions += episode_decisions
                if condition in {"skillopt", "skillboost"}:
                    validation_bank_decisions += (
                        config.skill_validation_variants * episode_decisions
                    )
                if (
                    condition not in ({"baseline", "context"} | SKILL_CONDITIONS)
                    and event.allow_state_update
                    and event.expose_feedback
                ):
                    reflections += 1
                if (
                    condition in SKILL_CONDITIONS
                    and event.allow_state_update
                    and event.expose_feedback
                ):
                    if condition == "skillopt":
                        candidate_count = 1
                        skill_optimizers += 1
                        validation_decisions += 2 * validation_bank_decisions
                    elif condition == "skillboost":
                        candidate_count = config.skillboost_candidate_count
                        skill_optimizers += 1 + candidate_count
                        validation_decisions += (1 + candidate_count) * validation_bank_decisions
                    elif condition == "skillx":
                        skill_optimizers += 2
                    elif condition == "trace2skill":
                        skill_optimizers += config.trace2skill_analyst_count + 1
                    elif condition == "skillgrad":
                        skill_optimizers += 3
            for checkpoint in stream.checkpoints:
                for probe_id in checkpoint.probe_episode_ids:
                    scenario = scenario_by_id[probe_id]
                    decisions += (
                        probe_variants
                        * repeats
                        * math.ceil(scenario.horizon / config.decision_interval)
                    )
        decisions *= campaigns
        reflections *= campaigns
        skill_optimizers *= campaigns
        validation_decisions *= campaigns
        decisions += validation_decisions
        by_condition[condition] = {
            "decision_calls": decisions,
            "candidate_validation_decision_calls": validation_decisions,
            "reflection_calls": reflections,
            "skill_optimizer_calls": skill_optimizers,
            "logical_calls": decisions + reflections + skill_optimizers,
        }
        total_decisions += decisions
        total_reflections += reflections
        total_skill_optimizers += skill_optimizers
        total_validation_decisions += validation_decisions
    logical_calls = total_decisions + total_reflections + total_skill_optimizers
    completion_token_cap_sum = (
        total_decisions * config.decision_max_tokens
        + (total_reflections + total_skill_optimizers) * config.reflection_max_tokens
    )
    thinking_token_cap_sum = logical_calls * int(config.thinking_budget or 0)
    output_token_upper_bound = completion_token_cap_sum + thinking_token_cap_sum
    return {
        "by_condition": by_condition,
        "decision_calls": total_decisions,
        "candidate_validation_decision_calls": total_validation_decisions,
        "reflection_calls": total_reflections,
        "skill_optimizer_calls": total_skill_optimizers,
        "logical_calls": logical_calls,
        "max_http_attempts": logical_calls * (config.max_retries + 1),
        "output_token_upper_bound": output_token_upper_bound,
        "configured_completion_token_cap_sum": completion_token_cap_sum,
        "configured_thinking_token_cap_sum": thinking_token_cap_sum,
        "decision_interval": config.decision_interval,
    }


class ModelTradingAgent(BaseTradingAgent):
    """Remote decision agent whose experience artifacts persist across a task stream."""

    CONDITIONS = {
        "baseline",
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
    supports_state_ablation = True

    def __init__(
        self,
        condition: str,
        seed: int = 0,
        config: Optional[ModelConfig] = None,
        max_entries: int = 12,
        completion_fn: CompletionFn = call_model,
    ) -> None:
        if condition not in self.CONDITIONS:
            raise ValueError(f"unsupported model-agent condition: {condition}")
        self.condition = condition
        self.supports_state_ablation = condition != "baseline"
        self.seed = seed
        self.config = config or ModelConfig()
        self.store = StrategyStore(max_entries=max_entries)
        self.reflection: List[Dict[str, Any]] = []
        self.context: List[Dict[str, Any]] = []
        self.skill_document = ""
        self.skill_version = 0
        self.skill_best_score: Optional[float] = None
        self.skill_update_history: List[Dict[str, Any]] = []
        self.skill_method_state: Dict[str, Any] = {}
        if condition == "skillx":
            self.skill_method_state = {"library": empty_skillx_library()}
        elif condition == "trace2skill":
            self.skill_method_state = {"proposal_history": []}
        elif condition == "skillgrad":
            self.skill_method_state = empty_skillgrad_state()
        self.current_mode = "openai_compatible"
        self.read_only = False
        self.state_access_enabled = True
        self._completion_fn = completion_fn
        self._trace: List[Dict[str, Any]] = []
        self._decision_summaries: List[str] = []
        self._current_episode_steps: List[Dict[str, Any]] = []
        self._visible_rules: List[StrategyEntry] = []
        self._used_rule_ids: List[str] = []
        self._retrieved = False
        self._execution_seed = 0
        self._consecutive_failures = 0
        self._candidate_validator: Optional[Callable[[Sequence[Tuple[str, str]]], Dict[str, Any]]] = None
        self._validation_audit_context: Optional[Dict[str, Any]] = None

    def clone(self, read_only: bool = True) -> "ModelTradingAgent":
        clone = copy.deepcopy(self)
        clone._completion_fn = self._completion_fn
        clone._candidate_validator = None
        clone._validation_audit_context = None
        clone.read_only = read_only
        clone._trace = []
        clone._decision_summaries = []
        clone._current_episode_steps = []
        clone._visible_rules = []
        clone._used_rule_ids = []
        clone._retrieved = False
        return clone

    def begin_episode(self, scenario: ScenarioSpec, execution_seed: int) -> None:
        # The ScenarioSpec contains benchmark-only labels. They are intentionally
        # not retained or serialized into a model prompt.
        del scenario
        self._execution_seed = int(execution_seed)
        self.current_mode = "openai_compatible"
        self._trace = []
        self._decision_summaries = []
        self._current_episode_steps = []
        self._visible_rules = []
        self._used_rule_ids = []
        self._retrieved = False

    def set_state_access(self, enabled: bool) -> None:
        self.state_access_enabled = bool(enabled)

    @property
    def supports_candidate_validation(self) -> bool:
        return self.condition in {"skillopt", "skillboost"}

    @property
    def candidate_validation_variants(self) -> int:
        return self.config.skill_validation_variants

    def set_candidate_validator(
        self,
        validator: Optional[Callable[[Sequence[Tuple[str, str]]], Dict[str, Any]]],
    ) -> None:
        self._candidate_validator = validator

    def set_skill_document(self, document: str) -> None:
        self.skill_document = clean_skill_document(document, self.config.skill_document_max_chars)

    def set_validation_audit_context(self, context: Optional[Dict[str, Any]]) -> None:
        self._validation_audit_context = copy.deepcopy(context) if context else None

    def _retrieve_if_ready(self, observation: Observation) -> None:
        if self._retrieved:
            return
        if self.condition in SKILL_CONDITIONS:
            available = bool(self.skill_document and self.state_access_enabled)
            self._trace.append(
                {
                    "event": "retrieve",
                    "artifact": "skill_document",
                    "hit": available,
                    "disabled": not self.state_access_enabled,
                    "match_type": "global_skill" if available else "miss",
                    "rule_id": f"skill:v{self.skill_version}" if available else None,
                    "skill_version": self.skill_version,
                    "mechanism": "skill_invocation",
                    "mechanisms": ["skill_invocation"] if available else [],
                }
            )
            if available:
                self._trace.append(
                    {
                        "event": "apply",
                        "artifact": "skill_document",
                        "skill_version": self.skill_version,
                        "mechanism": "skill_invocation",
                    }
                )
            self._retrieved = True
            return
        if self.condition == "context":
            available = self.context if self.state_access_enabled else []
            self._trace.append(
                {
                    "event": "retrieve",
                    "artifact": "context",
                    "hit": bool(available),
                    "disabled": not self.state_access_enabled,
                    "match_type": "window" if available else "miss",
                    "rule_id": None,
                    "window_episode_count": len(available),
                    "mechanism": "context_injection",
                    "mechanisms": ["context_injection"] if available else [],
                }
            )
            self._retrieved = True
            return
        if len(observation.prices) < 4:
            return
        signature = feature_signature(
            observation.prices,
            observation.fundamentals,
            observation.spread_bps,
            observation.liquidity,
        )
        match_type = "miss"
        if not self.state_access_enabled:
            self._trace.append(
                {
                    "event": "retrieve",
                    "signature": signature,
                    "hit": False,
                    "disabled": True,
                    "match_type": "miss",
                    "rule_id": None,
                    "mechanism": "memory_invocation",
                }
            )
            self._retrieved = True
            return
        if self.condition in {"episodic_memory", "consolidated_memory"}:
            entry, match_type = self.store.retrieve_with_match(signature)
            if entry is not None:
                self._visible_rules = [entry]
        elif self.condition == "reflection":
            candidates = [item for item in self.reflection if item.get("signature") == signature]
            if candidates:
                selected = max(candidates, key=lambda item: (item.get("score", 0.0), item.get("episode_id", "")))
                self._visible_rules = [
                    StrategyEntry(
                        rule_id=(
                            "raw:"
                            + hashlib.sha256(str(selected["episode_id"]).encode("utf-8")).hexdigest()[:12]
                        ),
                        signature=signature,
                        mode=str(selected.get("policy_label", "raw_experience")),
                        evidence_episode_ids=[str(selected["episode_id"])],
                        evidence_scores=[float(selected.get("score", 0.0))],
                        confidence=0.0,
                        hypothesis=str(selected.get("hypothesis", "")),
                    )
                ]
                match_type = "exact"
        entry = self._visible_rules[0] if self._visible_rules else None
        self._trace.append(
            {
                "event": "retrieve",
                "signature": signature,
                "hit": entry is not None,
                "disabled": False,
                "match_type": match_type,
                "rule_id": entry.rule_id if entry else None,
                "mechanism": "memory_invocation",
                "mechanisms": ["memory_invocation"]
                + (["scope_qualification"] if match_type == "scope_transfer" else []),
            }
        )
        if entry is not None:
            self._trace.append(
                {
                    "event": "apply",
                    "rule_id": entry.rule_id,
                    "match_type": match_type,
                    "mechanism": "memory_invocation",
                }
            )
        self._retrieved = True

    @staticmethod
    def _memory_payload(entries: Sequence[StrategyEntry]) -> List[Dict[str, Any]]:
        return [
            {
                "rule_id": entry.rule_id,
                "policy_label": entry.mode,
                "hypothesis": entry.hypothesis,
                "scope": entry.scope,
                "confidence": round(entry.confidence, 6),
                "application_count": entry.application_count,
            }
            for entry in entries
        ]

    def _context_payload(self) -> List[Dict[str, Any]]:
        if self.condition != "context" or not self.state_access_enabled:
            return []
        return copy.deepcopy(self.context)

    def _skill_payload(self) -> str:
        if self.condition not in SKILL_CONDITIONS or not self.state_access_enabled:
            return ""
        return self.skill_document

    @staticmethod
    def _history_observation_payload(observation: Observation) -> Dict[str, Any]:
        """Keep direct observable state without duplicating cumulative price arrays."""
        return {
            "step": observation.step,
            "price": observation.price,
            "public_fundamental": observation.fundamental,
            "public_signal": observation.public_signal,
            "spread_bps": observation.spread_bps,
            "liquidity": observation.liquidity,
            "market_features": observation.market_features,
            "account": {
                "cash": observation.cash,
                "position": observation.position,
                "marked_wealth": observation.marked_wealth,
                "max_drawdown_so_far": observation.max_drawdown_so_far,
                "turnover_so_far": observation.turnover_so_far,
            },
            "risk_contract": observation.risk_contract.to_dict(),
        }

    @staticmethod
    def _observation_payload(observation: Observation) -> Dict[str, Any]:
        return {
            "step": observation.step,
            "horizon": observation.horizon,
            "market_layer": observation.layer,
            "asset_symbol": observation.asset_symbol,
            "price": observation.price,
            "public_fundamental": observation.fundamental,
            "public_signal": observation.public_signal,
            "spread_bps": observation.spread_bps,
            "liquidity": observation.liquidity,
            "market_features": observation.market_features,
            "observed_prices": observation.prices,
            "observed_fundamentals": observation.fundamentals,
            "account": {
                "cash": observation.cash,
                "position": observation.position,
                "marked_wealth": observation.marked_wealth,
                "initial_wealth": observation.initial_wealth,
                "max_drawdown_so_far": observation.max_drawdown_so_far,
                "turnover_so_far": observation.turnover_so_far,
            },
            "risk_contract": observation.risk_contract.to_dict(),
        }

    def _request_json(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        phase: str,
        request_nonce: str = "",
        audit_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        logical_call_id = uuid.uuid4().hex
        prompt_json = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        started = time.monotonic()
        call_config = self.config
        resolved_api_seed = self.config.api_seed
        if resolved_api_seed is not None:
            seed_material = (
                f"{resolved_api_seed}|{self._execution_seed}|{phase}|{request_nonce}"
            )
            resolved_api_seed = int(
                hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8], 16
            )
            call_config = replace(self.config, api_seed=resolved_api_seed)
        event: Dict[str, Any] = {
            "event": "model_call",
            "logical_call_id": logical_call_id,
            "phase": phase,
            "provider": "openai_compatible",
            "requested_model": self.config.model,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "api_seed": resolved_api_seed,
            "enable_thinking": self.config.enable_thinking,
            "max_tokens": max_tokens,
            "prompt_hash": hashlib.sha256(prompt_json.encode("utf-8")).hexdigest(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        if audit_context:
            event["audit_context"] = copy.deepcopy(audit_context)
        try:
            response = self._completion_fn(messages, call_config, max_tokens, phase=phase)
            content = response.get("content") or ""
            event.update(
                {
                    "transport_ok": True,
                    "request_id": response.get("request_id"),
                    "response_model": response.get("response_model"),
                    "finish_reason": response.get("finish_reason"),
                    "elapsed_seconds": float(response.get("elapsed_seconds", time.monotonic() - started)),
                    "attempts": response.get("attempts", []),
                    "usage": _normalize_usage(response.get("usage")),
                    "response_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                }
            )
            try:
                parsed = _extract_json(content)
            except Exception as exc:
                event.update(
                    {
                        "ok": False,
                        "parse_ok": False,
                        "error_type": type(exc).__name__,
                        "error": _redact(exc, self.config.api_key_env),
                    }
                )
                if self.config.save_raw_responses:
                    event["response"] = _redact(content, self.config.api_key_env)
                self._trace.append(event)
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.config.max_consecutive_failures:
                    raise ModelCircuitOpenError(
                        f"model-provider circuit opened after {self._consecutive_failures} consecutive failures"
                    )
                return None
            event.update({"ok": True, "parse_ok": True, "parsed": parsed, "error": None})
            self._consecutive_failures = 0
            if self.config.save_raw_responses:
                event["response"] = _redact(content, self.config.api_key_env)
            self._trace.append(event)
            return parsed
        except (ModelAuthenticationError, ModelRequestError, ModelCircuitOpenError):
            raise
        except Exception as exc:
            event.update(
                {
                    "ok": False,
                    "elapsed_seconds": round(time.monotonic() - started, 4),
                    "usage": _normalize_usage({}),
                    "attempts": list(getattr(exc, "attempts", [])),
                    "transport_ok": False,
                    "parse_ok": False,
                    "error_type": type(exc).__name__,
                    "error": _redact(exc, self.config.api_key_env),
                }
            )
            self._trace.append(event)
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.config.max_consecutive_failures:
                raise ModelCircuitOpenError(
                    f"model-provider circuit opened after {self._consecutive_failures} consecutive failures"
                )
            return None

    @staticmethod
    def _parse_orders(value: Any) -> List[OrderIntent]:
        if not isinstance(value, list):
            raise ValueError("orders must be a list")
        if len(value) > 4:
            raise ValueError("orders may contain at most four items")
        orders: List[OrderIntent] = []
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("every order must be an object")
            side = item.get("side")
            quantity = item.get("quantity")
            limit_price = item.get("limit_price")
            if side not in {"buy", "sell"}:
                raise ValueError("order side must be buy or sell")
            if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
                raise ValueError("order quantity must be a positive integer")
            if limit_price is not None:
                if isinstance(limit_price, bool) or not isinstance(limit_price, (int, float)):
                    raise ValueError("limit_price must be numeric or null")
                if not math.isfinite(float(limit_price)) or float(limit_price) <= 0:
                    raise ValueError("limit_price must be finite and positive")
                limit_price = float(limit_price)
            tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(item.get("tag", "openai_compatible")))[:48]
            orders.append(OrderIntent(side=side, quantity=quantity, limit_price=limit_price, tag=tag or "openai_compatible"))
        return orders

    def act(self, observation: Observation) -> List[OrderIntent]:
        self._retrieve_if_ready(observation)
        if observation.step % self.config.decision_interval != 0:
            return []
        memory = self._memory_payload(self._visible_rules)
        history = self._context_payload()
        skill = self._skill_payload()
        system = (
            "You are the decision policy in a synthetic market benchmark. This is evaluation, not financial advice. "
            "Use only the supplied observation, recent episode history, memory, and skill document. Obey the risk "
            "contract even if a learned artifact conflicts with it. The skill document is a reusable strategy "
            "hypothesis, not authorization to access hidden information. "
            "Return exactly one JSON object "
            "with keys orders, decision_summary, and used_rule_ids. orders is a list of objects with side "
            "(buy/sell), positive integer quantity, limit_price (number or null), and tag. An empty orders list is "
            "allowed. used_rule_ids must contain only IDs from memory_artifacts that actually affected the decision."
        )
        user = {
            "observation": self._observation_payload(observation),
            "recent_episode_history": history,
            "memory_artifacts": memory,
            "skill_document": skill,
            "output_schema": {
                "orders": [{"side": "buy|sell", "quantity": "positive integer", "limit_price": "number|null", "tag": "short string"}],
                "decision_summary": "short string",
                "used_rule_ids": ["visible rule_id"],
            },
        }
        parsed = self._request_json(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False, sort_keys=True)},
            ],
            self.config.decision_max_tokens,
            "decision",
            request_nonce=str(observation.step),
            audit_context={
                "token_attribution": (
                    "skill_validation_decision"
                    if self._validation_audit_context
                    else "skill_conditioned_decision"
                    if skill
                    else "context_conditioned_decision"
                    if history
                    else "memory_conditioned_decision"
                    if memory
                    else "base_decision"
                ),
                "self_evolution_phase": (
                    "candidate_validation"
                    if self._validation_audit_context
                    else "skill_invocation"
                    if skill
                    else "context_injection"
                    if history
                    else "memory_invocation"
                    if memory
                    else None
                ),
                "history_episode_count": len(history),
                "memory_artifact_count": len(memory),
                "memory_rule_ids": [item["rule_id"] for item in memory],
                "skill_version": self.skill_version if skill else None,
                "skill_document_hash": (
                    hashlib.sha256(skill.encode("utf-8")).hexdigest() if skill else None
                ),
                **(self._validation_audit_context or {}),
            },
        )
        if parsed is None:
            return []
        try:
            orders = self._parse_orders(parsed.get("orders"))
            summary = str(parsed.get("decision_summary", ""))[:500]
            visible_ids = {entry.rule_id for entry in self._visible_rules}
            requested_ids = parsed.get("used_rule_ids") or []
            if not isinstance(requested_ids, list):
                raise ValueError("used_rule_ids must be a list")
            used = [str(item) for item in requested_ids if str(item) in visible_ids]
            self._used_rule_ids.extend(item for item in used if item not in self._used_rule_ids)
            if summary:
                self._decision_summaries.append(summary)
            if self.condition == "context" or self.condition in SKILL_CONDITIONS:
                self._current_episode_steps.append(
                    {
                        "observation": self._history_observation_payload(observation),
                        "orders": [
                            {
                                "side": order.side,
                                "quantity": order.quantity,
                                "limit_price": order.limit_price,
                                "tag": order.tag,
                            }
                            for order in orders
                        ],
                        "decision_summary": summary,
                    }
                )
                self._current_episode_steps = self._current_episode_steps[
                    -self.config.context_steps :
                ]
            self._trace.append(
                {
                    "event": "action",
                    "step": observation.step,
                    "order_count": len(orders),
                    "used_rule_ids": used,
                    "mechanism": "decision_execution",
                }
            )
            return orders
        except Exception as exc:
            self._trace.append(
                {
                    "event": "parse_error",
                    "phase": "decision",
                    "error": _redact(exc, self.config.api_key_env),
                }
            )
            return []

    @staticmethod
    def _sanitize_policy_label(value: Any) -> str:
        allowed = {
            "momentum",
            "mean_reversion",
            "fundamental",
            "liquidity_aware",
            "risk_reducing",
            "event_defensive",
            "no_trade",
        }
        if not isinstance(value, str):
            raise ValueError("policy_label must be a string")
        label = re.sub(r"[^a-z0-9_.-]+", "_", value.strip().lower())[:48]
        label = label.strip("_.-")
        if label not in allowed:
            raise ValueError("policy_label is not in the registered strategy taxonomy")
        return label

    @staticmethod
    def _clean_text(value: Any, limit: int) -> str:
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", str(value))
        return " ".join(text.split())[:limit]

    def _reflection_messages(self, result: EpisodeResult) -> List[Dict[str, str]]:
        system = (
            "You are the self-evolution module of a synthetic-market decision agent. Diagnose the observable "
            "outcome and propose one auditable memory operation. Return exactly one JSON object. action is upsert, "
            "invalidate, or none. policy_label is a short reusable label; hypothesis states observable conditions "
            "and behavior; invalidate_rule_ids may contain only visible memory rule IDs. Do not infer hidden task labels."
        )
        user = {
            "outcome": {
                "score": result.score,
                "return_pct": result.return_pct,
                "max_drawdown": result.max_drawdown,
                "turnover": result.turnover,
                "violation_count": len(result.violations),
                "fee_paid": result.fee_paid,
                "observable_feature_signature": result.feature_signature,
            },
            "recent_decision_summaries": self._decision_summaries[-8:],
            "memory_artifacts": self._memory_payload(self._visible_rules),
            "used_rule_ids": list(self._used_rule_ids),
            "output_schema": {
                "attribution": "short evidence-based diagnosis",
                "action": "upsert|invalidate|none",
                "policy_label": (
                    "momentum|mean_reversion|fundamental|liquidity_aware|risk_reducing|"
                    "event_defensive|no_trade"
                ),
                "hypothesis": "observable condition and policy",
                "invalidate_rule_ids": ["visible rule_id"],
            },
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False, sort_keys=True)},
        ]

    def _skill_learning_packet(self, result: EpisodeResult) -> Dict[str, Any]:
        """Observable-only evidence supplied to the skill optimizer."""
        return {
            "outcome": {
                "score": result.score,
                "return_pct": result.return_pct,
                "max_drawdown": result.max_drawdown,
                "turnover": result.turnover,
                "violation_count": len(result.violations),
                "fee_paid": result.fee_paid,
                "observable_feature_signature": result.feature_signature,
            },
            "trajectory": copy.deepcopy(self._current_episode_steps),
            "recent_decision_summaries": self._decision_summaries[-8:],
        }

    def _validate_skill_candidates(
        self,
        candidates: Sequence[Tuple[str, str]],
    ) -> Optional[Dict[str, Any]]:
        if self._candidate_validator is None:
            self._trace.append(
                {
                    "event": "write",
                    "artifact": "skill_document",
                    "committed": False,
                    "reason": "candidate_validator_unavailable",
                    "mechanism": "verified_skill_acceptance",
                }
            )
            return None
        result = self._candidate_validator(candidates)
        if not isinstance(result, dict) or not isinstance(result.get("reports"), dict):
            self._trace.append(
                {
                    "event": "parse_error",
                    "phase": "skill_validation",
                    "error": "candidate validator returned an invalid report",
                }
            )
            return None
        for event in result.get("agent_trace", []):
            if isinstance(event, dict):
                self._trace.append(copy.deepcopy(event))
        return result

    def _record_skill_update(self, record: Dict[str, Any]) -> None:
        self.skill_update_history.append(copy.deepcopy(record))
        self.skill_update_history = self.skill_update_history[-50:]
        self._trace.append({"event": "write", "artifact": "skill_document", **record})

    def _run_skillopt_update(self, result: EpisodeResult) -> None:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the SkillOpt text optimizer for a synthetic-market agent. Diagnose the observable "
                    "episode and propose at most the stated edit budget against one reusable skill document. "
                    "Allowed operations are append, insert_after, replace, and delete. For non-append operations, "
                    "target must be an exact unique substring of the current document. Return exactly one JSON object "
                    "with diagnosis and edits. Do not use hidden task labels or episode identifiers."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "current_skill_document": self.skill_document,
                        "learning_evidence": self._skill_learning_packet(result),
                        "edit_budget": self.config.skillopt_edit_budget,
                        "output_schema": {
                            "diagnosis": "short evidence-based diagnosis",
                            "edits": [
                                {
                                    "op": "append|insert_after|replace|delete",
                                    "target": "exact substring; empty only for append",
                                    "content": "new reusable skill text; empty only for delete",
                                }
                            ],
                        },
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]
        parsed = self._request_json(
            messages,
            self.config.reflection_max_tokens,
            "skillopt_optimize",
            request_nonce=f"skill-v{self.skill_version + 1}",
            audit_context={
                "token_attribution": "self_evolution_skill_optimizer",
                "self_evolution_phase": "skillopt_bounded_edit",
                "condition": self.condition,
                "incumbent_skill_version": self.skill_version,
            },
        )
        if parsed is None:
            return
        candidate, applied, rejected = apply_skillopt_edits(
            self.skill_document,
            parsed.get("edits"),
            edit_budget=self.config.skillopt_edit_budget,
            max_chars=self.config.skill_document_max_chars,
        )
        if not applied or candidate == self.skill_document:
            self._record_skill_update(
                {
                    "method": "skillopt",
                    "committed": False,
                    "reason": "no_valid_edit",
                    "applied_edits": applied,
                    "rejected_edits": rejected,
                    "mechanism": "bounded_skill_edit",
                }
            )
            return
        validation = self._validate_skill_candidates(
            [("incumbent", self.skill_document), ("candidate_0", candidate)]
        )
        if validation is None:
            return
        reports = validation["reports"]
        incumbent = reports.get("incumbent", {})
        proposal = reports.get("candidate_0", {})
        incumbent_score = float(incumbent.get("mean_score", 0.0) or 0.0)
        proposal_score = float(proposal.get("mean_score", 0.0) or 0.0)
        accepted = proposal_score > incumbent_score
        if accepted:
            self.skill_document = candidate
            self.skill_version += 1
            self.skill_best_score = proposal_score
        self._record_skill_update(
            {
                "method": "skillopt",
                "committed": accepted,
                "reason": "strict_validation_improvement" if accepted else "validation_gate_rejected",
                "skill_version": self.skill_version,
                "incumbent_score": incumbent_score,
                "candidate_score": proposal_score,
                "validation_case_count": len(proposal.get("cases", [])),
                "applied_edits": applied,
                "rejected_edits": rejected,
                "mechanism": "verified_skill_acceptance",
            }
        )

    def _run_skillboost_update(self, result: EpisodeResult) -> None:
        packet = self._skill_learning_packet(result)
        diagnosis = self._request_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You are the SkillBoost diagnosis stage for a synthetic-market agent. Reconstruct the failure "
                        "from observable evidence, identify the earliest causal deviation, cluster plausible root "
                        "causes, and name behaviors that a repair must preserve. Return exactly one JSON object."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "current_skill_document": self.skill_document,
                            "learning_evidence": packet,
                            "output_schema": {
                                "earliest_causal_deviation": "short string",
                                "root_causes": ["observable root cause"],
                                "protected_behaviors": ["behavior that must not regress"],
                            },
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            self.config.reflection_max_tokens,
            "skillboost_diagnose",
            request_nonce=f"skill-v{self.skill_version + 1}",
            audit_context={
                "token_attribution": "self_evolution_skill_optimizer",
                "self_evolution_phase": "skillboost_shared_diagnosis",
                "condition": self.condition,
                "incumbent_skill_version": self.skill_version,
            },
        )
        if diagnosis is None:
            return
        strategies = [
            "conservative_repair",
            "failure_focused_repair",
            "risk_aware_repair",
            "transfer_oriented_repair",
        ]
        generated: List[Tuple[str, str]] = []
        seen = {self.skill_document}
        for index in range(self.config.skillboost_candidate_count):
            strategy = strategies[index] if index < len(strategies) else f"diverse_repair_{index + 1}"
            parsed = self._request_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are the SkillBoost exploration stage. Using the fixed diagnosis, produce one complete "
                            "replacement skill document under the assigned repair prior. Preserve valid existing "
                            "guidance, make rules observable and executable, and obey the risk contract. Return exactly "
                            "one JSON object with skill_document and rationale. Keep skill_document concise, use no "
                            f"code fences, and stay below {self.config.skill_document_max_chars} characters."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "repair_prior": strategy,
                                "fixed_diagnosis": diagnosis,
                                "current_skill_document": self.skill_document,
                                "learning_evidence": packet,
                                "output_schema": {
                                    "skill_document": "complete reusable Markdown skill",
                                    "rationale": "short explanation",
                                },
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    },
                ],
                self.config.reflection_max_tokens,
                "skillboost_generate",
                request_nonce=f"candidate-{index}-{strategy}",
                audit_context={
                    "token_attribution": "self_evolution_skill_optimizer",
                    "self_evolution_phase": "skillboost_candidate_generation",
                    "condition": self.condition,
                    "candidate_id": f"candidate_{index}",
                    "repair_prior": strategy,
                },
            )
            if parsed is None:
                continue
            document = clean_skill_document(
                parsed.get("skill_document", ""), self.config.skill_document_max_chars
            )
            if document and document not in seen:
                candidate_id = f"candidate_{index}"
                generated.append((candidate_id, document))
                seen.add(document)
        if not generated:
            self._record_skill_update(
                {
                    "method": "skillboost",
                    "committed": False,
                    "reason": "no_distinct_candidate",
                    "mechanism": "prior_guided_skill_exploration",
                }
            )
            return
        validation = self._validate_skill_candidates(
            [("incumbent", self.skill_document), *generated]
        )
        if validation is None:
            return
        reports = validation["reports"]
        winner, assessments = select_skillboost_candidate(
            reports.get("incumbent", {}),
            [(candidate_id, reports.get(candidate_id, {})) for candidate_id, _ in generated],
            max_case_regression=self.config.skillboost_max_case_regression,
            max_slice_regression=self.config.skillboost_max_slice_regression,
        )
        accepted = winner is not None
        if accepted:
            self.skill_document = dict(generated)[winner]
            self.skill_version += 1
            self.skill_best_score = float(reports[winner].get("mean_score", 0.0) or 0.0)
        self._record_skill_update(
            {
                "method": "skillboost",
                "committed": accepted,
                "reason": "verified_candidate_selected" if accepted else "all_candidates_rejected",
                "skill_version": self.skill_version,
                "winner": winner,
                "candidate_count": len(generated),
                "validation_case_count": len(reports.get("incumbent", {}).get("cases", [])),
                "max_case_regression": self.config.skillboost_max_case_regression,
                "max_slice_regression": self.config.skillboost_max_slice_regression,
                "assessments": assessments,
                "mechanism": "verified_skill_acceptance",
            }
        )

    def _skill_optimizer_context(self, phase: str, **extra: Any) -> Dict[str, Any]:
        return {
            "token_attribution": "self_evolution_skill_optimizer",
            "self_evolution_phase": phase,
            "condition": self.condition,
            "incumbent_skill_version": self.skill_version,
            **extra,
        }

    def _run_skillx_update(self, result: EpisodeResult) -> None:
        """Adapt SkillX's extraction and three-level library consolidation loop."""
        packet = self._skill_learning_packet(result)
        extraction = self._request_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You are SkillX's trajectory-to-skill extraction stage for a synthetic-market agent. "
                        "Extract reusable skills at exactly three abstraction levels: planning (high-level intent), "
                        "functional (multi-step reusable procedure), and atomic (single executable operation). "
                        "For each proposal choose add, modify, or keep relative to the incumbent library. Ground every "
                        "proposal only in observable trajectory evidence. Return exactly one JSON object and never use "
                        "hidden task labels or identifiers."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "incumbent_library": self.skill_method_state.get(
                                "library", empty_skillx_library()
                            ),
                            "learning_evidence": packet,
                            "output_schema": {
                                "proposals": {
                                    "planning": [
                                        {
                                            "operation": "add|modify|keep",
                                            "target_name": "existing name or empty",
                                            "name": "skill name",
                                            "content": "reusable guidance",
                                            "activation_signals": ["observable signal"],
                                            "tools": ["relevant interface"],
                                            "evidence": "observable support",
                                        }
                                    ],
                                    "functional": "same item schema",
                                    "atomic": "same item schema",
                                }
                            },
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            self.config.reflection_max_tokens,
            "skillx_extract",
            request_nonce=f"skill-v{self.skill_version + 1}-extract",
            audit_context=self._skill_optimizer_context("skillx_hierarchical_extraction"),
        )
        if extraction is None:
            return
        consolidation = self._request_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You are SkillX's library merge and filter stage. Apply valid add/modify/keep proposals, merge "
                        "semantic duplicates, remove brittle episode-specific advice, and return the complete updated "
                        "three-level library. Preserve useful incumbent skills unless evidence justifies revision. Each "
                        "level must be a JSON list. Return exactly one JSON object."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "frozen_incumbent_library": self.skill_method_state.get(
                                "library", empty_skillx_library()
                            ),
                            "extracted_proposals": extraction.get("proposals", {}),
                            "capacity_per_level": self.config.skillx_max_items_per_level,
                            "output_schema": {
                                "library": {
                                    "planning": [
                                        {
                                            "name": "unique skill name",
                                            "content": "reusable guidance",
                                            "activation_signals": ["observable signal"],
                                            "tools": ["relevant interface"],
                                            "source_count": "positive integer",
                                        }
                                    ],
                                    "functional": "same item schema",
                                    "atomic": "same item schema",
                                },
                                "merge_log": ["short auditable decision"],
                                "filtered_items": ["short reason"],
                            },
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            self.config.reflection_max_tokens,
            "skillx_consolidate",
            request_nonce=f"skill-v{self.skill_version + 1}-consolidate",
            audit_context=self._skill_optimizer_context("skillx_merge_filter"),
        )
        if consolidation is None:
            return
        library, rejected = sanitize_skillx_library(
            consolidation.get("library"),
            max_items_per_level=self.config.skillx_max_items_per_level,
            max_content_chars=self.config.skill_document_max_chars,
        )
        item_count = sum(len(library[level]) for level in ("planning", "functional", "atomic"))
        incumbent = self.skill_method_state.get("library", empty_skillx_library())
        document = render_skillx_library(library, self.config.skill_document_max_chars)
        fatal_rejections = {"library_not_object", "level_not_list"}
        schema_complete = not any(item.get("reason") in fatal_rejections for item in rejected)
        accepted = schema_complete and item_count > 0 and library != incumbent and bool(document)
        if accepted:
            self.skill_method_state["library"] = library
            self.skill_document = document
            self.skill_version += 1
        self._record_skill_update(
            {
                "method": "skillx",
                "committed": accepted,
                "reason": "hierarchical_library_updated" if accepted else "no_valid_library_change",
                "skill_version": self.skill_version,
                "state_artifact": "hierarchical_skill_library",
                "level_counts": {level: len(library[level]) for level in library},
                "schema_rejections": rejected,
                "merge_log": consolidation.get("merge_log", []),
                "filtered_items": consolidation.get("filtered_items", []),
                "mechanism": "hierarchical_skill_consolidation",
            }
        )

    def _run_trace2skill_update(self, result: EpisodeResult) -> None:
        """Adapt Trace2Skill's frozen-snapshot map/reduce patch evolution."""
        packet = self._skill_learning_packet(result)
        incumbent = self.skill_document
        incumbent_hash = hashlib.sha256(incumbent.encode("utf-8")).hexdigest()
        proposals: List[Dict[str, Any]] = []
        proposal_rejections: List[Dict[str, Any]] = []
        for analyst_index in range(self.config.trace2skill_analyst_count):
            proposal = self._request_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are one independent Trace2Skill map-stage analyst. Diagnose the observable trace "
                            "against the frozen incumbent skill and propose a concise local patch. Do not assume any "
                            "other analyst's output. Use operations add_section, insert_after, insert_before, "
                            "replace_in_section, append_to_section, or delete_section. Return exactly one JSON object."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "analyst_index": analyst_index,
                                "frozen_incumbent_skill": incumbent,
                                "learning_evidence": packet,
                                "output_schema": {
                                    "analysis_label": "success|failure|mixed",
                                    "evidence": ["observable evidence"],
                                    "protected_behaviors": ["behavior to retain"],
                                    "patch": [
                                        {
                                            "operation": (
                                                "add_section|insert_after|insert_before|replace_in_section|"
                                                "append_to_section|delete_section"
                                            ),
                                            "target": "section or exact anchor",
                                            "content": "concise change",
                                            "rationale": "why this generalizes",
                                        }
                                    ],
                                },
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    },
                ],
                self.config.reflection_max_tokens,
                "trace2skill_analyze",
                request_nonce=f"skill-v{self.skill_version + 1}-analyst-{analyst_index}",
                audit_context=self._skill_optimizer_context(
                    "trace2skill_parallel_patch_proposal",
                    analyst_index=analyst_index,
                    frozen_incumbent_hash=incumbent_hash,
                ),
            )
            if proposal is not None:
                clean_proposal, rejected = sanitize_trace2skill_proposal(proposal)
                proposal_rejections.extend(
                    {"analyst_index": analyst_index, **item} for item in rejected
                )
                if clean_proposal.get("patch"):
                    proposals.append(clean_proposal)
        if not proposals:
            self._record_skill_update(
                {
                    "method": "trace2skill",
                    "committed": False,
                    "reason": "no_valid_patch_proposal",
                    "state_artifact": "trace_patch_skill",
                    "schema_rejections": proposal_rejections,
                    "mechanism": "parallel_patch_proposal",
                }
            )
            return
        consolidated = self._request_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You are Trace2Skill's reduce/apply stage. Consolidate independent patches against the same "
                        "frozen incumbent, resolve conflicts hierarchically, preserve protected behavior, and emit one "
                        "complete replacement skill document. Reject episode-specific or unsupported edits. Return "
                        "exactly one JSON object."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "frozen_incumbent_skill": incumbent,
                            "independent_patch_proposals": proposals,
                            "output_schema": {
                                "skill_document": "complete conflict-free reusable Markdown skill",
                                "resolved_conflicts": ["short conflict decision"],
                                "applied_patch_ids": ["analyst/patch index"],
                                "rejected_patch_ids": ["analyst/patch index and reason"],
                            },
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            self.config.reflection_max_tokens,
            "trace2skill_consolidate",
            request_nonce=f"skill-v{self.skill_version + 1}-reduce",
            audit_context=self._skill_optimizer_context(
                "trace2skill_hierarchical_consolidation",
                proposal_count=len(proposals),
                frozen_incumbent_hash=incumbent_hash,
            ),
        )
        if consolidated is None:
            return
        candidate = sanitize_trace2skill_document(
            consolidated.get("skill_document"), self.config.skill_document_max_chars
        )
        accepted = bool(candidate) and candidate != incumbent
        if accepted:
            self.skill_document = candidate
            self.skill_version += 1
            history = self.skill_method_state.setdefault("proposal_history", [])
            history.append(
                {
                    "skill_version": self.skill_version,
                    "frozen_incumbent_hash": incumbent_hash,
                    "proposal_count": len(proposals),
                    "proposal_hashes": [
                        hashlib.sha256(
                            json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8")
                        ).hexdigest()
                        for item in proposals
                    ],
                    "resolved_conflicts": consolidated.get("resolved_conflicts", []),
                    "applied_patch_ids": consolidated.get("applied_patch_ids", []),
                    "rejected_patch_ids": consolidated.get("rejected_patch_ids", []),
                }
            )
            self.skill_method_state["proposal_history"] = history[-50:]
        self._record_skill_update(
            {
                "method": "trace2skill",
                "committed": accepted,
                "reason": "conflict_free_patch_applied" if accepted else "empty_or_unchanged_document",
                "skill_version": self.skill_version,
                "state_artifact": "trace_patch_skill",
                "proposal_count": len(proposals),
                "schema_rejections": proposal_rejections,
                "resolved_conflicts": consolidated.get("resolved_conflicts", []),
                "applied_patch_ids": consolidated.get("applied_patch_ids", []),
                "rejected_patch_ids": consolidated.get("rejected_patch_ids", []),
                "mechanism": "hierarchical_patch_consolidation",
            }
        )

    def _run_skillgrad_update(self, result: EpisodeResult) -> None:
        """Adapt SkillGrad's diagnose, momentum, and layer-aware patch pipeline."""
        packet = self._skill_learning_packet(result)
        diagnosis = self._request_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You are SkillGrad's diagnoser. Compare the executor's observable trajectory and outcome with "
                        "the current layered skill package. Identify a causal mechanism and the missing reasoning or "
                        "action step. Return exactly one JSON object; do not use hidden task metadata."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "current_skill_package": {
                                key: self.skill_method_state.get(key)
                                for key in ("routing", "body", "references")
                            },
                            "learning_evidence": packet,
                            "output_schema": {
                                "label": "success|failure|mixed",
                                "signal": "observable evidence",
                                "causal_mechanism": "cause of result",
                                "robust_action": "generalizable action",
                                "skipped_reasoning_step": "missing or weak step",
                            },
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            self.config.reflection_max_tokens,
            "skillgrad_diagnose",
            request_nonce=f"skill-v{self.skill_version + 1}-diagnose",
            audit_context=self._skill_optimizer_context("skillgrad_diagnosis"),
        )
        if diagnosis is None:
            return
        momentum = self._request_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You are SkillGrad's momentum updater. Consolidate the new diagnosis with persistent success "
                        "and failure patterns. Return the complete bounded pattern list plus a task-local overlay that "
                        "states the gap and proposed change. Merge repetitions instead of duplicating them. Return "
                        "exactly one JSON object."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "persistent_patterns": self.skill_method_state.get(
                                "momentum_patterns", []
                            ),
                            "new_diagnosis": diagnosis,
                            "max_patterns": self.config.skillgrad_max_patterns,
                            "output_schema": {
                                "patterns": [
                                    {
                                        "pattern_id": "stable identifier",
                                        "kind": "success|failure|mixed",
                                        "anchor": "skill section or behavior",
                                        "appeared_in": "positive occurrence count",
                                        "description": "cross-task pattern",
                                        "latest_executor_action": "recent observed action",
                                        "remedy_log": ["prior or current remedy"],
                                    }
                                ],
                                "overlay": {
                                    "signal": "current signal",
                                    "pattern": "matched persistent pattern",
                                    "anchor": "target layer or section",
                                    "gap": "current skill gap",
                                    "proposed_change": "specific change",
                                },
                            },
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            self.config.reflection_max_tokens,
            "skillgrad_momentum",
            request_nonce=f"skill-v{self.skill_version + 1}-momentum",
            audit_context=self._skill_optimizer_context("skillgrad_pattern_momentum"),
        )
        if momentum is None:
            return
        patterns, overlay, momentum_rejections = sanitize_skillgrad_momentum(
            momentum, max_patterns=self.config.skillgrad_max_patterns
        )
        if (
            any(item.get("reason") in {"momentum_not_object", "patterns_not_list"} for item in momentum_rejections)
            or (self.skill_method_state.get("momentum_patterns") and not patterns)
        ):
            self._record_skill_update(
                {
                    "method": "skillgrad",
                    "committed": False,
                    "reason": "invalid_momentum_state",
                    "state_artifact": "layered_skill_package",
                    "momentum_schema_rejections": momentum_rejections,
                    "mechanism": "gradient_style_layered_patch",
                }
            )
            return
        patched = self._request_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You are SkillGrad's layer-aware patcher. Apply the diagnostic overlay to the correct layer: "
                        "L1 routing metadata controls activation, L2 body contains always-loaded general guidance, "
                        "and L3 references hold conditional detailed procedures and edge cases. Return the complete "
                        "updated package, preserving useful prior content. Return exactly one JSON object."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "current_skill_package": {
                                key: self.skill_method_state.get(key)
                                for key in ("routing", "body", "references")
                            },
                            "persistent_patterns": patterns,
                            "task_overlay": overlay,
                            "output_schema": {
                                "routing": {
                                    "name": "skill name",
                                    "description": "routing description",
                                    "activation_signals": ["observable activation signal"],
                                },
                                "body": "L2 reusable Markdown guidance",
                                "references": [
                                    {
                                        "name": "reference name",
                                        "when_to_load": "observable condition",
                                        "content": "L3 procedure or edge-case guidance",
                                    }
                                ],
                                "applied_pattern_ids": ["pattern id"],
                                "rationale": "short layer choice explanation",
                            },
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            self.config.reflection_max_tokens,
            "skillgrad_patch",
            request_nonce=f"skill-v{self.skill_version + 1}-patch",
            audit_context=self._skill_optimizer_context("skillgrad_layer_aware_patch"),
        )
        if patched is None:
            return
        package, package_rejections = sanitize_skillgrad_package(
            patched,
            max_references=self.config.skillgrad_max_references,
            max_content_chars=self.config.skill_document_max_chars,
        )
        candidate_state = {
            **package,
            "momentum_patterns": patterns,
            "last_overlay": overlay,
        }
        candidate_document = render_skillgrad_package(
            candidate_state, self.config.skill_document_max_chars
        )
        fatal_package_reasons = {"package_not_object", "routing_not_object", "references_not_list", "empty_body"}
        package_complete = not any(
            item.get("reason") in fatal_package_reasons for item in package_rejections
        )
        accepted = package_complete and candidate_state != self.skill_method_state
        if accepted:
            self.skill_method_state = candidate_state
            self.skill_document = candidate_document
            self.skill_version += 1
        self._record_skill_update(
            {
                "method": "skillgrad",
                "committed": accepted,
                "reason": "layer_aware_patch_applied" if accepted else "invalid_or_unchanged_package",
                "skill_version": self.skill_version,
                "state_artifact": "layered_skill_package",
                "pattern_count": len(patterns),
                "reference_count": len(package.get("references", [])),
                "applied_pattern_ids": patched.get("applied_pattern_ids", []),
                "momentum_schema_rejections": momentum_rejections,
                "package_schema_rejections": package_rejections,
                "mechanism": "gradient_style_layered_patch",
            }
        )

    def end_episode(self, result: EpisodeResult, allow_state_update: bool, expose_feedback: bool) -> None:
        if (
            not allow_state_update
            or not expose_feedback
            or self.read_only
            or self.condition == "baseline"
        ):
            return

        if self.condition == "context":
            self.context.append(
                {
                    "trajectory": copy.deepcopy(self._current_episode_steps),
                    "outcome": {
                        "score": result.score,
                        "return_pct": result.return_pct,
                        "max_drawdown": result.max_drawdown,
                        "turnover": result.turnover,
                        "violation_count": len(result.violations),
                        "fee_paid": result.fee_paid,
                    },
                }
            )
            self.context = self.context[-self.config.context_episodes :]
            self._trace.append(
                {
                    "event": "write",
                    "artifact": "context",
                    "committed": True,
                    "history_episode_count": len(self.context),
                    "window_step_count": sum(
                        len(item.get("trajectory", [])) for item in self.context
                    ),
                    "mechanism": "context_injection",
                }
            )
            return

        if self.condition == "skillopt":
            self._run_skillopt_update(result)
            return
        if self.condition == "skillboost":
            self._run_skillboost_update(result)
            return
        if self.condition == "skillx":
            self._run_skillx_update(result)
            return
        if self.condition == "trace2skill":
            self._run_trace2skill_update(result)
            return
        if self.condition == "skillgrad":
            self._run_skillgrad_update(result)
            return

        target_store = self.store
        if self.condition in {"episodic_memory", "consolidated_memory"}:
            for rule_id in self._used_rule_ids:
                recorded = target_store.record_application_outcome(rule_id, result.episode_id, result.score)
                self._trace.append(
                    {
                        "event": "attribute",
                        "rule_id": rule_id,
                        "score": result.score,
                        "recorded": recorded,
                        "mechanism": "outcome_attribution",
                    }
                )

        parsed = self._request_json(
            self._reflection_messages(result),
            self.config.reflection_max_tokens,
            "reflection",
            request_nonce="end_episode",
            audit_context={
                "token_attribution": "self_evolution_reflection",
                "self_evolution_phase": "experience_update",
                "condition": self.condition,
                "visible_memory_artifact_count": len(self._visible_rules),
                "used_rule_count": len(self._used_rule_ids),
            },
        )
        if parsed is None:
            return
        raw_action = parsed.get("action", "none")
        raw_invalidate = parsed.get("invalidate_rule_ids") or []
        if not isinstance(raw_action, str) or raw_action.lower() not in {"upsert", "invalidate", "none"}:
            self._trace.append(
                {
                    "event": "parse_error",
                    "phase": "reflection",
                    "error": "action must be upsert, invalidate, or none",
                }
            )
            return
        if not isinstance(raw_invalidate, list):
            self._trace.append(
                {
                    "event": "parse_error",
                    "phase": "reflection",
                    "error": "invalidate_rule_ids must be a list",
                }
            )
            return
        action = raw_action.lower()
        visible_ids = {entry.rule_id for entry in self._visible_rules}
        invalidate = [
            str(item)
            for item in raw_invalidate
            if str(item) in visible_ids
        ]
        if action == "invalidate":
            removed = [rule_id for rule_id in invalidate if target_store.remove_rule(rule_id)]
            self._trace.append(
                {
                    "event": "invalidate",
                    "rule_ids": removed,
                    "committed": bool(removed),
                    "mechanism": "conflict_revision",
                }
            )
            return
        if action != "upsert":
            self._trace.append(
                {
                    "event": "write",
                    "artifact": "strategy",
                    "committed": False,
                    "reason": "model_selected_none",
                    "mechanism": "experience_compression",
                }
            )
            return
        try:
            policy_label = self._sanitize_policy_label(parsed.get("policy_label"))
            if not isinstance(parsed.get("hypothesis"), str):
                raise ValueError("hypothesis must be a string")
            hypothesis = self._clean_text(parsed.get("hypothesis", ""), 1000)
            if not hypothesis:
                raise ValueError("hypothesis must be non-empty")
        except ValueError as exc:
            self._trace.append(
                {
                    "event": "parse_error",
                    "phase": "reflection",
                    "error": str(exc),
                }
            )
            return
        attribution = self._clean_text(parsed.get("attribution", ""), 1000)
        if self.condition == "reflection":
            self.reflection.append(
                {
                    "episode_id": result.episode_id,
                    "signature": result.feature_signature,
                    "policy_label": policy_label,
                    "hypothesis": hypothesis,
                    "attribution": attribution,
                    "score": result.score,
                }
            )
            if len(self.reflection) > self.store.max_entries:
                self.reflection = self.reflection[-self.store.max_entries :]
            self._trace.append(
                {
                    "event": "write",
                    "artifact": "reflection",
                    "signature": result.feature_signature,
                    "mode": policy_label,
                    "committed": True,
                    "mechanism": "experience_compression",
                }
            )
            return
        gate = 2 if self.condition == "consolidated_memory" else 1
        existing = target_store.retrieve(result.feature_signature)
        if existing is None:
            mechanism = "experience_compression"
        elif existing.mode == policy_label:
            mechanism = "evidence_consolidation"
        else:
            mechanism = "conflict_revision"
        committed, entry = target_store.propose(
            result.feature_signature,
            policy_label,
            result.episode_id,
            result.score,
            gate_evidence=gate,
            hypothesis=hypothesis,
            scope=StrategyStore._scope(result.feature_signature),
        )
        self._trace.append(
            {
                "event": "write",
                "artifact": "strategy",
                "signature": result.feature_signature,
                "mode": policy_label,
                "committed": committed,
                "rule_id": entry.rule_id if entry else None,
                "gate_evidence": gate,
                "mechanism": mechanism,
            }
        )

    def get_trace(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._trace)

    def state_snapshot(self) -> Dict[str, Any]:
        return {
            "type": "openai_compatible",
            "condition": self.condition,
            "config": self.config.public_dict(),
            "store": self.store.to_dict(),
            "reflection": copy.deepcopy(self.reflection),
            "context": copy.deepcopy(self.context),
            "skill_state": {
                "document": self.skill_document,
                "version": self.skill_version,
                "best_validation_score": self.skill_best_score,
                "method_state": copy.deepcopy(self.skill_method_state),
                "update_history": copy.deepcopy(self.skill_update_history),
            },
            "mechanism_state": {
                "active_artifacts": len(self.store.active_entries()) + int(bool(self.skill_document)),
                "pending_hypotheses": len(self.store.pending),
                "superseded_artifacts": sum(
                    1 for entry in self.store.entries if entry.status == "invalidated"
                ),
                "skill_version": self.skill_version,
                "accepted_skill_updates": sum(
                    1 for item in self.skill_update_history if item.get("committed")
                ),
                "rejected_skill_updates": sum(
                    1 for item in self.skill_update_history if not item.get("committed")
                ),
            },
            "state_hash": self.state_hash(),
        }

    def state_hash(self) -> str:
        payload = {
            "condition": self.condition,
            "store": self.store.to_dict(),
            "reflection": self.reflection,
            "context": self.context,
            "skill_document": self.skill_document,
            "skill_version": self.skill_version,
            "skill_method_state": self.skill_method_state,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
