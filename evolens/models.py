"""Serializable data models used by the dataset and benchmark runner."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RiskContract:
    initial_cash: float = 10_000.0
    initial_position: int = 20
    max_abs_position: int = 50
    max_order_size: int = 8
    max_turnover: float = 4.0
    max_drawdown: float = 0.20
    allow_short: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "RiskContract":
        return cls(**value)


@dataclass(frozen=True)
class MarketSourceSpec:
    """Opaque pointer to a frozen market resource.

    Provider, symbol, timestamps, and future bars deliberately live in the
    dataset resource catalog rather than in the object passed to an agent.
    """

    kind: str = "procedural"
    resource_id: str = ""
    clock: str = "step"
    calibration_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "MarketSourceSpec":
        return cls(**value)


@dataclass(frozen=True)
class ScenarioSpec:
    episode_id: str
    family_id: str
    role: str
    layer: str
    variant: str
    split: str
    hidden: bool
    horizon: int
    initial_price: float
    mechanism_params: Dict[str, float]
    opponent_mix: Dict[str, int]
    risk_contract: RiskContract
    observation_mapping_seed: int
    environment_seed: int
    evolution_mechanisms: List[str] = field(default_factory=list)
    scoring_spec: Dict[str, float] = field(default_factory=dict)
    market_source: MarketSourceSpec = field(default_factory=MarketSourceSpec)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["risk_contract"] = self.risk_contract.to_dict()
        if self.market_source == MarketSourceSpec():
            # Preserve the byte representation of already published V1/V2
            # episode files.
            result.pop("market_source", None)
        else:
            result["market_source"] = self.market_source.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ScenarioSpec":
        payload = dict(value)
        payload["risk_contract"] = RiskContract.from_dict(payload["risk_contract"])
        payload["market_source"] = MarketSourceSpec.from_dict(payload.get("market_source", {}))
        return cls(**payload)


@dataclass(frozen=True)
class StreamEvent:
    episode_id: str
    allow_state_update: bool
    expose_feedback: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "StreamEvent":
        return cls(**value)


@dataclass(frozen=True)
class CheckpointSpec:
    checkpoint_id: str
    after_event: int
    label: str
    probe_episode_ids: List[str]
    target_mechanisms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "CheckpointSpec":
        return cls(**value)


@dataclass(frozen=True)
class StreamSpec:
    stream_id: str
    template: str
    layer: str
    focal_family: str
    events: List[StreamEvent]
    checkpoints: List[CheckpointSpec]
    stream_seed: int
    target_mechanisms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "template": self.template,
            "layer": self.layer,
            "focal_family": self.focal_family,
            "events": [event.to_dict() for event in self.events],
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
            "stream_seed": self.stream_seed,
            "target_mechanisms": list(self.target_mechanisms),
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "StreamSpec":
        return cls(
            stream_id=value["stream_id"],
            template=value["template"],
            layer=value["layer"],
            focal_family=value["focal_family"],
            events=[StreamEvent.from_dict(item) for item in value["events"]],
            checkpoints=[CheckpointSpec.from_dict(item) for item in value["checkpoints"]],
            stream_seed=value["stream_seed"],
            target_mechanisms=list(value.get("target_mechanisms", [])),
        )


@dataclass(frozen=True)
class OrderIntent:
    side: str
    quantity: int
    limit_price: Optional[float] = None
    tag: str = "policy"


@dataclass(frozen=True)
class Observation:
    episode_id: str
    layer: str
    step: int
    horizon: int
    asset_symbol: str
    price: float
    fundamental: float
    public_signal: float
    spread_bps: float
    liquidity: float
    prices: List[float]
    fundamentals: List[float]
    cash: float
    position: int
    marked_wealth: float
    initial_wealth: float
    max_drawdown_so_far: float
    turnover_so_far: float
    risk_contract: RiskContract
    market_features: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Trade:
    step: int
    buyer: str
    seller: str
    quantity: int
    price: float
    buyer_fee: float
    seller_fee: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeResult:
    episode_id: str
    family_id: str
    role: str
    layer: str
    strategy_mode: str
    initial_wealth: float
    final_wealth: float
    pnl: float
    return_pct: float
    score: float
    max_drawdown: float
    turnover: float
    violations: List[str]
    trades: List[Trade]
    prices: List[float]
    fundamentals: List[float]
    feature_signature: str
    fee_paid: float
    market_diagnostics: Dict[str, Any]
    agent_trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["trades"] = [trade.to_dict() for trade in self.trades]
        return result


@dataclass
class EvaluationRecord:
    condition: str
    campaign_id: int
    stream_id: str
    checkpoint_id: str
    checkpoint_order: int
    probe_episode_id: str
    family_id: str
    role: str
    layer: str
    repeat_id: int
    score: float
    return_pct: float
    final_wealth: float
    max_drawdown: float
    turnover: float
    violation_count: int
    fee_paid: float
    state_hash: str
    strategy_mode: str
    stream_template: str = "unspecified"
    track_kind: str = "unspecified"
    retrieval_count: int = 0
    retrieval_hit_count: int = 0
    application_count: int = 0
    action_count: int = 0
    execution_seed: int = 0
    mechanism_ids: List[str] = field(default_factory=list)
    exact_retrieval_count: int = 0
    scope_transfer_retrieval_count: int = 0
    model_call_count: int = 0
    model_error_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    agent_trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
