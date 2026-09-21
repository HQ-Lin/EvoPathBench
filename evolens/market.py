"""Deterministic exogenous and endogenous market simulators."""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .calibration import CalibrationCatalog
from .features import feature_signature
from .market_data import FrozenMarketCatalog, ReplayWindow
from .models import EpisodeResult, Observation, OrderIntent, RiskContract, ScenarioSpec, Trade


@dataclass
class _Account:
    cash: float
    position: int
    turnover_notional: float = 0.0
    fee_paid: float = 0.0


@dataclass
class _AuctionOrder:
    agent_id: str
    side: str
    quantity: int
    limit_price: float
    tag: str


@dataclass(frozen=True)
class _LatentMarketPath:
    fundamentals: List[float]
    volatility_multipliers: List[float]
    regimes: List[str]
    permanent_event_shocks: List[float]
    transitory_event_shocks: List[float]


class MarketSimulator:
    """Runs one benchmark episode from a frozen ScenarioSpec."""

    def __init__(
        self,
        market_catalog: Optional[FrozenMarketCatalog] = None,
        calibration_catalog: Optional[CalibrationCatalog] = None,
    ) -> None:
        self.market_catalog = market_catalog
        self.calibration_catalog = calibration_catalog

    def run(self, scenario: ScenarioSpec, agent: Any, execution_seed: int = 0) -> EpisodeResult:
        agent.begin_episode(scenario, execution_seed)
        if scenario.market_source.kind == "binance_replay":
            if scenario.layer != "exogenous":
                raise ValueError("Binance historical replay only supports the exogenous layer")
            if self.market_catalog is None:
                raise ValueError("Binance replay scenario requires a frozen market catalog")
            result = self._run_historical_exogenous(
                scenario, agent, self.market_catalog.get(scenario.market_source.resource_id)
            )
        elif scenario.market_source.kind == "calibrated_procedural":
            if self.calibration_catalog is None:
                raise ValueError("calibrated scenario requires a frozen calibration profile")
            self.calibration_catalog.get(scenario.market_source.calibration_id)
            if scenario.layer == "exogenous":
                result = self._run_exogenous(scenario, agent)
            elif scenario.layer == "endogenous":
                result = self._run_endogenous(scenario, agent)
            else:
                raise ValueError(f"unsupported market layer: {scenario.layer}")
        elif scenario.layer == "exogenous":
            result = self._run_exogenous(scenario, agent)
        elif scenario.layer == "endogenous":
            result = self._run_endogenous(scenario, agent)
        else:
            raise ValueError(f"unsupported market layer: {scenario.layer}")
        return result

    @staticmethod
    def _process_version(scenario: ScenarioSpec) -> int:
        return int(scenario.mechanism_params.get("price_process_version", 1.0))

    @staticmethod
    def _student_t(rng: random.Random, degrees_of_freedom: float) -> float:
        """Draw a standardized heavy-tailed shock using only the stdlib RNG."""
        degrees_of_freedom = max(2.1, degrees_of_freedom)
        numerator = rng.gauss(0.0, 1.0)
        denominator = math.sqrt(rng.gammavariate(degrees_of_freedom / 2.0, 2.0) / degrees_of_freedom)
        raw = numerator / max(denominator, 1e-12)
        # Student-t(df) has variance df / (df - 2).  Scaling it to unit
        # variance makes jump_scale comparable across different tail choices.
        return raw * math.sqrt((degrees_of_freedom - 2.0) / degrees_of_freedom)

    @staticmethod
    def _regime_weights(family: str, params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        weights = {
            "balanced": 0.45,
            "directional": 0.20,
            "volatile": 0.20,
            "reversal": 0.15,
        }
        if family == "trend":
            weights.update(balanced=0.25, directional=0.50, volatile=0.15, reversal=0.10)
        elif family == "mean_reversion":
            weights.update(balanced=0.55, directional=0.12, volatile=0.18, reversal=0.15)
        elif family == "event_jump":
            weights.update(balanced=0.38, directional=0.12, volatile=0.38, reversal=0.12)
        elif family == "liquidity":
            weights.update(balanced=0.35, directional=0.15, volatile=0.38, reversal=0.12)
        elif family == "opponent_reflexivity":
            weights.update(balanced=0.25, directional=0.35, volatile=0.25, reversal=0.15)
        elif family == "risk_contract":
            weights.update(balanced=0.25, directional=0.12, volatile=0.48, reversal=0.15)
        if params and all(f"regime_weight_{name}" in params for name in weights):
            weights = {name: max(0.001, params[f"regime_weight_{name}"]) for name in weights}
        return weights

    def _price_innovation(self, rng: random.Random, scenario: ScenarioSpec) -> float:
        if scenario.market_source.kind != "calibrated_procedural":
            return rng.gauss(0.0, 1.0)
        if self.calibration_catalog is None:
            raise ValueError("calibrated innovation requires calibration catalog")
        quantiles = self.calibration_catalog.get(
            scenario.market_source.calibration_id
        ).fit_estimate["standardized_innovation_quantiles"]
        position = rng.random() * (len(quantiles) - 1)
        lower = int(math.floor(position))
        upper = min(len(quantiles) - 1, lower + 1)
        weight = position - lower
        return quantiles[lower] * (1.0 - weight) + quantiles[upper] * weight

    @staticmethod
    def _draw_regime(rng: random.Random, weights: Dict[str, float]) -> str:
        draw = rng.random() * sum(weights.values())
        cumulative = 0.0
        for regime, weight in weights.items():
            cumulative += weight
            if draw <= cumulative:
                return regime
        return next(reversed(weights))

    def _latent_market_path(self, scenario: ScenarioSpec) -> _LatentMarketPath:
        """Generate V2 latent regimes, clustered volatility, and random events."""
        rng = random.Random(scenario.environment_seed ^ 0xA5A5A5)
        params = scenario.mechanism_params
        weights = self._regime_weights(scenario.family_id, params)
        regime = self._draw_regime(rng, weights)
        regime_names = ("balanced", "directional", "volatile", "reversal")
        has_transition_matrix = scenario.market_source.kind == "calibrated_procedural" and all(
            f"regime_transition_{source}_{target}" in params
            for source in regime_names
            for target in regime_names
        )
        regime_persistence = params.get("regime_persistence", 0.82)
        volatility_persistence = params.get("volatility_persistence", 0.88)
        volatility_of_volatility = params.get("volatility_of_volatility", 0.18)
        base_volatility = max(1e-6, params.get("fundamental_volatility", 0.002))
        log_volatility = math.log(base_volatility)
        jump_intensity = params.get("jump_intensity", 0.018)
        jump_scale = max(1e-6, params.get("jump_scale", params.get("jump_size", 0.018)))
        jump_tail_df = params.get("jump_tail_df", 4.0)
        aftershock_decay = params.get("jump_aftershock_decay", 0.68)
        permanent_probability = params.get("permanent_event_probability", 0.55)
        calibrated_scale = (
            params.get("calibration_shock_scale", 1.0)
            if scenario.market_source.kind == "calibrated_procedural"
            else 1.0
        )
        structural_sign = -1.0 if scenario.variant == "reversed" else 1.0
        direction_probability = 0.68 if scenario.family_id in {"trend", "event_jump"} else 0.50
        if structural_sign < 0:
            direction_probability = 1.0 - direction_probability
        # Event-jump tasks must actually contain an event, but its location,
        # sign, magnitude, permanence, and any additional events stay random.
        # This avoids the old fixed-midpoint shortcut without producing empty
        # positive examples that no agent could learn from.
        required_event_step = (
            rng.randrange(scenario.horizon) if scenario.family_id == "event_jump" and scenario.horizon else None
        )

        value = scenario.initial_price
        aftershock = 0.0
        fundamentals: List[float] = []
        volatility_multipliers: List[float] = []
        regimes: List[str] = []
        permanent_events: List[float] = []
        transitory_events: List[float] = []
        regime_volatility = {"balanced": 0.85, "directional": 0.95, "volatile": 1.75, "reversal": 1.15}
        event_multiplier = {"balanced": 0.70, "directional": 0.90, "volatile": 1.80, "reversal": 1.20}

        for step in range(scenario.horizon):
            if step > 0:
                if has_transition_matrix:
                    regime = self._draw_regime(
                        rng,
                        {
                            target: params[f"regime_transition_{regime}_{target}"]
                            for target in regime_names
                        },
                    )
                elif rng.random() > regime_persistence:
                    candidates = dict(weights)
                    candidates[regime] *= 0.25
                    regime = self._draw_regime(rng, candidates)

            target_volatility = base_volatility * regime_volatility[regime]
            log_target = math.log(max(target_volatility, 1e-8))
            log_volatility = (
                log_target
                + volatility_persistence * (log_volatility - log_target)
                + volatility_of_volatility * rng.gauss(0.0, 1.0)
            )
            volatility = min(base_volatility * 4.5, max(base_volatility * 0.25, math.exp(log_volatility)))
            volatility *= 1.0 + min(2.0, 0.35 * aftershock)

            event_probability = min(
                0.45,
                jump_intensity * event_multiplier[regime] + min(0.12, 0.025 * aftershock),
            )
            permanent_event = 0.0
            transitory_event = 0.0
            realized_event = 0.0
            event_draw = rng.random()
            if step == required_event_step or event_draw < event_probability:
                sign = 1.0 if rng.random() < direction_probability else -1.0
                magnitude = min(0.22, jump_scale * abs(self._student_t(rng, jump_tail_df)))
                realized_event = sign * magnitude
                if rng.random() < permanent_probability:
                    permanent_event = realized_event
                else:
                    transitory_event = realized_event

            base_drift = params.get("fundamental_drift", 0.0)
            directional_drift = base_drift + 0.30 * abs(params.get("price_drift", 0.0))
            if regime == "directional":
                drift = structural_sign * directional_drift
            elif regime == "reversal":
                drift = -structural_sign * max(abs(directional_drift), 0.0004)
            elif regime == "volatile":
                drift = 0.10 * structural_sign * base_drift
            else:
                drift = 0.35 * structural_sign * base_drift
            value *= math.exp(
                drift
                + calibrated_scale
                * (volatility * self._price_innovation(rng, scenario) + permanent_event)
            )
            value = max(1.0, value)

            fundamentals.append(value)
            volatility_multipliers.append(volatility / base_volatility)
            regimes.append(regime)
            permanent_events.append(permanent_event)
            transitory_events.append(transitory_event)
            aftershock = aftershock * aftershock_decay
            if realized_event:
                aftershock += min(4.0, abs(realized_event) / jump_scale)

        return _LatentMarketPath(
            fundamentals=fundamentals,
            volatility_multipliers=volatility_multipliers,
            regimes=regimes,
            permanent_event_shocks=permanent_events,
            transitory_event_shocks=transitory_events,
        )

    def _legacy_fundamental_path(self, scenario: ScenarioSpec) -> List[float]:
        rng = random.Random(scenario.environment_seed ^ 0xA5A5A5)
        params = scenario.mechanism_params
        value = scenario.initial_price
        path: List[float] = []
        drift = params.get("fundamental_drift", 0.0)
        volatility = params.get("fundamental_volatility", 0.002)
        jump_step = int(params.get("jump_step", -1))
        jump_size = params.get("jump_size", 0.0)
        for step in range(scenario.horizon):
            shock = rng.gauss(0.0, volatility)
            if step == jump_step:
                shock += jump_size
            value *= math.exp(drift + shock)
            path.append(max(1.0, value))
        return path

    def _fundamental_path(self, scenario: ScenarioSpec) -> List[float]:
        if self._process_version(scenario) >= 2:
            return self._latent_market_path(scenario).fundamentals
        return self._legacy_fundamental_path(scenario)

    def _legacy_exogenous_price_path(self, scenario: ScenarioSpec, fundamentals: List[float]) -> List[float]:
        rng = random.Random(scenario.environment_seed)
        params = scenario.mechanism_params
        family = scenario.family_id
        price = scenario.initial_price
        anchor = scenario.initial_price
        deviation = 0.0
        path: List[float] = []
        volatility = params.get("price_volatility", 0.007)
        drift = params.get("price_drift", 0.0)
        reversal = -1.0 if scenario.variant == "reversed" else 1.0
        for step in range(scenario.horizon):
            noise = rng.gauss(0.0, volatility)
            if family == "trend":
                price *= math.exp(reversal * drift + noise)
            elif family == "mean_reversion":
                deviation = 0.55 * deviation + noise
                if step == 0:
                    deviation += params.get("initial_gap", 0.025) * reversal
                price = fundamentals[step] * math.exp(deviation)
            elif family == "event_jump":
                jump_step = int(params.get("jump_step", scenario.horizon // 2))
                jump = params.get("jump_size", 0.06) * reversal if step == jump_step else 0.0
                price *= math.exp(noise + jump)
            elif family == "liquidity":
                price *= math.exp(noise + 0.25 * drift)
            elif family == "opponent_reflexivity":
                recent = (path[-1] / path[-3] - 1.0) if len(path) >= 3 else 0.0
                price *= math.exp(noise + 0.30 * recent)
            else:  # risk_contract and future compatible families
                pull = 0.08 * math.log(max(anchor, 1.0) / max(price, 1.0))
                price *= math.exp(noise + pull)
            path.append(max(1.0, price))
        return path

    def _layered_exogenous_price_path(
        self,
        scenario: ScenarioSpec,
        latent: _LatentMarketPath,
    ) -> List[float]:
        """Map the V2 latent path to observable prices with temporary mispricing."""
        rng = random.Random(scenario.environment_seed)
        params = scenario.mechanism_params
        structural_sign = -1.0 if scenario.variant == "reversed" else 1.0
        persistence = params.get("mispricing_persistence", 0.75)
        coupling = params.get("fundamental_price_coupling", 0.16)
        base_volatility = params.get("price_volatility", 0.007)
        microstructure_noise = params.get("microstructure_noise", 0.0008)
        calibrated_scale = (
            params.get("calibration_shock_scale", 1.0)
            if scenario.market_source.kind == "calibrated_procedural"
            else 1.0
        )
        mispricing = params.get("initial_gap", 0.0) * structural_sign
        previous_fundamental = scenario.initial_price
        path: List[float] = []

        for step, fundamental in enumerate(latent.fundamentals):
            regime = latent.regimes[step]
            local_persistence = persistence
            if regime == "directional":
                local_persistence = min(0.985, local_persistence + 0.05)
            elif regime == "reversal":
                local_persistence = max(0.35, local_persistence - 0.12)
            elif regime == "volatile":
                local_persistence = max(0.45, local_persistence - 0.06)

            force = 0.0
            if scenario.family_id == "trend":
                force = structural_sign * params.get("price_drift", 0.0)
            elif scenario.family_id == "liquidity":
                force = 0.20 * structural_sign * params.get("price_drift", 0.0)
            elif scenario.family_id == "opponent_reflexivity" and len(path) >= 3:
                force = 0.28 * (path[-1] / path[-3] - 1.0)
            elif scenario.family_id == "risk_contract" and path:
                force = 0.08 * math.log(scenario.initial_price / max(path[-1], 1.0))

            fundamental_return = math.log(fundamental / max(previous_fundamental, 1e-12))
            noise = calibrated_scale * (
                base_volatility
                * latent.volatility_multipliers[step]
                * self._price_innovation(rng, scenario)
            )
            mispricing = (
                local_persistence * mispricing
                + force
                + noise
                + calibrated_scale * latent.transitory_event_shocks[step]
                - (1.0 - coupling) * fundamental_return
            )
            micro_noise = calibrated_scale * microstructure_noise * rng.gauss(0.0, 1.0)
            observed = fundamental * math.exp(min(0.35, max(-0.35, mispricing + micro_noise)))
            path.append(max(1.0, observed))
            previous_fundamental = fundamental
        return path

    def _exogenous_price_path(self, scenario: ScenarioSpec, fundamentals: List[float]) -> List[float]:
        """Compatibility wrapper retained for callers of the original private helper."""
        if self._process_version(scenario) >= 2:
            latent = self._latent_market_path(scenario)
            return self._layered_exogenous_price_path(scenario, latent)
        return self._legacy_exogenous_price_path(scenario, fundamentals)

    @staticmethod
    def _layered_diagnostics(latent: _LatentMarketPath, prices: List[float]) -> Dict[str, float]:
        event_count = sum(
            1
            for permanent, transitory in zip(
                latent.permanent_event_shocks,
                latent.transitory_event_shocks,
            )
            if permanent or transitory
        )
        returns = [current / previous - 1.0 for previous, current in zip(prices, prices[1:]) if previous > 0]
        return {
            "price_process_version": 2.0,
            "event_count": float(event_count),
            "permanent_event_count": float(sum(1 for value in latent.permanent_event_shocks if value)),
            "transitory_event_count": float(sum(1 for value in latent.transitory_event_shocks if value)),
            "regime_switch_count": float(
                sum(1 for previous, current in zip(latent.regimes, latent.regimes[1:]) if previous != current)
            ),
            "mean_volatility_multiplier": statistics.fmean(latent.volatility_multipliers),
            "max_volatility_multiplier": max(latent.volatility_multipliers),
            "realized_price_volatility": statistics.pstdev(returns) if len(returns) > 1 else 0.0,
        }

    @staticmethod
    def _marked_wealth(account: _Account, price: float) -> float:
        return account.cash + account.position * price

    @staticmethod
    def _asset_symbol(seed: int) -> str:
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        rng = random.Random(seed)
        return "".join(rng.choice(alphabet) for _ in range(4))

    @staticmethod
    def _liquidation_wealth(
        account: _Account,
        valuation_price: float,
        spread_bps: float,
        liquidity: float,
        fee_bps: float,
    ) -> float:
        if account.position == 0:
            return account.cash
        half_spread = spread_bps / 20_000.0
        impact = min(0.04, abs(account.position) / max(liquidity, 1.0) * 0.0015)
        if account.position > 0:
            exit_price = valuation_price * (1.0 - half_spread - impact)
        else:
            exit_price = valuation_price * (1.0 + half_spread + impact)
        notional = account.position * exit_price
        fee = abs(notional) * fee_bps / 10_000.0
        return account.cash + notional - fee

    @staticmethod
    def _score(
        initial_wealth: float,
        final_wealth: float,
        max_drawdown: float,
        contract: RiskContract,
        violations: List[str],
    ) -> Tuple[float, float]:
        return_pct = final_wealth / initial_wealth - 1.0
        unique_violations = len(set(violations))
        penalty = 0.01 * unique_violations
        if max_drawdown > contract.max_drawdown:
            penalty += 2.0 * (max_drawdown - contract.max_drawdown)
        return return_pct - penalty, return_pct

    @staticmethod
    def _validate_intent(
        intent: OrderIntent,
        account: _Account,
        contract: RiskContract,
        reference_price: float,
        initial_wealth: float,
    ) -> Optional[str]:
        if intent.side not in {"buy", "sell"}:
            return "invalid_side"
        if intent.quantity <= 0:
            return "invalid_quantity"
        if intent.quantity > contract.max_order_size:
            return "max_order_size"
        signed = intent.quantity if intent.side == "buy" else -intent.quantity
        next_position = account.position + signed
        if abs(next_position) > contract.max_abs_position:
            return "max_position"
        if not contract.allow_short and next_position < 0:
            return "short_not_allowed"
        projected_turnover = (account.turnover_notional + intent.quantity * reference_price) / initial_wealth
        if projected_turnover > contract.max_turnover:
            return "max_turnover"
        return None

    def _observation(
        self,
        scenario: ScenarioSpec,
        step: int,
        price: float,
        fundamental: float,
        public_signal: float,
        prices: List[float],
        fundamentals: List[float],
        account: _Account,
        initial_wealth: float,
        max_drawdown: float,
        market_features: Optional[Dict[str, float]] = None,
    ) -> Observation:
        params = scenario.mechanism_params
        return Observation(
            episode_id=scenario.episode_id,
            layer=scenario.layer,
            step=step,
            horizon=scenario.horizon,
            asset_symbol=self._asset_symbol(scenario.observation_mapping_seed),
            price=price,
            fundamental=fundamental,
            public_signal=public_signal,
            spread_bps=params.get("spread_bps", 10.0),
            liquidity=(market_features or {}).get("execution_liquidity", params.get("liquidity", 20.0)),
            prices=list(prices),
            fundamentals=list(fundamentals),
            cash=account.cash,
            position=account.position,
            marked_wealth=self._marked_wealth(account, price),
            initial_wealth=initial_wealth,
            max_drawdown_so_far=max_drawdown,
            turnover_so_far=account.turnover_notional / initial_wealth,
            risk_contract=scenario.risk_contract,
            market_features=dict(market_features or {}),
        )

    def _run_historical_exogenous(
        self,
        scenario: ScenarioSpec,
        agent: Any,
        window: ReplayWindow,
    ) -> EpisodeResult:
        """Replay closed Binance bars with causal next-open execution."""
        horizon = scenario.horizon
        series = (
            window.decision_prices,
            window.execution_prices,
            window.valuation_prices,
            window.reference_values,
            window.decision_time_us,
            window.execution_time_us,
        )
        if any(len(values) != horizon for values in series):
            raise ValueError(f"replay resource {window.resource_id} does not match horizon {horizon}")
        if any(decision >= execution for decision, execution in zip(window.decision_time_us, window.execution_time_us)):
            raise ValueError(f"replay resource {window.resource_id} violates next-open causality")
        params = scenario.mechanism_params
        contract = scenario.risk_contract
        account = _Account(contract.initial_cash, contract.initial_position)
        initial_wealth = self._marked_wealth(account, window.decision_prices[0])
        peak_wealth = initial_wealth
        max_drawdown = 0.0
        violations: List[str] = []
        trades: List[Trade] = []
        observed_prices: List[float] = []
        observed_references: List[float] = []
        spread_bps = params.get("spread_bps", 5.0)
        liquidity = params.get("liquidity", 20.0)
        fee_bps = params.get("fee_bps", 2.0)

        for step in range(horizon):
            decision_price = window.decision_prices[step]
            execution_reference = window.execution_prices[step]
            valuation_price = window.valuation_prices[step]
            reference = window.reference_values[step]
            observed_prices.append(decision_price)
            observed_references.append(reference)
            step_liquidity = liquidity * min(4.0, max(0.25, window.quote_volume_relative[step]))
            obs = self._observation(
                scenario,
                step,
                decision_price,
                reference,
                reference,
                observed_prices,
                observed_references,
                account,
                initial_wealth,
                max_drawdown,
                market_features={
                    "quote_volume_relative": window.quote_volume_relative[step],
                    "trade_count_relative": window.trade_count_relative[step],
                    "taker_buy_ratio": window.taker_buy_ratio[step],
                    "execution_liquidity": step_liquidity,
                },
            )
            for intent in agent.act(obs):
                problem = self._validate_intent(intent, account, contract, execution_reference, initial_wealth)
                if problem:
                    violations.append(problem)
                    continue
                half_spread = spread_bps / 20_000.0
                impact = min(0.03, intent.quantity / max(step_liquidity, 1.0) * 0.001)
                execution_price = execution_reference * (
                    1.0 + half_spread + impact if intent.side == "buy" else 1.0 - half_spread - impact
                )
                if intent.limit_price is not None:
                    if intent.side == "buy" and execution_price > intent.limit_price:
                        continue
                    if intent.side == "sell" and execution_price < intent.limit_price:
                        continue
                notional = execution_price * intent.quantity
                fee = notional * fee_bps / 10_000.0
                if intent.side == "buy":
                    if account.cash + 1e-9 < notional + fee:
                        violations.append("insufficient_cash")
                        continue
                    account.cash -= notional + fee
                    account.position += intent.quantity
                    buyer, seller = "focus", "historical_external"
                else:
                    if not contract.allow_short and account.position < intent.quantity:
                        violations.append("insufficient_inventory")
                        continue
                    account.cash += notional - fee
                    account.position -= intent.quantity
                    buyer, seller = "historical_external", "focus"
                account.turnover_notional += notional
                account.fee_paid += fee
                trades.append(
                    Trade(
                        step,
                        buyer,
                        seller,
                        intent.quantity,
                        execution_price,
                        fee if buyer == "focus" else 0.0,
                        fee if seller == "focus" else 0.0,
                    )
                )
            wealth = self._marked_wealth(account, valuation_price)
            peak_wealth = max(peak_wealth, wealth)
            if peak_wealth > 0:
                max_drawdown = max(max_drawdown, 1.0 - wealth / peak_wealth)

        final_reference = window.valuation_prices[-1]
        final_wealth = self._liquidation_wealth(account, final_reference, spread_bps, liquidity, fee_bps)
        score, return_pct = self._score(initial_wealth, final_wealth, max_drawdown, contract, violations)
        signature = feature_signature(observed_prices, observed_references, spread_bps, liquidity)
        returns = [current / previous - 1.0 for previous, current in zip(observed_prices, observed_prices[1:])]
        diagnostics: Dict[str, Any] = {
            "track_kind": "binance_replay",
            "resource_id": window.resource_id,
            "source_snapshot_sha256": window.source_snapshot_sha256,
            "execution_clock": "bar_close_next_open",
            "trade_count": float(len(trades)),
            "final_price": final_reference,
            "final_fundamental": window.reference_values[-1],
            "absolute_price_gap": abs(final_reference / max(window.reference_values[-1], 1e-12) - 1.0),
            "cash_conservation_error": 0.0,
            "asset_conservation_error": 0.0,
            "realized_price_volatility": statistics.pstdev(returns) if len(returns) > 1 else 0.0,
            "mean_taker_buy_ratio": statistics.fmean(window.taker_buy_ratio),
            "field_contract": window.field_contract,
        }
        return EpisodeResult(
            episode_id=scenario.episode_id,
            family_id=scenario.family_id,
            role=scenario.role,
            layer=scenario.layer,
            strategy_mode=agent.current_mode,
            initial_wealth=initial_wealth,
            final_wealth=final_wealth,
            pnl=final_wealth - initial_wealth,
            return_pct=return_pct,
            score=score,
            max_drawdown=max_drawdown,
            turnover=account.turnover_notional / initial_wealth,
            violations=violations,
            trades=trades,
            prices=observed_prices,
            fundamentals=observed_references,
            feature_signature=signature,
            fee_paid=account.fee_paid,
            market_diagnostics=diagnostics,
            agent_trace=agent.get_trace(),
        )

    def _run_exogenous(self, scenario: ScenarioSpec, agent: Any) -> EpisodeResult:
        params = scenario.mechanism_params
        latent: Optional[_LatentMarketPath] = None
        if self._process_version(scenario) >= 2:
            latent = self._latent_market_path(scenario)
            fundamentals = latent.fundamentals
            prices = self._layered_exogenous_price_path(scenario, latent)
        else:
            fundamentals = self._legacy_fundamental_path(scenario)
            prices = self._legacy_exogenous_price_path(scenario, fundamentals)
        signal_rng = random.Random(scenario.environment_seed ^ 0x51A1)
        contract = scenario.risk_contract
        account = _Account(contract.initial_cash, contract.initial_position)
        initial_wealth = self._marked_wealth(account, scenario.initial_price)
        peak_wealth = initial_wealth
        max_drawdown = 0.0
        violations: List[str] = []
        trades: List[Trade] = []
        observed_prices: List[float] = []
        observed_fundamentals: List[float] = []
        spread_bps = params.get("spread_bps", 10.0)
        liquidity = params.get("liquidity", 20.0)
        fee_bps = params.get("fee_bps", 2.0)
        for step, (price, fundamental) in enumerate(zip(prices, fundamentals)):
            observed_prices.append(price)
            observed_fundamentals.append(fundamental)
            public_signal = fundamental * (1.0 + signal_rng.gauss(0.0, params.get("signal_noise", 0.003)))
            obs = self._observation(
                scenario,
                step,
                price,
                fundamental,
                public_signal,
                observed_prices,
                observed_fundamentals,
                account,
                initial_wealth,
                max_drawdown,
            )
            intents = agent.act(obs)
            for intent in intents:
                problem = self._validate_intent(intent, account, contract, price, initial_wealth)
                if problem:
                    violations.append(problem)
                    continue
                half_spread = spread_bps / 20_000.0
                impact = min(0.03, intent.quantity / max(liquidity, 1.0) * 0.001)
                execution_price = price * (
                    1.0 + half_spread + impact if intent.side == "buy" else 1.0 - half_spread - impact
                )
                if intent.limit_price is not None:
                    if intent.side == "buy" and execution_price > intent.limit_price:
                        continue
                    if intent.side == "sell" and execution_price < intent.limit_price:
                        continue
                notional = execution_price * intent.quantity
                fee = notional * fee_bps / 10_000.0
                if intent.side == "buy":
                    if account.cash + 1e-9 < notional + fee:
                        violations.append("insufficient_cash")
                        continue
                    account.cash -= notional + fee
                    account.position += intent.quantity
                    buyer, seller = "focus", "external"
                else:
                    if not contract.allow_short and account.position < intent.quantity:
                        violations.append("insufficient_inventory")
                        continue
                    account.cash += notional - fee
                    account.position -= intent.quantity
                    buyer, seller = "external", "focus"
                account.turnover_notional += notional
                account.fee_paid += fee
                trades.append(Trade(step, buyer, seller, intent.quantity, execution_price, fee if buyer == "focus" else 0.0, fee if seller == "focus" else 0.0))
            wealth = self._marked_wealth(account, price)
            peak_wealth = max(peak_wealth, wealth)
            if peak_wealth > 0:
                max_drawdown = max(max_drawdown, 1.0 - wealth / peak_wealth)
        final_wealth = self._liquidation_wealth(account, prices[-1], spread_bps, liquidity, fee_bps)
        score, return_pct = self._score(initial_wealth, final_wealth, max_drawdown, contract, violations)
        signature = feature_signature(prices, fundamentals, spread_bps, liquidity)
        diagnostics = {
            "track_kind": scenario.market_source.kind,
            "trade_count": float(len(trades)),
            "final_price": prices[-1],
            "final_fundamental": fundamentals[-1],
            "absolute_price_gap": abs(prices[-1] / fundamentals[-1] - 1.0),
            "cash_conservation_error": 0.0,
            "asset_conservation_error": 0.0,
        }
        if latent is not None:
            diagnostics.update(self._layered_diagnostics(latent, prices))
        return EpisodeResult(
            episode_id=scenario.episode_id,
            family_id=scenario.family_id,
            role=scenario.role,
            layer=scenario.layer,
            strategy_mode=agent.current_mode,
            initial_wealth=initial_wealth,
            final_wealth=final_wealth,
            pnl=final_wealth - initial_wealth,
            return_pct=return_pct,
            score=score,
            max_drawdown=max_drawdown,
            turnover=account.turnover_notional / initial_wealth,
            violations=violations,
            trades=trades,
            prices=prices,
            fundamentals=fundamentals,
            feature_signature=signature,
            fee_paid=account.fee_paid,
            market_diagnostics=diagnostics,
            agent_trace=agent.get_trace(),
        )

    def _opponent_orders(
        self,
        kind: str,
        agent_id: str,
        price: float,
        fundamental: float,
        history: List[float],
        rng: random.Random,
        spread_bps: float,
    ) -> List[_AuctionOrder]:
        quantity = 2
        edge = max(0.0015, spread_bps / 10_000.0)
        if kind == "fundamental":
            if fundamental > price * 1.002:
                return [_AuctionOrder(agent_id, "buy", quantity, price * (1.0 + edge), kind)]
            if fundamental < price * 0.998:
                return [_AuctionOrder(agent_id, "sell", quantity, price * (1.0 - edge), kind)]
        elif kind in {"momentum", "contrarian"}:
            move = history[-1] / history[-3] - 1.0 if len(history) >= 3 else 0.0
            signal = move if kind == "momentum" else -move
            if signal > 0.001:
                return [_AuctionOrder(agent_id, "buy", quantity, price * (1.0 + edge), kind)]
            if signal < -0.001:
                return [_AuctionOrder(agent_id, "sell", quantity, price * (1.0 - edge), kind)]
        elif kind == "market_maker":
            width = max(0.002, spread_bps / 20_000.0)
            return [
                _AuctionOrder(agent_id, "buy", quantity, price * (1.0 - width), kind),
                _AuctionOrder(agent_id, "sell", quantity, price * (1.0 + width), kind),
            ]
        elif kind == "noise":
            side = "buy" if rng.random() < 0.5 else "sell"
            limit = price * (1.0 + edge if side == "buy" else 1.0 - edge)
            return [_AuctionOrder(agent_id, side, 1 + rng.randrange(3), limit, kind)]
        return []

    @staticmethod
    def _find_cross(buys: List[_AuctionOrder], sells: List[_AuctionOrder]) -> Optional[Tuple[int, int]]:
        best: Optional[Tuple[int, int, float]] = None
        for buy_index, buy in enumerate(buys):
            for sell_index, sell in enumerate(sells):
                if buy.agent_id == sell.agent_id or buy.limit_price + 1e-12 < sell.limit_price:
                    continue
                priority = buy.limit_price - sell.limit_price
                if best is None or priority > best[2]:
                    best = (buy_index, sell_index, priority)
        return (best[0], best[1]) if best is not None else None

    def _run_endogenous(self, scenario: ScenarioSpec, agent: Any) -> EpisodeResult:
        params = scenario.mechanism_params
        latent: Optional[_LatentMarketPath] = None
        if self._process_version(scenario) >= 2:
            latent = self._latent_market_path(scenario)
            fundamentals = latent.fundamentals
        else:
            fundamentals = self._legacy_fundamental_path(scenario)
        contract = scenario.risk_contract
        spread_bps = params.get("spread_bps", 10.0)
        liquidity = params.get("liquidity", 20.0)
        fee_bps = params.get("fee_bps", 2.0)
        accounts: Dict[str, _Account] = {"focus": _Account(contract.initial_cash, contract.initial_position)}
        opponents: List[Tuple[str, str]] = []
        for kind, count in sorted(scenario.opponent_mix.items()):
            for index in range(count):
                opponent_id = f"{kind}-{index}"
                opponents.append((opponent_id, kind))
                accounts[opponent_id] = _Account(contract.initial_cash, contract.initial_position)
        initial_cash_total = sum(account.cash for account in accounts.values())
        initial_asset_total = sum(account.position for account in accounts.values())
        focus = accounts["focus"]
        initial_wealth = self._marked_wealth(focus, scenario.initial_price)
        peak_wealth = initial_wealth
        max_drawdown = 0.0
        violations: List[str] = []
        trades: List[Trade] = []
        prices: List[float] = []
        observed_fundamentals: List[float] = []
        price = scenario.initial_price
        fee_sink = 0.0
        opponent_rng = random.Random(scenario.environment_seed ^ 0xC0FFEE)
        signal_rng = random.Random(scenario.environment_seed ^ 0x51A1)
        market_rng = random.Random(scenario.environment_seed ^ 0xBADA55)
        persistent_orderflow_impact = 0.0
        absolute_order_imbalance = 0.0
        absolute_orderflow_impact = 0.0
        for step, fundamental in enumerate(fundamentals):
            prices.append(price)
            observed_fundamentals.append(fundamental)
            signal = fundamental * (1.0 + signal_rng.gauss(0.0, params.get("signal_noise", 0.003)))
            obs = self._observation(
                scenario,
                step,
                price,
                fundamental,
                signal,
                prices,
                observed_fundamentals,
                focus,
                initial_wealth,
                max_drawdown,
            )
            submitted: List[_AuctionOrder] = []
            for intent in agent.act(obs):
                problem = self._validate_intent(intent, focus, contract, price, initial_wealth)
                if problem:
                    violations.append(problem)
                    continue
                limit = intent.limit_price
                if limit is None:
                    aggressiveness = max(0.003, spread_bps / 10_000.0)
                    limit = price * (1.0 + aggressiveness if intent.side == "buy" else 1.0 - aggressiveness)
                submitted.append(_AuctionOrder("focus", intent.side, intent.quantity, limit, intent.tag))
            for opponent_id, kind in opponents:
                submitted.extend(self._opponent_orders(kind, opponent_id, price, fundamental, prices, opponent_rng, spread_bps))
            submitted_buy_quantity = sum(order.quantity for order in submitted if order.side == "buy")
            submitted_sell_quantity = sum(order.quantity for order in submitted if order.side == "sell")
            submitted_quantity = submitted_buy_quantity + submitted_sell_quantity
            order_imbalance = (
                (submitted_buy_quantity - submitted_sell_quantity) / submitted_quantity
                if submitted_quantity
                else 0.0
            )
            buys = sorted((order for order in submitted if order.side == "buy"), key=lambda order: order.limit_price, reverse=True)
            sells = sorted((order for order in submitted if order.side == "sell"), key=lambda order: order.limit_price)
            step_trade_prices: List[float] = []
            while buys and sells:
                cross = self._find_cross(buys, sells)
                if cross is None:
                    break
                buy_index, sell_index = cross
                buy = buys[buy_index]
                sell = sells[sell_index]
                buyer = accounts[buy.agent_id]
                seller = accounts[sell.agent_id]
                trade_price = (buy.limit_price + sell.limit_price) / 2.0
                max_cash_qty = int(buyer.cash / (trade_price * (1.0 + fee_bps / 10_000.0)))
                max_position_qty = contract.max_abs_position - buyer.position
                max_inventory_qty = seller.position if not contract.allow_short else contract.max_abs_position + seller.position
                quantity = min(buy.quantity, sell.quantity, max_cash_qty, max_position_qty, max_inventory_qty)
                if quantity <= 0:
                    if max_cash_qty <= 0 or max_position_qty <= 0:
                        buys.pop(buy_index)
                    if max_inventory_qty <= 0:
                        sells.pop(sell_index)
                    continue
                notional = quantity * trade_price
                buyer_fee = notional * fee_bps / 10_000.0
                seller_fee = notional * fee_bps / 10_000.0
                buyer.cash -= notional + buyer_fee
                buyer.position += quantity
                seller.cash += notional - seller_fee
                seller.position -= quantity
                buyer.turnover_notional += notional
                seller.turnover_notional += notional
                buyer.fee_paid += buyer_fee
                seller.fee_paid += seller_fee
                fee_sink += buyer_fee + seller_fee
                trades.append(Trade(step, buy.agent_id, sell.agent_id, quantity, trade_price, buyer_fee, seller_fee))
                step_trade_prices.extend([trade_price] * quantity)
                buy.quantity -= quantity
                sell.quantity -= quantity
                if buy.quantity == 0:
                    buys.pop(buy_index)
                if sell.quantity == 0:
                    sells.pop(sell_index)
            if step_trade_prices:
                auction_price = statistics.fmean(step_trade_prices)
            elif buys and sells:
                auction_price = (buys[0].limit_price + sells[0].limit_price) / 2.0
            else:
                # A no-trade market retains its last price before any V2 latent
                # fundamental, event, and order-flow adjustments.
                auction_price = prices[-1]
            if latent is None:
                price = auction_price
            else:
                impact_decay = params.get("impact_decay", 0.55)
                orderflow_impact = params.get("orderflow_impact", 0.004)
                persistent_orderflow_impact = (
                    impact_decay * persistent_orderflow_impact + orderflow_impact * order_imbalance
                )
                coupling = params.get("fundamental_price_coupling", 0.16)
                fundamental_pull = coupling * math.log(fundamental / max(auction_price, 1e-12))
                micro_noise = (
                    params.get("price_volatility", 0.007)
                    * latent.volatility_multipliers[step]
                    * 0.30
                    * market_rng.gauss(0.0, 1.0)
                )
                log_adjustment = (
                    fundamental_pull
                    + persistent_orderflow_impact
                    + latent.transitory_event_shocks[step]
                    + micro_noise
                )
                price = max(1.0, auction_price * math.exp(min(0.18, max(-0.18, log_adjustment))))
                absolute_order_imbalance += abs(order_imbalance)
                absolute_orderflow_impact += abs(persistent_orderflow_impact)
            wealth = self._marked_wealth(focus, price)
            peak_wealth = max(peak_wealth, wealth)
            if peak_wealth > 0:
                max_drawdown = max(max_drawdown, 1.0 - wealth / peak_wealth)
        closing_prices = prices + [price]
        closing_fundamentals = fundamentals + [fundamentals[-1]]
        final_wealth = self._liquidation_wealth(focus, fundamentals[-1], spread_bps, liquidity, fee_bps)
        score, return_pct = self._score(initial_wealth, final_wealth, max_drawdown, contract, violations)
        signature = feature_signature(closing_prices, closing_fundamentals, spread_bps, liquidity)
        final_cash_total = sum(account.cash for account in accounts.values())
        final_asset_total = sum(account.position for account in accounts.values())
        diagnostics = {
            "track_kind": scenario.market_source.kind,
            "trade_count": float(len(trades)),
            "final_price": price,
            "final_fundamental": fundamentals[-1],
            "absolute_price_gap": abs(prices[-1] / fundamentals[-1] - 1.0),
            "cash_conservation_error": final_cash_total + fee_sink - initial_cash_total,
            "asset_conservation_error": float(final_asset_total - initial_asset_total),
            "fee_sink": fee_sink,
        }
        if latent is not None:
            diagnostics.update(self._layered_diagnostics(latent, closing_prices))
            diagnostics.update(
                {
                    "mean_abs_order_imbalance": absolute_order_imbalance / max(1, scenario.horizon),
                    "mean_abs_orderflow_impact": absolute_orderflow_impact / max(1, scenario.horizon),
                }
            )
        return EpisodeResult(
            episode_id=scenario.episode_id,
            family_id=scenario.family_id,
            role=scenario.role,
            layer=scenario.layer,
            strategy_mode=agent.current_mode,
            initial_wealth=initial_wealth,
            final_wealth=final_wealth,
            pnl=final_wealth - initial_wealth,
            return_pct=return_pct,
            score=score,
            max_drawdown=max_drawdown,
            turnover=focus.turnover_notional / initial_wealth,
            violations=violations,
            trades=trades,
            prices=closing_prices,
            fundamentals=closing_fundamentals,
            feature_signature=signature,
            fee_paid=focus.fee_paid,
            market_diagnostics=diagnostics,
            agent_trace=agent.get_trace(),
        )
