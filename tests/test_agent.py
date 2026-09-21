import json
import math
import os
import unittest
from unittest.mock import patch

from evolens.cli import _resolve_state_off_conditions
from evolens.agent import (
    ModelConfig,
    ModelCompletionClient,
    ModelCircuitOpenError,
    ModelRequestError,
    ModelTradingAgent,
    _extract_json,
    call_model,
    estimate_model_calls,
)
from evolens.models import (
    CheckpointSpec,
    EpisodeResult,
    Observation,
    RiskContract,
    ScenarioSpec,
    StreamEvent,
    StreamSpec,
)


def _scenario(episode_id: str = "train-001") -> ScenarioSpec:
    return ScenarioSpec(
        episode_id=episode_id,
        family_id="trend",
        role="learn_near",
        layer="exogenous",
        variant="base",
        split="train",
        hidden=False,
        horizon=8,
        initial_price=100.0,
        mechanism_params={"drift": 0.001},
        opponent_mix={},
        risk_contract=RiskContract(),
        observation_mapping_seed=7,
        environment_seed=11,
    )


def _observation(episode_id: str = "train-001") -> Observation:
    return Observation(
        episode_id=episode_id,
        layer="exogenous",
        step=3,
        horizon=8,
        asset_symbol="SYNTH",
        price=103.0,
        fundamental=101.0,
        public_signal=101.5,
        spread_bps=8.0,
        liquidity=1.0,
        prices=[100.0, 101.0, 102.0, 103.0],
        fundamentals=[100.0, 100.3, 100.7, 101.0],
        cash=8_000.0,
        position=20,
        marked_wealth=10_060.0,
        initial_wealth=10_000.0,
        max_drawdown_so_far=0.01,
        turnover_so_far=0.2,
        risk_contract=RiskContract(),
    )


def _result(episode_id: str = "train-001") -> EpisodeResult:
    return EpisodeResult(
        episode_id=episode_id,
        family_id="trend",
        role="learn_near",
        layer="exogenous",
        strategy_mode="remote",
        initial_wealth=10_000.0,
        final_wealth=10_100.0,
        pnl=100.0,
        return_pct=0.01,
        score=0.02,
        max_drawdown=0.015,
        turnover=0.4,
        violations=[],
        trades=[],
        prices=[100.0, 101.0, 102.0, 103.0],
        fundamentals=[100.0, 100.3, 100.7, 101.0],
        feature_signature="directional:liquid",
        fee_paid=0.2,
        market_diagnostics={},
    )


class FakeCompletion:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, messages, config, max_tokens, phase=None):
        self.calls.append(
            {
                "messages": messages,
                "config": config,
                "max_tokens": max_tokens,
                "phase": phase,
            }
        )
        if phase == "reflection":
            content = json.dumps(
                {
                    "attribution": "The trend rule produced a positive risk-adjusted outcome.",
                    "action": "upsert",
                    "policy_label": "momentum",
                    "hypothesis": "Follow persistent public price trends in a liquid regime.",
                    "invalidate_rule_ids": [],
                }
            )
        else:
            content = "```json\n" + json.dumps(
                {
                    "orders": [
                        {"side": "buy", "quantity": 3, "limit_price": None, "tag": "remote"}
                    ],
                    "decision_summary": "Small trend-following position increase.",
                    "used_rule_ids": [],
                }
            ) + "\n```"
        return {
            "content": content,
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            "request_id": "offline-fake-request",
            "elapsed_seconds": 0.01,
        }


class ModelAgentTests(unittest.TestCase):
    def _agent(self, condition: str, completion=None) -> ModelTradingAgent:
        return ModelTradingAgent(
            condition=condition,
            seed=17,
            config=ModelConfig(model="offline-test", decision_interval=1),
            completion_fn=completion or FakeCompletion(),
        )

    def test_condition_registry_uses_canonical_academic_names(self):
        self.assertIn("baseline", ModelTradingAgent.CONDITIONS)
        self.assertIn("context", ModelTradingAgent.CONDITIONS)
        self.assertIn("reflection", ModelTradingAgent.CONDITIONS)
        self.assertIn("episodic_memory", ModelTradingAgent.CONDITIONS)
        self.assertIn("consolidated_memory", ModelTradingAgent.CONDITIONS)
        for retired in (
            "no_evolve",
            "sham",
            "sliding_window",
            "raw_history",
            "raw_replay",
            "structured",
            "gated",
        ):
            with self.assertRaises(ValueError):
                ModelTradingAgent(retired, config=ModelConfig(model="offline"))
        self.assertFalse(
            ModelTradingAgent(
                "baseline", config=ModelConfig(model="offline")
            ).supports_state_ablation
        )

    def test_state_off_defaults_follow_selected_stateful_conditions(self):
        selected = ["baseline", "reflection", "skillx"]
        self.assertEqual(
            _resolve_state_off_conditions(selected, None, False),
            ["reflection", "skillx"],
        )
        with self.assertRaises(ValueError):
            _resolve_state_off_conditions(selected, "episodic_memory", False)

    def test_extract_json_accepts_fences_and_surrounding_text(self):
        fenced = "```json\n{\"orders\": [], \"decision_summary\": \"ok\"}\n```"
        self.assertEqual(_extract_json(fenced)["orders"], [])

        surrounded = 'Model preface {"action":"none","invalidate_rule_ids":[]} trailing text'
        self.assertEqual(_extract_json(surrounded)["action"], "none")

        skill_with_inner_fence = json.dumps(
            {"skill_document": "Use this example:\n```python\npass\n```", "rationale": "ok"}
        )
        self.assertEqual(_extract_json(skill_with_inner_fence)["rationale"], "ok")

        with self.assertRaises(ValueError):
            _extract_json("there is no JSON object here")

    def test_episodic_memory_reflection_commits_auditable_strategy(self):
        completion = FakeCompletion()
        agent = self._agent("episodic_memory", completion)
        agent.begin_episode(_scenario(), execution_seed=101)

        orders = agent.act(_observation())
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].side, "buy")
        self.assertEqual(orders[0].quantity, 3)

        agent.end_episode(_result(), allow_state_update=True, expose_feedback=True)
        snapshot = agent.state_snapshot()
        active = [item for item in snapshot["store"]["entries"] if item["status"] == "active"]
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["signature"], "directional:liquid")
        self.assertEqual(active[0]["mode"], "momentum")
        self.assertEqual(active[0]["evidence_episode_ids"], ["train-001"])
        self.assertIn("persistent public price trends", active[0]["hypothesis"])
        self.assertEqual([call["phase"] for call in completion.calls], ["decision", "reflection"])

        events = agent.get_trace()
        self.assertTrue(any(event.get("event") == "write" and event.get("committed") for event in events))
        self.assertEqual(
            len([event for event in events if event.get("event") == "model_call"]),
            2,
        )

    def test_context_injects_bounded_raw_episode_history_without_reflection(self):
        completion = FakeCompletion()
        agent = ModelTradingAgent(
            condition="context",
            seed=17,
            config=ModelConfig(
                model="offline-test",
                decision_interval=1,
                context_episodes=2,
                context_steps=1,
            ),
            completion_fn=completion,
        )

        agent.begin_episode(_scenario("hidden-train-id-1"), execution_seed=1)
        agent.act(_observation("hidden-train-id-1"))
        first_hash = agent.state_hash()
        agent.end_episode(
            _result("hidden-train-id-1"),
            allow_state_update=True,
            expose_feedback=True,
        )
        self.assertNotEqual(agent.state_hash(), first_hash)
        self.assertEqual([call["phase"] for call in completion.calls], ["decision"])

        agent.begin_episode(_scenario("hidden-train-id-2"), execution_seed=2)
        agent.act(_observation("hidden-train-id-2"))
        second_prompt = json.loads(completion.calls[-1]["messages"][1]["content"])
        self.assertEqual(len(second_prompt["recent_episode_history"]), 1)
        self.assertNotIn("hidden-train-id-1", json.dumps(second_prompt, ensure_ascii=False))
        model_call = next(
            event for event in agent.get_trace() if event.get("event") == "model_call"
        )
        self.assertEqual(
            model_call["audit_context"]["token_attribution"],
            "context_conditioned_decision",
        )
        agent.end_episode(
            _result("hidden-train-id-2"),
            allow_state_update=True,
            expose_feedback=True,
        )

        agent.begin_episode(_scenario("hidden-train-id-3"), execution_seed=3)
        agent.act(_observation("hidden-train-id-3"))
        agent.end_episode(
            _result("hidden-train-id-3"),
            allow_state_update=True,
            expose_feedback=True,
        )
        snapshot = agent.state_snapshot()
        self.assertEqual(len(snapshot["context"]), 2)
        self.assertTrue(
            all(len(item["trajectory"]) <= 1 for item in snapshot["context"])
        )

    def test_context_state_off_hides_history(self):
        completion = FakeCompletion()
        agent = ModelTradingAgent(
            condition="context",
            seed=17,
            config=ModelConfig(model="offline-test", decision_interval=1),
            completion_fn=completion,
        )
        agent.context = [
            {
                "trajectory": [{"decision_summary": "HISTORY_SENTINEL"}],
                "outcome": {"score": 0.1},
            }
        ]
        agent.set_state_access(False)
        agent.begin_episode(_scenario("probe-hidden-id"), execution_seed=4)
        agent.act(_observation("probe-hidden-id"))
        prompt = json.dumps(completion.calls[-1]["messages"], ensure_ascii=False)
        self.assertNotIn("HISTORY_SENTINEL", prompt)
        retrieval = next(
            event for event in agent.get_trace() if event.get("event") == "retrieve"
        )
        self.assertTrue(retrieval["disabled"])
        self.assertFalse(retrieval["hit"])

    def test_state_off_hides_persistent_memory_from_model(self):
        completion = FakeCompletion()
        agent = self._agent("episodic_memory", completion)
        committed, entry = agent.store.propose(
            "directional:liquid",
            "momentum",
            "prior-episode",
            0.03,
            hypothesis="SECRET_MEMORY_SENTINEL",
        )
        self.assertTrue(committed)
        self.assertIsNotNone(entry)

        agent.set_state_access(False)
        agent.begin_episode(_scenario("probe-001"), execution_seed=202)
        agent.act(_observation("probe-001"))

        prompt_text = json.dumps(completion.calls[0]["messages"], ensure_ascii=False)
        self.assertNotIn("SECRET_MEMORY_SENTINEL", prompt_text)
        self.assertNotIn(entry.rule_id, prompt_text)
        retrieval = next(event for event in agent.get_trace() if event.get("event") == "retrieve")
        self.assertTrue(retrieval["disabled"])
        self.assertFalse(retrieval["hit"])

    def test_hidden_scenario_labels_never_enter_prompts(self):
        completion = FakeCompletion()
        agent = self._agent("episodic_memory", completion)
        scenario = _scenario("episode-secret-sentinel")
        scenario = ScenarioSpec(
            **{
                **scenario.to_dict(),
                "family_id": "FAMILY_SECRET_SENTINEL",
                "role": "ROLE_SECRET_SENTINEL",
                "mechanism_params": {"HIDDEN_PARAM_SENTINEL": 99.0},
                "risk_contract": scenario.risk_contract,
            }
        )
        agent.begin_episode(scenario, execution_seed=999999)
        agent.act(_observation("episode-secret-sentinel"))
        result = _result("episode-secret-sentinel")
        result.family_id = "FAMILY_SECRET_SENTINEL"
        result.role = "ROLE_SECRET_SENTINEL"
        agent.end_episode(result, allow_state_update=True, expose_feedback=True)

        prompt_text = json.dumps(
            [call["messages"] for call in completion.calls],
            ensure_ascii=False,
        )
        for sentinel in (
            "FAMILY_SECRET_SENTINEL",
            "ROLE_SECRET_SENTINEL",
            "HIDDEN_PARAM_SENTINEL",
            "episode-secret-sentinel",
            "999999",
        ):
            self.assertNotIn(sentinel, prompt_text)

    def test_empty_state_on_and_off_send_identical_decision_prompt(self):
        on_completion = FakeCompletion()
        on_agent = self._agent("episodic_memory", on_completion)
        on_agent.begin_episode(_scenario(), execution_seed=1)
        on_agent.act(_observation())

        off_completion = FakeCompletion()
        off_agent = self._agent("episodic_memory", off_completion)
        off_agent.set_state_access(False)
        off_agent.begin_episode(_scenario(), execution_seed=1)
        off_agent.act(_observation())

        self.assertEqual(on_completion.calls[0]["messages"], off_completion.calls[0]["messages"])

    def test_baseline_and_empty_context_send_identical_decision_prompt(self):
        baseline_completion = FakeCompletion()
        baseline = self._agent("baseline", baseline_completion)
        baseline.begin_episode(_scenario(), execution_seed=1)
        baseline.act(_observation())

        history_completion = FakeCompletion()
        history = self._agent("context", history_completion)
        history.begin_episode(_scenario(), execution_seed=1)
        history.act(_observation())

        self.assertEqual(
            baseline_completion.calls[0]["messages"],
            history_completion.calls[0]["messages"],
        )

    def test_environment_key_is_redacted_from_error_trace(self):
        secret = "unit-test-openai_compatible-secret-value"

        def failing_completion(messages, config, max_tokens, phase=None):
            raise RuntimeError(f"remote failure Authorization: Bearer {secret}")

        with patch.dict(os.environ, {"EVOPATHBENCH_API_KEY": secret}, clear=False):
            agent = self._agent("episodic_memory", failing_completion)
            agent.begin_episode(_scenario(), execution_seed=303)
            orders = agent.act(_observation())
            serialized = json.dumps(
                {"trace": agent.get_trace(), "state": agent.state_snapshot()},
                ensure_ascii=False,
            )

        self.assertEqual(orders, [])
        self.assertNotIn(secret, serialized)
        model_call = next(event for event in agent.get_trace() if event.get("event") == "model_call")
        self.assertFalse(model_call["ok"])

    def test_call_estimate_includes_probes_state_off_and_reflection(self):
        scenario = _scenario()
        stream = StreamSpec(
            stream_id="cost-test",
            template="accumulation",
            layer="exogenous",
            focal_family="trend",
            events=[StreamEvent(scenario.episode_id, True, True)],
            checkpoints=[
                CheckpointSpec("K0", 0, "before", [scenario.episode_id]),
                CheckpointSpec("K1", 1, "after", [scenario.episode_id]),
            ],
            stream_seed=1,
        )
        estimate = estimate_model_calls(
            [scenario],
            [stream],
            ["episodic_memory"],
            ModelConfig(model="offline", decision_interval=2),
            campaigns=3,
            repeats=2,
            state_off_conditions=["episodic_memory"],
        )
        self.assertEqual(estimate["decision_calls"], 108)
        self.assertEqual(estimate["reflection_calls"], 3)
        self.assertEqual(estimate["logical_calls"], 111)

    def test_call_estimate_skips_reflection_for_context(self):
        scenario = _scenario()
        stream = StreamSpec(
            stream_id="context-cost-test",
            template="accumulation",
            layer="exogenous",
            focal_family="trend",
            events=[StreamEvent(scenario.episode_id, True, True)],
            checkpoints=[],
            stream_seed=1,
        )
        estimate = estimate_model_calls(
            [scenario],
            [stream],
            ["context"],
            ModelConfig(model="offline", decision_interval=2),
            campaigns=2,
            repeats=1,
            state_off_conditions=[],
        )
        self.assertEqual(estimate["decision_calls"], 8)
        self.assertEqual(estimate["reflection_calls"], 0)

    def test_call_estimate_accounts_for_new_skill_pipelines_without_validation(self):
        scenario = _scenario()
        stream = StreamSpec(
            stream_id="new-skill-cost-test",
            template="accumulation",
            layer="exogenous",
            focal_family="trend",
            events=[StreamEvent(scenario.episode_id, True, True)],
            checkpoints=[],
            stream_seed=1,
        )
        config = ModelConfig(
            model="offline", decision_interval=2, trace2skill_analyst_count=2
        )
        estimate = estimate_model_calls(
            [scenario],
            [stream],
            ["skillx", "trace2skill", "skillgrad"],
            config,
            campaigns=1,
            repeats=1,
            state_off_conditions=[],
        )
        episode_decisions = math.ceil(scenario.horizon / config.decision_interval)
        self.assertEqual(estimate["by_condition"]["skillx"]["skill_optimizer_calls"], 2)
        self.assertEqual(
            estimate["by_condition"]["trace2skill"]["skill_optimizer_calls"], 3
        )
        self.assertEqual(estimate["by_condition"]["skillgrad"]["skill_optimizer_calls"], 3)
        self.assertEqual(estimate["candidate_validation_decision_calls"], 0)
        self.assertEqual(estimate["reflection_calls"], 0)
        self.assertEqual(estimate["logical_calls"], episode_decisions * 3 + 8)

    def test_paid_malformed_response_keeps_usage_and_request_id(self):
        def malformed(messages, config, max_tokens, phase=None):
            return {
                "content": "not-json",
                "usage": {"prompt_tokens": 42, "completion_tokens": 13},
                "request_id": "paid-malformed-request",
                "elapsed_seconds": 0.2,
                "attempts": [{"attempt": 1, "http_status": 200}],
            }

        agent = self._agent("episodic_memory", malformed)
        agent.begin_episode(_scenario(), execution_seed=1)
        self.assertEqual(agent.act(_observation()), [])
        event = next(item for item in agent.get_trace() if item.get("event") == "model_call")
        self.assertFalse(event["ok"])
        self.assertTrue(event["transport_ok"])
        self.assertFalse(event["parse_ok"])
        self.assertEqual(event["request_id"], "paid-malformed-request")
        self.assertEqual(event["usage"]["input_tokens"], 42)
        self.assertEqual(event["usage"]["output_tokens"], 13)
        self.assertEqual(event["attempts"][0]["http_status"], 200)

    def test_invalid_reflection_taxonomy_never_updates_state(self):
        class InvalidReflection(FakeCompletion):
            def __call__(self, messages, config, max_tokens, phase=None):
                value = super().__call__(messages, config, max_tokens, phase=phase)
                if phase == "reflection":
                    value["content"] = json.dumps(
                        {
                            "action": "upsert",
                            "policy_label": "invented-unregistered-policy",
                            "hypothesis": "should not persist",
                            "invalidate_rule_ids": [],
                        }
                    )
                return value

        agent = self._agent("episodic_memory", InvalidReflection())
        agent.begin_episode(_scenario(), execution_seed=1)
        agent.act(_observation())
        agent.end_episode(_result(), allow_state_update=True, expose_feedback=True)
        self.assertEqual(agent.state_snapshot()["store"]["entries"], [])
        self.assertTrue(
            any(
                event.get("event") == "parse_error" and event.get("phase") == "reflection"
                for event in agent.get_trace()
            )
        )

    def test_non_retryable_http_4xx_is_fatal(self):
        import httpx

        class Client:
            calls = 0

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def post(self, url, json=None, headers=None):
                Client.calls += 1
                return httpx.Response(400, request=httpx.Request("POST", url))

        with patch.dict(os.environ, {"EVOPATHBENCH_API_KEY": "fake-key"}, clear=False):
            with patch("httpx.Client", Client):
                with self.assertRaises(ModelRequestError):
                    call_model(
                        [{"role": "user", "content": "json"}],
                        ModelConfig(model="bad-model", max_retries=3),
                        10,
                    )
        self.assertEqual(Client.calls, 1)

    def test_shared_completion_client_reuses_pool_and_closes_once(self):
        import httpx

        class Client:
            instances = 0
            calls = 0
            closes = 0

            def __init__(self, *args, **kwargs):
                Client.instances += 1

            def post(self, url, json=None, headers=None):
                Client.calls += 1
                return httpx.Response(
                    200,
                    request=httpx.Request("POST", url),
                    json={
                        "id": f"request-{Client.calls}",
                        "model": "offline",
                        "choices": [
                            {
                                "message": {"content": "{}"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                    },
                )

            def close(self):
                Client.closes += 1

        config = ModelConfig(model="offline", max_retries=0)
        with patch.dict(os.environ, {"EVOPATHBENCH_API_KEY": "fake-key"}, clear=False):
            with patch("httpx.Client", Client):
                completion = ModelCompletionClient(config, max_connections=20)
                first = completion([], config, 10)
                second = completion([], config, 10)
                self.assertIs(completion.__deepcopy__({}), completion)
                completion.close()
                completion.close()
        self.assertEqual(Client.instances, 1)
        self.assertEqual(Client.calls, 2)
        self.assertEqual(Client.closes, 1)
        self.assertEqual(first["request_id"], "request-1")
        self.assertEqual(second["request_id"], "request-2")

    def test_consecutive_failure_circuit_breaker_stops_run(self):
        def always_fails(messages, config, max_tokens, phase=None):
            raise RuntimeError("temporary provider failure")

        agent = ModelTradingAgent(
            "episodic_memory",
            config=ModelConfig(
                model="offline",
                decision_interval=1,
                max_consecutive_failures=2,
            ),
            completion_fn=always_fails,
        )
        agent.begin_episode(_scenario(), execution_seed=1)
        self.assertEqual(agent.act(_observation()), [])
        second = _observation()
        object.__setattr__(second, "step", 4)
        with self.assertRaises(ModelCircuitOpenError):
            agent.act(second)

    def test_api_seed_is_derived_per_execution_and_step(self):
        completion = FakeCompletion()
        agent = ModelTradingAgent(
            "episodic_memory",
            config=ModelConfig(model="offline", api_seed=123),
            completion_fn=completion,
        )
        agent.begin_episode(_scenario(), execution_seed=456)
        agent.act(_observation())
        first_seed = completion.calls[0]["config"].api_seed

        completion_2 = FakeCompletion()
        agent_2 = ModelTradingAgent(
            "episodic_memory",
            config=ModelConfig(model="offline", api_seed=123),
            completion_fn=completion_2,
        )
        agent_2.begin_episode(_scenario(), execution_seed=457)
        agent_2.act(_observation())
        second_seed = completion_2.calls[0]["config"].api_seed
        self.assertNotEqual(first_seed, second_seed)
        self.assertNotEqual(first_seed, 123)


if __name__ == "__main__":
    unittest.main()
