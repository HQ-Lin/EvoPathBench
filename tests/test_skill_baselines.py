import json
import unittest

from evolens.agent import ModelConfig, ModelTradingAgent
from evolens.models import EpisodeResult, RiskContract, ScenarioSpec, StreamEvent, StreamSpec
from evolens.runner import BenchmarkRunner
from evolens.skill_baselines import apply_skillopt_edits, select_skillboost_candidate
from evolens.self_evolution_methods import (
    render_skillgrad_package,
    render_skillx_library,
    sanitize_skillgrad_package,
    sanitize_skillx_library,
    sanitize_trace2skill_proposal,
)


def scenario() -> ScenarioSpec:
    return ScenarioSpec(
        episode_id="learning-id-must-stay-private",
        family_id="trend",
        role="learn_near",
        layer="exogenous",
        variant="base",
        split="train",
        hidden=False,
        horizon=4,
        initial_price=100.0,
        mechanism_params={"drift": 0.001},
        opponent_mix={},
        risk_contract=RiskContract(),
        observation_mapping_seed=1,
        environment_seed=2,
    )


def episode_result() -> EpisodeResult:
    return EpisodeResult(
        episode_id="learning-id-must-stay-private",
        family_id="trend",
        role="learn_near",
        layer="exogenous",
        strategy_mode="openai_compatible",
        initial_wealth=10_000.0,
        final_wealth=10_010.0,
        pnl=10.0,
        return_pct=0.001,
        score=0.01,
        max_drawdown=0.002,
        turnover=0.1,
        violations=[],
        trades=[],
        prices=[100.0, 101.0],
        fundamentals=[100.0, 100.5],
        feature_signature="directional:liquid",
        fee_paid=0.1,
        market_diagnostics={},
    )


class SkillCompletion:
    def __init__(self) -> None:
        self.calls = []
        self.generated = 0

    def __call__(self, messages, config, max_tokens, phase=None):
        self.calls.append({"phase": phase, "messages": messages})
        if phase == "skillopt_optimize":
            payload = {
                "diagnosis": "Use smaller, risk-aware orders.",
                "edits": [
                    {
                        "op": "append",
                        "target": "",
                        "content": "SKILLOPT_SENTINEL: prefer small orders under drawdown pressure.",
                    }
                ],
            }
        elif phase == "skillboost_diagnose":
            payload = {
                "earliest_causal_deviation": "position increased too early",
                "root_causes": ["weak confirmation"],
                "protected_behaviors": ["obey position limits"],
            }
        elif phase == "skillboost_generate":
            payload = {
                "skill_document": f"SKILLBOOST_CANDIDATE_{self.generated}",
                "rationale": "distinct repair prior",
            }
            self.generated += 1
        else:
            payload = {"orders": [], "decision_summary": "hold", "used_rule_ids": []}
        return {
            "content": json.dumps(payload),
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            "request_id": f"fake-{len(self.calls)}",
        }


class NewMethodCompletion:
    def __init__(self) -> None:
        self.calls = []
        self.skillgrad_patch_count = 0

    def __call__(self, messages, config, max_tokens, phase=None):
        self.calls.append({"phase": phase, "messages": messages})
        if phase == "skillx_extract":
            payload = {
                "proposals": {
                    "planning": [{"operation": "add", "name": "Control risk first"}],
                    "functional": [],
                    "atomic": [],
                }
            }
        elif phase == "skillx_consolidate":
            payload = {
                "library": {
                    "planning": [
                        {
                            "name": "Control risk first",
                            "content": "Reduce exposure when drawdown rises.",
                            "activation_signals": ["drawdown increases"],
                            "tools": ["risk contract"],
                            "source_count": 1,
                        }
                    ],
                    "functional": [
                        {
                            "name": "Confirm before entry",
                            "content": "Require price and public signal agreement.",
                            "activation_signals": ["new entry"],
                            "tools": [],
                            "source_count": 1,
                        }
                    ],
                    "atomic": [],
                },
                "merge_log": ["added two grounded skills"],
                "filtered_items": [],
            }
        elif phase == "trace2skill_analyze":
            payload = {
                "analysis_label": "mixed",
                "evidence": ["drawdown rose"],
                "protected_behaviors": ["position limit"],
                "patch": [
                    {
                        "operation": "add_section",
                        "target": "Risk",
                        "content": "Scale down under drawdown.",
                        "rationale": "observable risk response",
                    }
                ],
            }
        elif phase == "trace2skill_consolidate":
            payload = {
                "skill_document": "# Risk\nScale down under drawdown.",
                "resolved_conflicts": ["merged duplicate risk patches"],
                "applied_patch_ids": ["0/0"],
                "rejected_patch_ids": ["1/0 duplicate"],
            }
        elif phase == "skillgrad_diagnose":
            payload = {
                "label": "mixed",
                "signal": "drawdown rose",
                "causal_mechanism": "entry was too large",
                "robust_action": "scale entry by risk budget",
                "skipped_reasoning_step": "risk sizing",
            }
        elif phase == "skillgrad_momentum":
            prior = json.loads(messages[-1]["content"])["persistent_patterns"]
            appeared = prior[0]["appeared_in"] + 1 if prior else 1
            payload = {
                "patterns": [
                    {
                        "pattern_id": "risk-sizing",
                        "kind": "failure",
                        "anchor": "L2 risk sizing",
                        "appeared_in": appeared,
                        "description": "Large entries amplify drawdown.",
                        "latest_executor_action": "entered too large",
                        "remedy_log": ["scale by risk budget"],
                    }
                ],
                "overlay": {
                    "signal": "drawdown rose",
                    "pattern": "risk-sizing",
                    "anchor": "L2",
                    "gap": "missing sizing rule",
                    "proposed_change": "add risk sizing",
                },
            }
        elif phase == "skillgrad_patch":
            self.skillgrad_patch_count += 1
            payload = {
                "routing": {
                    "name": "Market risk control",
                    "description": "Use when choosing order size.",
                    "activation_signals": ["before order placement"],
                },
                "body": f"Scale order size by remaining risk budget. v{self.skillgrad_patch_count}",
                "references": [
                    {
                        "name": "Drawdown response",
                        "when_to_load": "drawdown is positive",
                        "content": "Cut exposure as drawdown increases.",
                    }
                ],
                "applied_pattern_ids": ["risk-sizing"],
                "rationale": "general rule belongs in L2",
            }
        else:
            payload = {"orders": [], "decision_summary": "hold", "used_rule_ids": []}
        return {
            "content": json.dumps(payload),
            "usage": {"prompt_tokens": 11, "completion_tokens": 5},
            "request_id": f"new-method-{len(self.calls)}",
        }


class SkillBaselineTests(unittest.TestCase):
    def test_skillx_schema_is_bounded_and_rendered_by_level(self):
        library, rejected = sanitize_skillx_library(
            {
                "planning": [
                    {"name": "Risk", "content": "Cap exposure."},
                    {"name": "Risk", "content": "duplicate"},
                ],
                "functional": [],
                "atomic": [],
            },
            max_items_per_level=2,
            max_content_chars=100,
        )
        self.assertEqual(len(library["planning"]), 1)
        self.assertEqual(rejected[-1]["reason"], "duplicate_name")
        self.assertIn("Planning skills", render_skillx_library(library, 1000))

    def test_skillgrad_schema_has_three_explicit_layers(self):
        package, rejected = sanitize_skillgrad_package(
            {
                "routing": {"name": "Risk", "description": "route", "activation_signals": []},
                "body": "General rule",
                "references": [
                    {"name": "Edge", "when_to_load": "stress", "content": "Reduce."}
                ],
            },
            max_references=2,
            max_content_chars=200,
        )
        self.assertEqual(rejected, [])
        rendered = render_skillgrad_package(package, 1000)
        self.assertIn("L1 routing", rendered)
        self.assertIn("L2 general", rendered)
        self.assertIn("L3 reference", rendered)

    def test_trace2skill_rejects_unsupported_patch_operations(self):
        proposal, rejected = sanitize_trace2skill_proposal(
            {
                "analysis_label": "failure",
                "patch": [
                    {"operation": "shell_exec", "target": "Risk", "content": "bad"},
                    {
                        "operation": "append_to_section",
                        "target": "Risk",
                        "content": "Add a limit.",
                    },
                ],
            }
        )
        self.assertEqual(len(proposal["patch"]), 1)
        self.assertEqual(rejected[0]["reason"], "unsupported_patch_operation")

    def test_skillx_uses_openai_compatible_extraction_and_hierarchical_consolidation(self):
        completion = NewMethodCompletion()
        agent = ModelTradingAgent(
            "skillx", config=ModelConfig(model="offline"), completion_fn=completion
        )
        agent.begin_episode(scenario(), execution_seed=20)
        agent.end_episode(episode_result(), True, True)
        snapshot = agent.state_snapshot()["skill_state"]
        self.assertEqual(
            [call["phase"] for call in completion.calls],
            ["skillx_extract", "skillx_consolidate"],
        )
        self.assertEqual(snapshot["version"], 1)
        self.assertEqual(len(snapshot["method_state"]["library"]["planning"]), 1)
        self.assertIn("Control risk first", snapshot["document"])

    def test_trace2skill_analysts_share_frozen_incumbent_then_reduce(self):
        completion = NewMethodCompletion()
        agent = ModelTradingAgent(
            "trace2skill",
            config=ModelConfig(model="offline", trace2skill_analyst_count=2),
            completion_fn=completion,
        )
        agent.skill_document = "# Existing\nKeep position limits."
        agent.begin_episode(scenario(), execution_seed=21)
        agent.end_episode(episode_result(), True, True)
        self.assertEqual(
            [call["phase"] for call in completion.calls],
            ["trace2skill_analyze", "trace2skill_analyze", "trace2skill_consolidate"],
        )
        analyst_payloads = [
            json.loads(call["messages"][-1]["content"])
            for call in completion.calls[:2]
        ]
        self.assertEqual(
            analyst_payloads[0]["frozen_incumbent_skill"],
            analyst_payloads[1]["frozen_incumbent_skill"],
        )
        trace = [event for event in agent.get_trace() if event.get("event") == "model_call"]
        hashes = [event["audit_context"].get("frozen_incumbent_hash") for event in trace[:2]]
        self.assertEqual(hashes[0], hashes[1])
        self.assertEqual(agent.state_snapshot()["skill_state"]["version"], 1)

    def test_skillgrad_persists_momentum_and_layered_package(self):
        completion = NewMethodCompletion()
        agent = ModelTradingAgent(
            "skillgrad", config=ModelConfig(model="offline"), completion_fn=completion
        )
        for seed in (30, 31):
            agent.begin_episode(scenario(), execution_seed=seed)
            agent.end_episode(episode_result(), True, True)
        snapshot = agent.state_snapshot()["skill_state"]
        self.assertEqual(
            [call["phase"] for call in completion.calls],
            [
                "skillgrad_diagnose",
                "skillgrad_momentum",
                "skillgrad_patch",
                "skillgrad_diagnose",
                "skillgrad_momentum",
                "skillgrad_patch",
            ],
        )
        self.assertEqual(snapshot["version"], 2)
        self.assertEqual(snapshot["method_state"]["momentum_patterns"][0]["appeared_in"], 2)
        self.assertIn("L3 reference", snapshot["document"])

    def test_new_skill_methods_never_prompt_hidden_task_identifiers(self):
        for condition in ("skillx", "trace2skill", "skillgrad"):
            completion = NewMethodCompletion()
            agent = ModelTradingAgent(
                condition, config=ModelConfig(model="offline"), completion_fn=completion
            )
            agent.begin_episode(scenario(), execution_seed=40)
            agent.end_episode(episode_result(), True, True)
            prompts = json.dumps(completion.calls, ensure_ascii=False)
            self.assertNotIn("learning-id-must-stay-private", prompts)

    def test_new_skill_methods_support_state_off_ablation(self):
        for condition in ("skillx", "trace2skill", "skillgrad"):
            completion = NewMethodCompletion()
            agent = ModelTradingAgent(
                condition, config=ModelConfig(model="offline"), completion_fn=completion
            )
            agent.skill_document = "PRIVATE_SKILL_STATE_SENTINEL"
            agent.skill_version = 1
            agent.set_state_access(False)
            agent.begin_episode(scenario(), execution_seed=41)
            from evolens.market import MarketSimulator

            # Running a full short episode exercises retrieval and every decision prompt.
            MarketSimulator().run(scenario(), agent, execution_seed=41)
            prompts = json.dumps(completion.calls, ensure_ascii=False)
            self.assertNotIn("PRIVATE_SKILL_STATE_SENTINEL", prompts)
            retrieval = next(
                item for item in agent.get_trace() if item.get("event") == "retrieve"
            )
            self.assertTrue(retrieval["disabled"])
            self.assertFalse(retrieval["hit"])
    def test_skillopt_edits_are_bounded_and_exact(self):
        document, applied, rejected = apply_skillopt_edits(
            "Alpha\nBeta",
            [
                {"op": "replace", "target": "Beta", "content": "Gamma"},
                {"op": "delete", "target": "missing", "content": ""},
                {"op": "append", "target": "", "content": "Ignored by budget"},
            ],
            edit_budget=2,
            max_chars=1000,
        )
        self.assertEqual(document, "Alpha\nGamma")
        self.assertEqual(len(applied), 1)
        self.assertEqual(rejected[0]["reason"], "target_must_match_once")

    def test_skillboost_selector_enforces_case_regression_cap(self):
        incumbent = {
            "mean_score": 0.0,
            "cases": [{"case_id": "a", "score": 0.0}, {"case_id": "b", "score": 0.0}],
        }
        winner, assessments = select_skillboost_candidate(
            incumbent,
            [
                (
                    "safe",
                    {
                        "mean_score": 0.5,
                        "cases": [
                            {"case_id": "a", "score": 0.5},
                            {"case_id": "b", "score": 0.5},
                        ],
                    },
                ),
                (
                    "risky",
                    {
                        "mean_score": 1.0,
                        "cases": [
                            {"case_id": "a", "score": 3.0},
                            {"case_id": "b", "score": -1.0},
                        ],
                    },
                ),
            ],
            max_case_regression=0.25,
        )
        self.assertEqual(winner, "safe")
        risky = next(row for row in assessments if row["candidate_id"] == "risky")
        self.assertFalse(risky["eligible"])
        self.assertIn("case_regression_cap_exceeded", risky["reasons"])

    def test_skillboost_selector_protects_old_family_slice(self):
        incumbent = {
            "mean_score": 0.0,
            "cases": [
                {"case_id": "old", "slice_id": "old-family", "score": 0.0},
                {"case_id": "new", "slice_id": "new-family", "score": 0.0},
            ],
        }
        winner, assessments = select_skillboost_candidate(
            incumbent,
            [
                (
                    "plastic-but-forgetting",
                    {
                        "mean_score": 0.5,
                        "cases": [
                            {"case_id": "old", "slice_id": "old-family", "score": -0.2},
                            {"case_id": "new", "slice_id": "new-family", "score": 1.2},
                        ],
                    },
                )
            ],
            max_case_regression=0.75,
            max_slice_regression=0.0,
        )
        self.assertIsNone(winner)
        self.assertIn("slice_regression_cap_exceeded", assessments[0]["reasons"])

    def test_skillopt_uses_openai_compatible_optimizer_and_strict_validation_gate(self):
        completion = SkillCompletion()
        agent = ModelTradingAgent(
            "skillopt",
            config=ModelConfig(model="offline", skillopt_edit_budget=1),
            completion_fn=completion,
        )

        def validator(candidates):
            self.assertEqual([item[0] for item in candidates], ["incumbent", "candidate_0"])
            return {
                "reports": {
                    "incumbent": {"mean_score": 0.0, "cases": [{"case_id": "x", "score": 0.0}]},
                    "candidate_0": {"mean_score": 0.2, "cases": [{"case_id": "x", "score": 0.2}]},
                },
                "agent_trace": [],
            }

        agent.set_candidate_validator(validator)
        agent.begin_episode(scenario(), execution_seed=10)
        agent.end_episode(episode_result(), True, True)
        snapshot = agent.state_snapshot()["skill_state"]
        self.assertEqual(snapshot["version"], 1)
        self.assertIn("SKILLOPT_SENTINEL", snapshot["document"])
        event = next(item for item in agent.get_trace() if item.get("event") == "model_call")
        self.assertEqual(event["phase"], "skillopt_optimize")
        self.assertEqual(
            event["audit_context"]["token_attribution"], "self_evolution_skill_optimizer"
        )
        prompts = json.dumps(completion.calls, ensure_ascii=False)
        self.assertNotIn("learning-id-must-stay-private", prompts)

    def test_skillboost_generates_multiple_openai_compatible_candidates_and_accepts_safe_one(self):
        completion = SkillCompletion()
        agent = ModelTradingAgent(
            "skillboost",
            config=ModelConfig(
                model="offline",
                skillboost_candidate_count=2,
                skillboost_max_case_regression=0.25,
            ),
            completion_fn=completion,
        )

        def validator(candidates):
            self.assertEqual(len(candidates), 3)
            return {
                "reports": {
                    "incumbent": {
                        "mean_score": 0.0,
                        "cases": [{"case_id": "a", "score": 0.0}, {"case_id": "b", "score": 0.0}],
                    },
                    "candidate_0": {
                        "mean_score": 0.5,
                        "cases": [{"case_id": "a", "score": 0.5}, {"case_id": "b", "score": 0.5}],
                    },
                    "candidate_1": {
                        "mean_score": 1.0,
                        "cases": [{"case_id": "a", "score": 3.0}, {"case_id": "b", "score": -1.0}],
                    },
                },
                "agent_trace": [],
            }

        agent.set_candidate_validator(validator)
        agent.begin_episode(scenario(), execution_seed=11)
        agent.end_episode(episode_result(), True, True)
        snapshot = agent.state_snapshot()["skill_state"]
        self.assertEqual(snapshot["version"], 1)
        self.assertEqual(snapshot["document"], "SKILLBOOST_CANDIDATE_0")
        self.assertEqual(
            [call["phase"] for call in completion.calls],
            ["skillboost_diagnose", "skillboost_generate", "skillboost_generate"],
        )

    def test_runner_uses_internal_paired_validation_without_probe_leakage(self):
        completion = SkillCompletion()
        task = scenario()
        stream = StreamSpec(
            stream_id="private-stream-id",
            template="accumulation",
            layer="exogenous",
            focal_family="trend",
            events=[StreamEvent(task.episode_id, True, True)],
            checkpoints=[],
            stream_seed=3,
        )

        def factory(condition, seed):
            return ModelTradingAgent(
                condition,
                seed=seed,
                config=ModelConfig(model="offline", decision_interval=2),
                completion_fn=completion,
            )

        result = BenchmarkRunner([task], [stream], agent_factory=factory).run(
            ["skillopt"],
            campaigns=1,
            repeats=1,
            state_off_conditions=[],
            max_concurrency=20,
        )
        self.assertEqual(result["evaluation_records"], [])
        training = result["training_records"][0]
        validation_calls = [
            event
            for event in training["agent_trace"]
            if event.get("event") == "model_call"
            and event.get("audit_context", {}).get("token_attribution")
            == "skill_validation_decision"
        ]
        self.assertEqual(len(validation_calls), 4)
        self.assertEqual(training["state_hash_before"], training["state_hash_after"])
        prompt_text = json.dumps(completion.calls, ensure_ascii=False)
        self.assertNotIn("learning-id-must-stay-private", prompt_text)
        self.assertNotIn("private-stream-id", prompt_text)


if __name__ == "__main__":
    unittest.main()
