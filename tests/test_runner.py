import tempfile
import threading
import time
import unittest
from pathlib import Path

from evolens.agents import AdaptiveTradingAgent, BaseTradingAgent
from evolens.dataset import ScenarioGenerator
from evolens.market import MarketSimulator
from evolens.runner import BenchmarkRunner, write_run
from evolens.streams import StreamGenerator


class RunnerTests(unittest.TestCase):
    def test_parallel_map_is_bounded_and_preserves_input_order(self):
        lock = threading.Lock()
        barrier = threading.Barrier(4)
        active = 0
        peak = 0

        def job(value):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            barrier.wait(timeout=2)
            time.sleep(0.005 * (4 - (value % 4)))
            with lock:
                active -= 1
            return value

        output = BenchmarkRunner._parallel_map(job, list(range(8)), 4)
        self.assertEqual(output, list(range(8)))
        self.assertEqual(peak, 4)

    def test_parallel_map_stops_replenishing_after_fatal_error(self):
        lock = threading.Lock()
        started = []

        def job(value):
            with lock:
                started.append(value)
            if value == 1:
                raise RuntimeError("fatal")
            time.sleep(0.02)
            return value

        with self.assertRaisesRegex(RuntimeError, "fatal"):
            BenchmarkRunner._parallel_map(job, list(range(100)), 4)
        self.assertLessEqual(len(started), 4)

    def test_runner_rejects_empty_duplicate_and_unselected_condition_protocols(self):
        scenarios = ScenarioGenerator(seed=2, horizon=6).generate(
            instances_per_role=5, families=["trend"], layers=["exogenous"]
        )
        streams = StreamGenerator(seed=2).generate(
            scenarios, templates=["accumulation"]
        )
        runner = BenchmarkRunner(scenarios, streams)
        with self.assertRaises(ValueError):
            runner.run([], campaigns=1, repeats=1)
        with self.assertRaises(ValueError):
            runner.run(["baseline", "baseline"], campaigns=1, repeats=1)
        with self.assertRaises(ValueError):
            runner.run(
                ["baseline"],
                campaigns=1,
                repeats=1,
                state_off_conditions=["episodic_memory"],
            )
        with self.assertRaises(ValueError):
            runner.run(["baseline"], campaigns=1, repeats=1, max_concurrency=0)
        with self.assertRaises(ValueError):
            runner.run(["baseline"], campaigns=1, repeats=1, max_concurrency=65)

    def test_parallel_rollouts_preserve_serial_results_and_order(self):
        scenarios = ScenarioGenerator(seed=29, horizon=6).generate(
            instances_per_role=5,
            families=["trend"],
            layers=["exogenous"],
        )
        streams = StreamGenerator(seed=31).generate(
            scenarios, templates=["accumulation"]
        )
        runner = BenchmarkRunner(scenarios, streams)
        serial = runner.run(
            ["episodic_memory"],
            campaigns=1,
            repeats=2,
            max_concurrency=1,
        )
        parallel = runner.run(
            ["episodic_memory"],
            campaigns=1,
            repeats=2,
            max_concurrency=20,
        )
        self.assertEqual(
            [record.to_dict() for record in serial["evaluation_records"]],
            [record.to_dict() for record in parallel["evaluation_records"]],
        )
        self.assertEqual(serial["training_records"], parallel["training_records"])
        self.assertEqual(serial["state_records"], parallel["state_records"])
        self.assertEqual(serial["summary"], parallel["summary"])

    def test_trajectory_partitioning_preserves_monolithic_semantics(self):
        scenarios = ScenarioGenerator(seed=37, horizon=6).generate(
            instances_per_role=5,
            families=["trend"],
            layers=["exogenous"],
        )
        streams = StreamGenerator(seed=41).generate(
            scenarios, templates=["accumulation"]
        )[:2]
        runner = BenchmarkRunner(scenarios, streams)
        conditions = ["baseline", "episodic_memory"]
        monolithic = runner.run(
            conditions,
            campaigns=2,
            repeats=1,
            state_off_conditions=["episodic_memory"],
        )
        evaluation = []
        training = []
        states = []
        for condition in conditions:
            for campaign_id in range(2):
                for stream in streams:
                    shard = runner.run(
                        [condition],
                        campaigns=1,
                        campaign_offset=campaign_id,
                        repeats=1,
                        stream_ids=[stream.stream_id],
                        state_off_conditions=(
                            [condition] if condition == "episodic_memory" else []
                        ),
                    )
                    evaluation.extend(shard["evaluation_records"])
                    training.extend(shard["training_records"])
                    states.extend(shard["state_records"])
        self.assertEqual(
            [record.to_dict() for record in monolithic["evaluation_records"]],
            [record.to_dict() for record in evaluation],
        )
        self.assertEqual(monolithic["training_records"], training)
        self.assertEqual(monolithic["state_records"], states)

    def test_campaign_offset_is_recorded_without_changing_agent_seed(self):
        scenarios = ScenarioGenerator(seed=43, horizon=6).generate(
            instances_per_role=5, families=["trend"], layers=["exogenous"]
        )
        streams = StreamGenerator(seed=47).generate(
            scenarios, templates=["accumulation"]
        )[:1]
        runner = BenchmarkRunner(scenarios, streams)
        result = runner.run(
            ["baseline"], campaigns=1, campaign_offset=4, repeats=1
        )
        self.assertTrue(result["evaluation_records"])
        self.assertTrue(
            all(record.campaign_id == 4 for record in result["evaluation_records"])
        )
        self.assertTrue(
            all(record["campaign_id"] == 4 for record in result["training_records"])
        )

    def test_scripted_reflection_respects_persistent_state_capacity(self):
        scenarios = ScenarioGenerator(seed=4, horizon=6).generate(
            instances_per_role=5,
            families=["trend"],
            layers=["exogenous"],
        )
        learning = [scenario for scenario in scenarios if scenario.role == "learn_near"][:3]
        agent = AdaptiveTradingAgent("reflection", seed=4, max_entries=2)
        market = MarketSimulator()
        for index, scenario in enumerate(learning):
            result = market.run(scenario, agent, execution_seed=index)
            agent.end_episode(result, allow_state_update=True, expose_feedback=True)
        self.assertEqual(len(agent.state_snapshot()["reflection"]), 2)

    def test_end_to_end_longitudinal_run(self):
        scenarios = ScenarioGenerator(seed=13, horizon=10).generate(
            instances_per_role=5,
            families=["trend"],
            layers=["exogenous"],
        )
        streams = StreamGenerator(seed=8).generate(scenarios, templates=["accumulation"])
        result = BenchmarkRunner(scenarios, streams).run(
            conditions=["baseline", "episodic_memory"],
            campaigns=1,
            repeats=2,
        )
        self.assertGreater(len(result["evaluation_records"]), 0)
        self.assertGreater(len(result["training_records"]), 0)
        k0 = [row for row in result["summary"]["paired_ceg"] if row["checkpoint_id"] == "K0"]
        self.assertTrue(k0)
        self.assertTrue(all(abs(row["mean_ceg"]) < 1e-12 for row in k0))
        self.assertTrue(result["summary"]["state_utilization_effect"])
        off_records = [
            record for record in result["evaluation_records"] if record.condition == "episodic_memory_state_off"
        ]
        self.assertTrue(off_records)
        episodic_memory_hashes = {
            (record.stream_id, record.checkpoint_id, record.probe_episode_id, record.repeat_id): record.state_hash
            for record in result["evaluation_records"]
            if record.condition == "episodic_memory"
        }
        self.assertTrue(
            all(
                episodic_memory_hashes[
                    (record.stream_id, record.checkpoint_id, record.probe_episode_id, record.repeat_id)
                ]
                == record.state_hash
                for record in off_records
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_run(Path(directory), result)
            self.assertTrue(Path(paths["summary"]).exists())
            self.assertTrue(Path(paths["evaluation_records"]).exists())

    def test_context_runs_longitudinally_and_supports_state_off(self):
        scenarios = ScenarioGenerator(seed=21, horizon=8).generate(
            instances_per_role=5,
            families=["trend"],
            layers=["exogenous"],
        )
        streams = StreamGenerator(seed=5).generate(scenarios, templates=["accumulation"])
        result = BenchmarkRunner(scenarios, streams).run(
            conditions=["context"],
            campaigns=1,
            repeats=1,
            state_off_conditions=["context"],
        )
        off_records = [
            record
            for record in result["evaluation_records"]
            if record.condition == "context_state_off"
        ]
        self.assertTrue(off_records)
        final_state = next(
            record["state"]
            for record in reversed(result["state_records"])
            if record["condition"] == "context"
        )
        self.assertGreater(len(final_state["context"]), 0)
        self.assertLessEqual(len(final_state["context"]), 5)

    def test_custom_agent_factory_is_used(self):
        scenarios = ScenarioGenerator(seed=8, horizon=8).generate(
            instances_per_role=5,
            families=["trend"],
            layers=["exogenous"],
        )
        streams = StreamGenerator(seed=3).generate(scenarios, templates=["accumulation"])
        calls = []

        def factory(condition: str, seed: int) -> BaseTradingAgent:
            calls.append((condition, seed))
            return AdaptiveTradingAgent("baseline", seed=seed)

        result = BenchmarkRunner(scenarios, streams, agent_factory=factory).run(
            conditions=["external-policy"], campaigns=1, repeats=1
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "external-policy")
        self.assertTrue(result["evaluation_records"])


if __name__ == "__main__":
    unittest.main()
