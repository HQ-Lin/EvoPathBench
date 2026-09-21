import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evolens.models import EvaluationRecord
from evolens.trajectory_store import TrajectoryStore, trajectory_key


def evaluation(condition, campaign, stream, score):
    return EvaluationRecord(
        condition=condition,
        campaign_id=campaign,
        stream_id=stream,
        checkpoint_id="K0",
        checkpoint_order=0,
        probe_episode_id="probe",
        family_id="trend",
        role="held_out",
        layer="exogenous",
        repeat_id=0,
        score=score,
        return_pct=score,
        final_wealth=10000.0,
        max_drawdown=0.0,
        turnover=0.0,
        violation_count=0,
        fee_paid=0.0,
        state_hash="state",
        strategy_mode="flat",
    )


def result_for(key, score):
    return {
        "config": {},
        "evaluation_records": [
            evaluation(key["condition"], key["campaign_id"], key["stream_id"], score)
        ],
        "training_records": [
            {
                "condition": key["condition"],
                "campaign_id": key["campaign_id"],
                "stream_id": key["stream_id"],
                "score": score,
            }
        ],
        "state_records": [
            {
                "condition": key["condition"],
                "campaign_id": key["campaign_id"],
                "stream_id": key["stream_id"],
                "state": {},
            }
        ],
        "summary": {},
    }


USAGE = {
    "logical_calls": 1,
    "failed_calls": 0,
    "input_tokens": 10,
    "output_tokens": 2,
    "reasoning_tokens": 0,
    "total_tokens": 12,
}


class TrajectoryStoreTests(unittest.TestCase):
    def test_restart_recovers_completed_shards_and_preserves_order(self):
        keys = [trajectory_key("baseline", 0, "s1"), trajectory_key("context", 0, "s1")]
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            first = TrajectoryStore(run, keys)
            first.begin(keys[0])
            first.complete(keys[0], result_for(keys[0], 1.0), USAGE)

            resumed = TrajectoryStore(run, keys)
            self.assertEqual(resumed.snapshot()["completed_trajectories"], 1)
            self.assertEqual(resumed.snapshot()["persisted_usage"]["input_tokens"], 10)
            resumed.begin(keys[1])
            resumed.complete(keys[1], result_for(keys[1], 2.0), USAGE)
            materialized = resumed.materialize({"conditions": ["baseline", "context"]})
            self.assertEqual(
                [row.score for row in materialized["evaluation_records"]], [1.0, 2.0]
            )
            self.assertEqual(resumed.snapshot()["completed_trajectories"], 2)
            self.assertEqual(resumed.snapshot()["persisted_usage"]["total_tokens"], 24)
            events = [
                json.loads(line)
                for line in (run / "progress" / "trajectory_ledger.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [event["event"] for event in events],
                ["STARTED", "COMPLETED", "STARTED", "COMPLETED"],
            )

    def test_incomplete_trajectory_is_retried_but_completed_one_cannot_overwrite(self):
        key = trajectory_key("baseline", 0, "s1")
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            interrupted = TrajectoryStore(run, [key])
            interrupted.begin(key)
            resumed = TrajectoryStore(run, [key])
            resumed.begin(key)
            resumed.complete(key, result_for(key, 1.0), USAGE)
            self.assertEqual(resumed.snapshot()["attempts"].popitem()[1], 2)
            with self.assertRaisesRegex(ValueError, "already completed"):
                resumed.begin(key)

    def test_corrupt_shard_is_quarantined_instead_of_counted(self):
        key = trajectory_key("baseline", 0, "s1")
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            store = TrajectoryStore(run, [key])
            store.begin(key)
            store.complete(key, result_for(key, 1.0), USAGE)
            shard = next((run / "checkpoints" / "trajectories").glob("*.json"))
            shard.write_text("{broken", encoding="utf-8")
            recovered = TrajectoryStore(run, [key])
            self.assertEqual(recovered.snapshot()["completed_trajectories"], 0)
            self.assertEqual(len(recovered.snapshot()["quarantined_shards"]), 1)
            self.assertTrue(list((run / "quarantine" / "trajectories").iterdir()))

    def test_concurrent_trajectories_are_committed_once_with_valid_ledger(self):
        keys = [trajectory_key("baseline", campaign, f"s{stream}") for campaign in range(3) for stream in range(8)]
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            store = TrajectoryStore(run, keys)

            def execute(item):
                store.begin(item)
                store.complete(item, result_for(item, 1.0), USAGE)

            with ThreadPoolExecutor(max_workers=12) as executor:
                list(executor.map(execute, keys))

            snapshot = store.snapshot()
            self.assertEqual(snapshot["completed_trajectories"], len(keys))
            self.assertEqual(snapshot["active_trajectories"], [])
            self.assertEqual(snapshot["persisted_usage"]["total_tokens"], 12 * len(keys))
            shards = list((run / "checkpoints" / "trajectories").glob("*.json"))
            self.assertEqual(len(shards), len(keys))
            events = [
                json.loads(line)
                for line in (run / "progress" / "trajectory_ledger.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(sum(row["event"] == "STARTED" for row in events), len(keys))
            self.assertEqual(sum(row["event"] == "COMPLETED" for row in events), len(keys))
            resumed = TrajectoryStore(run, keys)
            self.assertEqual(resumed.snapshot()["completed_trajectories"], len(keys))


if __name__ == "__main__":
    unittest.main()
