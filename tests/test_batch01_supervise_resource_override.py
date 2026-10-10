"""Running-resource warnings must never kill a valid bounded Batch01 run."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("batch01_supervise", TOOLS / "batch01_supervise.py")
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


class RuntimeResourceOverrideTests(unittest.TestCase):
    def run_supervision(self, changes=None, *, shared=False, live_count=1):
        # Every signal and subprocess side effect is mocked; no real run is touched.
        with tempfile.TemporaryDirectory(dir="/home/yjq", prefix="batch01-supervise-test-") as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoints" / "step.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"valid full-state checkpoint fixture")
            row = {
                "mode": "PROVISIONAL", "execution_mode": "PROVISIONAL",
                "run_id": "TEST_ONLY", "pid": 2000000000, "alive": True,
                "variant": "p1-control", "output_root": str(root),
                "gpu_uuid": "TEST-GPU", "lease_id": "TEST-LEASE",
                "step": 256, "start_step": 0, "authorized_end_step": 25000,
                "status": "PROVISIONAL_RUNNING", "last_exception": None,
                "disk_bytes": 12 * 1024**3, "torch_peak_allocated_mib": 4096,
                "checkpoint": {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
            }
            row.update(changes or {})
            if row.pop("test_bad_checkpoint_hash", False):
                row["checkpoint"]["sha256"] = "WRONG"
            rows = [row] + [dict(row, run_id=f"TEST-{i}", pid=row["pid"] + i) for i in range(1, live_count)]
            registry = {"pilot_runs": rows}
            state = {
                "QA_SAME_SHA": "PENDING", "base_freeze_status": "BASE_FREEZE_BLOCKED",
                "pilot_gpu_leases": [{"id": "TEST-LEASE", "expires_at": "2000-01-01T00:00:00+00:00", "external_sharing_authorized": shared}],
            }

            def fake_read(path):
                if Path(path).name == "batch_state.json":
                    return state
                if Path(path).name == "provisional_long_authorization_20261010.json":
                    return {"runs": {"p1-control": {"run_id": "TEST_ONLY", "start_step": 0, "authorized_end_step": 25000}}}
                return registry

            original_digest = supervisor.digest
            def fake_digest(path):
                return "RIGHT_AUTH_HASH" if path.name == "provisional_long_authorization_20261010.json" else original_digest(path)

            with patch.object(supervisor, "RUNTIME", root), \
                 patch.object(supervisor, "inspect_runs", return_value=(registry, rows)), \
                 patch.object(supervisor, "read", side_effect=fake_read), \
                 patch.object(supervisor, "digest", side_effect=fake_digest), \
                 patch.object(supervisor, "write"), \
                 patch.object(supervisor, "memory", return_value={"MemAvailable": 1, "MemTotal": 128 * 1024**3}), \
                 patch.object(supervisor, "process_rss", return_value=0), \
                 patch.object(supervisor.os, "statvfs", return_value=SimpleNamespace(f_bavail=0, f_frsize=4096)), \
                 patch.object(supervisor.subprocess, "check_output", side_effect=["999999999, TEST-GPU, 1024 MiB", "TEST-GPU, 0"]), \
                 patch.object(supervisor.subprocess, "run"), \
                 patch.object(supervisor, "alive", return_value=True), \
                 patch.object(supervisor.os, "kill") as kill:
                snapshot = supervisor.supervise({})
                return snapshot, list(kill.call_args_list)

    def test_all_resource_thresholds_and_external_process_only_warn(self):
        snapshot, signals = self.run_supervision()
        self.assertEqual(signals, [])
        self.assertEqual(snapshot["events"], [])
        self.assertFalse(snapshot["runtime_resource_stop_enabled"])
        self.assertTrue(snapshot["startup_resource_gates_enabled"])
        self.assertEqual(len(snapshot["resource_alerts"][0]["reasons"]), 7)
        self.assertEqual(snapshot["resource_alerts"][0]["action"], "RESOURCE_ALERT_ONLY")

    def test_changed_shared_gpu_count_never_stops_running_jobs(self):
        snapshot, signals = self.run_supervision(shared=True, live_count=3)
        self.assertEqual(signals, [])
        self.assertTrue(all("shared GPU above startup maximum2 own lines" in a["reasons"] for a in snapshot["resource_alerts"]))

    def test_missing_lease_does_not_stop_running_job(self):
        snapshot, signals = self.run_supervision({"lease_id": "MISSING"})
        self.assertEqual(signals, [])
        self.assertIn("GPU lease missing/expired", snapshot["resource_alerts"][0]["reasons"])

    def assert_scientific_stop(self, changes, expected):
        snapshot, signals = self.run_supervision(changes)
        self.assertEqual(len(signals), 1)
        self.assertEqual(snapshot["events"][0]["action"], "SIGTERM_OWN_RUN")
        self.assertIn(expected, snapshot["events"][0]["reason"])
        self.assertTrue(snapshot["resource_alerts"])

    def test_budget_gate_remains_enforced(self):
        self.assert_scientific_stop({"step": 25001}, "budget exceeded")

    def test_checkpoint_scope_gate_remains_enforced(self):
        self.assert_scientific_stop({"output_root": "/not-this-run"}, "checkpoint escapes")

    def test_checkpoint_hash_gate_remains_enforced(self):
        self.assert_scientific_stop({"test_bad_checkpoint_hash": True}, "checkpoint hash mismatch")

    def test_scientific_quarantine_remains_enforced(self):
        self.assert_scientific_stop({"scientific_quarantine": True}, "provenance quarantined")

    def test_failed_contract_remains_enforced(self):
        self.assert_scientific_stop({"last_exception": "SourceGuard rejected source"}, "failed contract")

    def test_long_authorization_hash_gate_remains_enforced(self):
        self.assert_scientific_stop({"execution_mode": "PROVISIONAL_LONG", "authorization_sha256": "WRONG"}, "long authorization pin mismatch")


if __name__ == "__main__":
    unittest.main()
