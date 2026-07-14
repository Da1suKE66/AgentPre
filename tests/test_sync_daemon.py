from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
DAEMON = REPO_ROOT / "scripts" / "sync_daemon.sh"
INSTALLER = REPO_ROOT / "scripts" / "install_sync_daemon.sh"
LEGACY_CRON_MARKER = "# AgentPre periodic GitHub sync"


def _run_daemon(
    command: str,
    *,
    root: Path,
    cache: Path,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "AGENTPRE_ROOT": str(root),
            "AGENTPRE_CACHE_ROOT": str(cache),
            "AGENTPRE_SYNC_INTERVAL_SECONDS": "1",
            "AGENTPRE_SYNC_HEARTBEAT_SECONDS": "1",
        }
    )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(DAEMON), command],
        env=env,
        check=False,
        text=True,
        capture_output=True,
        timeout=15,
    )


class SyncDaemonTests(unittest.TestCase):
    @staticmethod
    def _write_daemon_stub(root: Path, call_file: Path) -> None:
        scripts = root / "scripts"
        scripts.mkdir(parents=True)
        daemon_stub = scripts / "sync_daemon.sh"
        daemon_stub.write_text(
            "#!/bin/sh\n"
            f"printf '%s\\n' \"$1\" > {str(call_file)!r}\n",
            encoding="utf-8",
        )
        daemon_stub.chmod(0o755)

    def test_scripts_have_valid_bash_syntax(self) -> None:
        for script in (DAEMON, INSTALLER):
            with self.subTest(script=script.name):
                subprocess.run(["bash", "-n", str(script)], check=True)

    def test_declares_cache_state_singleton_and_no_gpu_policy(self) -> None:
        source = DAEMON.read_text(encoding="utf-8")
        self.assertIn("github-sync-daemon.pid", source)
        self.assertIn("github-sync-daemon.heartbeat", source)
        self.assertIn("github-sync-daemon.lock", source)
        self.assertIn("flock -n 9", source)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', source)
        self.assertIn('export NVIDIA_VISIBLE_DEVICES="void"', source)
        self.assertIn('SYNC_SCRIPT="${ROOT}/scripts/sync_to_github.sh"', source)
        self.assertIn('nohup "${SCRIPT_PATH}" run 8>&-', source)
        self.assertIn(
            'INTERVAL_MINUTES="${AGENTPRE_SYNC_INTERVAL_MINUTES:-30}"', source
        )

    def test_status_is_nonzero_when_no_daemon_is_running(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            result = _run_daemon(
                "status",
                root=root / "repo",
                cache=cache,
            )
            cache_created = cache.exists()
        self.assertEqual(result.returncode, 3)
        self.assertIn("is stopped", result.stdout)
        self.assertFalse(cache_created, "status must not create cache/state paths")

    def test_invalid_interval_is_rejected_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = _run_daemon(
                "start",
                root=root / "repo",
                cache=root / "cache",
                extra_env={"AGENTPRE_SYNC_INTERVAL_SECONDS": "0"},
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be a positive integer", result.stderr)

    def test_installer_removes_only_legacy_cron_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            root = temporary_root / "repo"
            fake_bin = temporary_root / "bin"
            fake_bin.mkdir()
            call_file = temporary_root / "daemon-call"
            crontab_file = temporary_root / "crontab"
            self._write_daemon_stub(root, call_file)

            crontab_file.write_text(
                "SHELL=/bin/bash\n"
                "5 * * * * /home/user/hourly-backup\n"
                f"{LEGACY_CRON_MARKER}\n"
                f"*/30 * * * * {root}/scripts/sync_to_github.sh\n"
                "# AgentPre periodic GitHub sync retained note\n"
                "15 2 * * * /home/user/nightly-report\n",
                encoding="utf-8",
            )
            fake_crontab = fake_bin / "crontab"
            fake_crontab.write_text(
                "#!/bin/sh\n"
                "case \"${1-}\" in\n"
                "  -l) /bin/cat \"${FAKE_CRONTAB_FILE}\" ;;\n"
                "  -) /bin/cat > \"${FAKE_CRONTAB_FILE}\" ;;\n"
                "  *) exit 64 ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            fake_crontab.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "AGENTPRE_ROOT": str(root),
                    "DAEMON_CALL_FILE": str(call_file),
                    "FAKE_CRONTAB_FILE": str(crontab_file),
                    "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
                }
            )
            result = subprocess.run(
                ["bash", str(INSTALLER)],
                env=env,
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(call_file.read_text(encoding="utf-8"), "start\n")
            self.assertEqual(
                crontab_file.read_text(encoding="utf-8"),
                "SHELL=/bin/bash\n"
                "5 * * * * /home/user/hourly-backup\n"
                "# AgentPre periodic GitHub sync retained note\n"
                "15 2 * * * /home/user/nightly-report\n",
            )
            self.assertIn("unrelated crontab entries were preserved", result.stdout)

    def test_installer_preserves_isolated_legacy_marker_and_next_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            root = temporary_root / "repo"
            fake_bin = temporary_root / "bin"
            fake_bin.mkdir()
            call_file = temporary_root / "daemon-call"
            crontab_file = temporary_root / "crontab"
            self._write_daemon_stub(root, call_file)

            original = (
                "SHELL=/bin/bash\n"
                f"{LEGACY_CRON_MARKER}\n"
                "15 2 * * * /home/user/nightly-report\n"
                f"{LEGACY_CRON_MARKER}\n"
            )
            crontab_file.write_text(original, encoding="utf-8")
            fake_crontab = fake_bin / "crontab"
            fake_crontab.write_text(
                "#!/bin/sh\n"
                "case \"${1-}\" in\n"
                "  -l) /bin/cat \"${FAKE_CRONTAB_FILE}\" ;;\n"
                "  -) /bin/cat > \"${FAKE_CRONTAB_FILE}\" ;;\n"
                "  *) exit 64 ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            fake_crontab.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "AGENTPRE_ROOT": str(root),
                    "DAEMON_CALL_FILE": str(call_file),
                    "FAKE_CRONTAB_FILE": str(crontab_file),
                    "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
                }
            )
            result = subprocess.run(
                ["bash", str(INSTALLER)],
                env=env,
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(call_file.read_text(encoding="utf-8"), "start\n")
            self.assertEqual(crontab_file.read_text(encoding="utf-8"), original)
            self.assertNotIn("Removed legacy", result.stdout)

    def test_installer_starts_when_crontab_command_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            root = temporary_root / "repo"
            empty_bin = temporary_root / "empty-bin"
            empty_bin.mkdir()
            call_file = temporary_root / "daemon-call"
            self._write_daemon_stub(root, call_file)

            env = os.environ.copy()
            env.update(
                {
                    "AGENTPRE_ROOT": str(root),
                    "DAEMON_CALL_FILE": str(call_file),
                    "PATH": str(empty_bin),
                }
            )
            result = subprocess.run(
                ["/bin/bash", str(INSTALLER)],
                env=env,
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(call_file.read_text(encoding="utf-8"), "start\n")
            self.assertIn("crontab is unavailable", result.stdout)

    @unittest.skipUnless(shutil.which("flock"), "flock is not installed")
    def test_lifecycle_is_singleton_and_runs_sync_without_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            root = temporary_root / "repo"
            cache = temporary_root / "cache"
            scripts = root / "scripts"
            scripts.mkdir(parents=True)
            sync_script = scripts / "sync_to_github.sh"
            sync_script.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf 'cuda=%s nvidia=%s\\n' \"${CUDA_VISIBLE_DEVICES-unset}\" "
                "\"${NVIDIA_VISIBLE_DEVICES-unset}\" >> "
                "\"${AGENTPRE_CACHE_ROOT}/sync-calls.log\"\n",
                encoding="utf-8",
            )
            sync_script.chmod(0o755)

            try:
                first_start = _run_daemon("start", root=root, cache=cache)
                self.assertEqual(first_start.returncode, 0, first_start.stderr)
                pid_file = cache / "run" / "github-sync-daemon.pid"
                first_pid = pid_file.read_text(encoding="utf-8").strip()

                second_start = _run_daemon("start", root=root, cache=cache)
                self.assertEqual(second_start.returncode, 0, second_start.stderr)
                self.assertEqual(
                    pid_file.read_text(encoding="utf-8").strip(), first_pid
                )
                self.assertIn("already running", second_start.stdout)

                deadline = time.monotonic() + 5.0
                calls_file = cache / "sync-calls.log"
                calls: list[str] = []
                while time.monotonic() < deadline:
                    if calls_file.exists():
                        calls = calls_file.read_text(encoding="utf-8").splitlines()
                        if len(calls) >= 2:
                            break
                    time.sleep(0.1)
                self.assertGreaterEqual(len(calls), 2)
                self.assertTrue(all(line == "cuda= nvidia=void" for line in calls))
                daemon_log = (
                    cache / "logs" / "github-sync-daemon.log"
                ).read_text(encoding="utf-8")
                self.assertGreaterEqual(daemon_log.count("sync_attempt_started"), 2)
                self.assertGreaterEqual(daemon_log.count("sync_attempt_finished"), 2)

                status = _run_daemon("status", root=root, cache=cache)
                self.assertEqual(status.returncode, 0)
                self.assertIn(f"pid={first_pid}", status.stdout)
                heartbeat = (
                    cache / "run" / "github-sync-daemon.heartbeat"
                ).read_text(encoding="utf-8")
                self.assertIn("state=", heartbeat)
                self.assertIn("interval_seconds=1", heartbeat)
            finally:
                _run_daemon("stop", root=root, cache=cache)

            self.assertFalse((cache / "run" / "github-sync-daemon.pid").exists())


if __name__ == "__main__":
    unittest.main()
