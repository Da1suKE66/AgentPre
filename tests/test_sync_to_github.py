from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync_to_github.sh"


def _git(repository: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    )


@unittest.skipUnless(shutil.which("git"), "git is not installed")
@unittest.skipUnless(shutil.which("flock"), "flock is not installed")
class SyncToGitHubTests(unittest.TestCase):
    def _repository(self, temporary_root: Path) -> tuple[Path, Path, Path]:
        repository = temporary_root / "repo"
        remote = temporary_root / "remote.git"
        cache = temporary_root / "cache"
        repository.mkdir()
        subprocess.run(
            ["git", "init", "-b", "main"],
            cwd=repository,
            check=True,
            text=True,
            capture_output=True,
        )
        _git(repository, "config", "user.name", "Test User")
        _git(repository, "config", "user.email", "test@example.invalid")
        (repository / "README.md").write_text("base\n", encoding="utf-8")
        _git(repository, "add", "README.md")
        _git(repository, "commit", "-m", "base")
        subprocess.run(
            ["git", "init", "--bare", str(remote)],
            check=True,
            text=True,
            capture_output=True,
        )
        return repository, remote, cache

    def _sync(
        self, repository: Path, remote: Path, cache: Path
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            {
                "AGENTPRE_ROOT": str(repository),
                "AGENTPRE_CACHE_ROOT": str(cache),
                "AGENTPRE_GIT_REMOTE": str(remote),
            }
        )
        return subprocess.run(
            ["bash", str(SYNC_SCRIPT)],
            env=env,
            check=False,
            text=True,
            capture_output=True,
            timeout=30,
        )

    def test_refreshes_unchanged_ordinary_index_after_isolated_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository, remote, cache = self._repository(Path(temporary))
            reports = repository / "reports"
            reports.mkdir()
            (reports / "result.md").write_text("result\n", encoding="utf-8")

            result = self._sync(repository, remote, cache)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(_git(repository, "status", "--porcelain").stdout, "")
            local_head = _git(repository, "rev-parse", "HEAD").stdout.strip()
            remote_head = _git(
                remote, "rev-parse", "refs/heads/main"
            ).stdout.strip()
            self.assertEqual(remote_head, local_head)

    def test_absent_ordinary_index_uses_sentinel_for_initial_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            repository = temporary_root / "repo"
            remote = temporary_root / "remote.git"
            cache = temporary_root / "cache"
            repository.mkdir()
            subprocess.run(
                ["git", "init", "-b", "main"],
                cwd=repository,
                check=True,
                text=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "init", "--bare", str(remote)],
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertFalse((repository / ".git" / "index").exists())
            (repository / "README.md").write_text(
                "initial content\n", encoding="utf-8"
            )

            result = self._sync(repository, remote, cache)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((repository / ".git" / "index").is_file())
            self.assertEqual(_git(repository, "status", "--porcelain").stdout, "")
            local_head = _git(repository, "rev-parse", "HEAD").stdout.strip()
            remote_head = _git(
                remote, "rev-parse", "refs/heads/main"
            ).stdout.strip()
            self.assertEqual(remote_head, local_head)

    def test_preserves_user_staging_when_index_changes_during_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository, remote, cache = self._repository(Path(temporary))
            reports = repository / "reports"
            reports.mkdir()
            (reports / "result.md").write_text("result\n", encoding="utf-8")
            user_file = repository / "user-staged.txt"
            user_file.write_text("keep staged\n", encoding="utf-8")

            hook = repository / ".git" / "hooks" / "post-commit"
            ordinary_index = repository / ".git" / "index"
            hook.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f"GIT_INDEX_FILE={str(ordinary_index)!r} "
                "git add -- user-staged.txt\n",
                encoding="utf-8",
            )
            hook.chmod(0o755)

            result = self._sync(repository, remote, cache)

            self.assertEqual(result.returncode, 6, result.stderr)
            self.assertIn("user staging was preserved", result.stderr)
            staged = _git(
                repository, "diff", "--cached", "--name-only"
            ).stdout.splitlines()
            self.assertIn("user-staged.txt", staged)
            staged_contents = _git(
                repository, "show", ":user-staged.txt"
            ).stdout
            self.assertEqual(staged_contents, "keep staged\n")
            remote_ref = subprocess.run(
                ["git", "rev-parse", "--verify", "refs/heads/main"],
                cwd=remote,
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(remote_ref.returncode, 0)


if __name__ == "__main__":
    unittest.main()
