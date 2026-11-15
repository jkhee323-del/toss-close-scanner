"""Exercise the *real shell script* against disposable local Git repos."""
from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sync_kr_to_test.sh"


def git(work, *args, check=True):
    return subprocess.run(
        ["git", *args], cwd=work, check=check,
        text=True, capture_output=True,
    )


def setup_repo(tmp_path, *, main_names=False, main_history=True):
    remote = tmp_path / "origin.git"
    git(tmp_path, "init", "--bare", str(remote))
    work = tmp_path / "working"
    git(tmp_path, "clone", str(remote), str(work))
    git(work, "config", "user.email", "test@example.org")
    git(work, "config", "user.name", "Automation Test")

    (work / "README.md").write_text("test\n")
    if main_history:
        (work / "data/history").mkdir(parents=True)
        (work / "data/history/005930.csv").write_text(
            "date,close\n2026-11-10,100\n"
        )
    if main_names:
        (work / "data").mkdir(exist_ok=True)
        (work / "data/kr_names.csv").write_text("code,name\n005930,new\n")
    git(work, "add", ".")
    git(work, "commit", "-m", "initial main")
    git(work, "branch", "-M", "main")
    git(work, "push", "-u", "origin", "main")

    git(work, "switch", "-c", "accuracy-lab")
    (work / "data/history").mkdir(parents=True, exist_ok=True)
    (work / "data/history/005930.csv").write_text(
        "date,close\n2026-11-03,99\n"
    )
    (work / "data/kr_names.csv").write_text("code,name\n005930,old\n")
    git(work, "add", ".")
    git(work, "commit", "-m", "stale accuracy lab")
    git(work, "push", "-u", "origin", "accuracy-lab")
    git(work, "switch", "main")
    main_sha = git(work, "rev-parse", "HEAD").stdout.strip()
    return work, main_sha


@pytest.mark.parametrize("main_names", [False, True])
def test_sync_copies_korean_history_and_handles_optional_name_file(
    tmp_path, main_names
):
    work, sha = setup_repo(tmp_path, main_names=main_names)
    result = subprocess.run(
        ["bash", str(SCRIPT), sha, "accuracy-lab"], cwd=work,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert (work / "data/history/005930.csv").read_text() == (
        "date,close\n2026-11-10,100\n"
    )
    assert (work / "data/kr_names.csv").read_text() == (
        "code,name\n005930,new\n" if main_names else "code,name\n005930,old\n"
    )
    git(work, "fetch", "origin", "accuracy-lab")
    assert (git(work, "rev-parse", "HEAD").stdout.strip() ==
            git(work, "rev-parse", "origin/accuracy-lab").stdout.strip())

    # Re-running the exact same sync is idempotent.
    git(work, "switch", "main")
    second = subprocess.run(
        ["bash", str(SCRIPT), sha, "accuracy-lab"], cwd=work,
        capture_output=True, text=True,
    )
    assert second.returncode == 0, second.stderr
    assert "No KR cache changes" in second.stdout


def test_sync_refuses_main_commit_without_korean_history(tmp_path):
    work, sha = setup_repo(tmp_path, main_history=False)
    original = git(work, "rev-parse", "HEAD").stdout.strip()
    failed = subprocess.run(
        ["bash", str(SCRIPT), sha, "accuracy-lab"], cwd=work,
        capture_output=True, text=True,
    )
    assert failed.returncode != 0
    assert "KR history missing" in failed.stdout
    assert git(work, "rev-parse", "HEAD").stdout.strip() == original
