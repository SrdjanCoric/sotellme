import subprocess
import sys
from pathlib import Path

import pytest
from release_version import compute_version, read_base_version, replace_version

PYPROJECT = '[project]\nname = "sotellme"\nversion = "0.1.0"\nrequires-python = ">=3.12"\n'

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "release_version.py"


def test_a_branch_push_builds_a_dev_version_off_the_run_number() -> None:
    assert compute_version("0.1.0", "branch", "main", 42) == "0.1.0.dev42"


def test_a_release_tag_publishes_the_tags_own_version() -> None:
    assert compute_version("0.1.0", "tag", "v0.2.0", 99) == "0.2.0"


def test_a_release_tag_may_omit_the_v_prefix() -> None:
    assert compute_version("0.1.0", "tag", "1.4.2", 99) == "1.4.2"


def test_a_tag_that_is_not_a_release_version_is_refused() -> None:
    with pytest.raises(ValueError, match="not a release version"):
        compute_version("0.1.0", "tag", "v0.2", 99)


def test_the_base_version_is_read_from_pyproject() -> None:
    assert read_base_version(PYPROJECT) == "0.1.0"


def test_replacing_the_version_rewrites_only_the_project_version_field() -> None:
    rewritten = replace_version(PYPROJECT, "0.1.0.dev42")
    assert read_base_version(rewritten) == "0.1.0.dev42"
    assert 'name = "sotellme"' in rewritten


def test_replacing_the_version_when_there_is_no_field_is_refused() -> None:
    with pytest.raises(ValueError, match="version field"):
        replace_version('[project]\nname = "sotellme"\n', "0.2.0")


def test_running_the_script_stamps_pyproject_and_prints_the_version(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(PYPROJECT)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--ref-type",
            "tag",
            "--ref-name",
            "v0.2.0",
            "--run-number",
            "7",
            "--pyproject",
            str(pyproject),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "0.2.0"
    assert read_base_version(pyproject.read_text()) == "0.2.0"
