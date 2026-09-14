# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "release_version.py"
sys.path.insert(0, str(SCRIPT.parent))

from release_version import pyprojects, read_version, refusals  # noqa: E402


def write_packages(tmp_path: Path, *versions: str) -> list[Path]:
    files = []
    for i, version in enumerate(versions):
        path = tmp_path / f"p{i}" / "pyproject.toml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(f'[project]\nname = "p{i}"\nversion = "{version}"\n')
        files.append(path)
    return files


def test_a_tag_naming_every_packages_version_is_accepted(tmp_path):
    assert refusals("v0.1.2", write_packages(tmp_path, "0.1.2", "0.1.2", "0.1.2")) == []


def test_a_tag_naming_a_different_version_is_refused_for_every_package(tmp_path):
    problems = refusals("v0.1.2", write_packages(tmp_path, "0.1.3", "0.1.3", "0.1.3"))
    assert len(problems) == 3
    assert all("'0.1.3'" in p and "'v0.1.2'" in p for p in problems)


def test_a_tag_on_a_version_that_was_never_bumped_is_refused(tmp_path):
    assert len(refusals("v0.1.3", write_packages(tmp_path, "0.1.2", "0.1.2", "0.1.2"))) == 3


def test_one_package_that_disagrees_with_the_tag_is_refused_and_named(tmp_path):
    files = write_packages(tmp_path, "0.1.2", "0.1.3", "0.1.2")
    problems = refusals("v0.1.2", files)
    assert problems == [f"{files[1]} says '0.1.3', tag 'v0.1.2' releases '0.1.2'"]


@pytest.mark.parametrize("tag", ["0.1.2", "v", "", "release-0.1.2"])
def test_a_tag_that_is_not_v_followed_by_a_version_is_refused(tmp_path, tag):
    assert refusals(tag, write_packages(tmp_path, "0.1.2")) != []


def test_a_pre_release_tag_must_spell_the_version_as_pyproject_does(tmp_path):
    # Same PEP 440 version, different string: refused, because the comparison fails closed.
    assert refusals("v0.1.2-rc.1", write_packages(tmp_path, "0.1.2rc1")) != []
    assert refusals("v0.1.2rc1", write_packages(tmp_path, "0.1.2rc1")) == []


def test_the_packages_in_this_repository_agree_with_each_other():
    # A pull request that moves one package's version and not the others fails here, before any tag
    # exists, rather than on the release run.
    versions = {path.parent.name: read_version(path) for path in pyprojects()}
    assert len(set(versions.values())) == 1, versions


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_the_command_release_yml_runs_refuses_a_mismatched_tag_with_a_non_zero_exit():
    version = read_version(pyprojects()[0])
    result = run(f"v{version}.999")
    assert result.returncode == 1
    assert result.stdout.count("::error::") == 3
    assert "client/pyproject.toml" in result.stdout


def test_the_command_release_yml_runs_accepts_the_tag_this_tree_releases():
    result = run(f"v{read_version(pyprojects()[0])}")
    assert result.returncode == 0, result.stdout


def test_the_command_refuses_to_run_without_exactly_one_tag():
    assert run().returncode == 2
    assert run("v0.1.2", "v0.1.3").returncode == 2
