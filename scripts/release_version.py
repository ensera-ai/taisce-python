# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
"""Refuses a release whose tag and package versions disagree.

`release.yml` builds each distribution with the version its `pyproject.toml` holds, and attests the
result against the tag that triggered the run. Nothing else connects the two. A tag `v0.1.2` on a tree
saying `0.1.3` would publish 0.1.3 with an attestation naming `v0.1.2`, and a version on PyPI cannot be
withdrawn. So the release job runs this before it installs, builds or attests anything.

This decides one thing: whether every package's version is exactly the tag without its `v`. It does not
decide whether the version is a sensible next version, whether it is already on PyPI, or whether it is
valid under PEP 440. PyPI and the build already refuse those.

The comparison is exact string equality rather than PEP 440 equality, which fails closed. `0.1.2rc1`
and `0.1.2-rc.1` are the same PEP 440 version but different strings, so a pre-release has to be tagged
exactly as `pyproject.toml` spells it. Normalising both sides would accept that pair, and it would need
`packaging` in a step that runs before anything is installed.

Only the standard library is used, for the same reason: this runs before `pip install`.
"""

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ("client", "agent-framework", "langgraph")


def pyprojects(root: Path = ROOT) -> list[Path]:
    return [root / package / "pyproject.toml" for package in PACKAGES]


def read_version(pyproject: Path) -> str:
    with pyproject.open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def refusals(tag: str, files: list[Path]) -> list[str]:
    """Every reason the tag cannot release these files; empty when it can."""
    if not tag.startswith("v") or len(tag) == 1:
        return [f"tag {tag!r} is not a release tag: it must be 'v' followed by the version"]
    want = tag[1:]
    return [
        f"{path} says {got!r}, tag {tag!r} releases {want!r}"
        for path in files
        if (got := read_version(path)) != want
    ]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: release_version.py <tag>", file=sys.stderr)
        return 2
    problems = refusals(argv[1], pyprojects())
    for problem in problems:
        # The `::error::` prefix makes the reason the annotation on the failed run, not a log line.
        print(f"::error::{problem}")
    if not problems:
        print(f"every package is {argv[1][1:]}, as tag {argv[1]} releases")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
