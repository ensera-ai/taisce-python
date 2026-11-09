# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
"""These tests run against a live deployment and nothing else.

They read TAISCE_API, TAISCE_TOKEN (read_write) and TAISCE_READER_TOKEN (read_only), and fail when
one is missing rather than skipping: a green run that skipped asserted nothing. The conformance
workflow stands a deployment up and runs them; the default test run has no deployment and does not
collect this directory.
"""
import os

import pytest


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.fail(f"{name} is not set: these tests run against a live deployment", pytrace=False)
    return value


@pytest.fixture
def api() -> str:
    return _required("TAISCE_API")


@pytest.fixture
def token() -> str:
    return _required("TAISCE_TOKEN")


@pytest.fixture
def reader_token() -> str:
    return _required("TAISCE_READER_TOKEN")
