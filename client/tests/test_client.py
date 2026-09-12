# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
import pytest

from taisce import Client, bundle_is_empty


def test_a_client_refuses_to_be_built_without_a_deployment_or_a_credential():
    with pytest.raises(ValueError):
        Client("", "tsk")
    with pytest.raises(ValueError):
        Client("http://127.0.0.1:1", " ")
    client = Client("http://127.0.0.1:1/", "tsk")
    assert client.base_url == "http://127.0.0.1:1"


def test_an_empty_bundle_is_empty():
    assert bundle_is_empty({"facts": [], "reports": [], "passages": []})
    assert not bundle_is_empty({"facts": [{"fact_id": "f1"}], "reports": [], "passages": []})
    assert not bundle_is_empty({"facts": [], "reports": [{"title": "t"}], "passages": []})
