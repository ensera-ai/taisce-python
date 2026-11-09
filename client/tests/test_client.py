# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
import httpx
import pytest

from taisce import (Client, ErasureNotClean, ExportTooLarge, Forbidden, TaisceError, bundle_is_empty)


def test_a_client_refuses_to_be_built_without_a_deployment_or_a_credential():
    with pytest.raises(ValueError):
        Client("", "tsk")
    with pytest.raises(ValueError):
        Client("http://127.0.0.1:1", " ")
    client = Client("http://127.0.0.1:1/", "tsk")
    assert client.base_url == "http://127.0.0.1:1"


@pytest.mark.parametrize("address", [
    "https://memory.example", "https://memory.example:8443/base",
    "http://localhost:18080", "http://127.0.0.1:5", "http://127.9.9.9", "http://[::1]:8080",
])
def test_https_or_plaintext_to_a_loopback_host_is_accepted(address):
    Client(address, "tsk")


@pytest.mark.parametrize("address", [
    "http://memory.example", "http://10.0.0.5:8080", "http://localhost.evil.example",
    "ftp://localhost", "memory.example", "https://",
])
def test_an_address_the_credential_would_cross_the_network_in_clear_to_reach_is_refused(address):
    # The credential is sent with every request; one mistyped scheme would hand it to the path.
    with pytest.raises(ValueError, match="in clear"):
        Client(address, "tsk")


def test_an_empty_bundle_is_empty():
    assert bundle_is_empty({"facts": [], "reports": [], "passages": []})
    assert not bundle_is_empty({"facts": [{"fact_id": "f1"}], "reports": [], "passages": []})
    assert not bundle_is_empty({"facts": [], "reports": [{"title": "t"}], "passages": []})


def _answering(status: int, body: dict) -> Client:
    # A transport that answers one fixed response. Used only for the branches a live deployment
    # cannot be made to take on demand — a receipt that is not clean, an export over the limit — and
    # for refusals that never reach the network; the behaviour itself is proved live in tests_live.
    transport = httpx.MockTransport(lambda request: httpx.Response(status, json=body))
    return Client("http://127.0.0.1:1", "tsk", http=httpx.AsyncClient(transport=transport))


@pytest.mark.asyncio
async def test_a_receipt_that_is_not_clean_is_raised_with_the_whole_receipt():
    client = _answering(200, {"request_id": "r", "scope": "p1", "data_subject_id": "alice", "completed_at": "t",
                              "deleted": {"messages": 2}, "residual": {"messages": 1}, "clean": False})
    with pytest.raises(ErasureNotClean) as raised:
        await client.erase_subject("alice", reason="request")
    assert raised.value.receipt.residual == {"messages": 1}
    assert raised.value.receipt.deleted == {"messages": 2}


@pytest.mark.asyncio
async def test_a_refused_erasure_and_an_oversized_export_are_their_own_errors():
    with pytest.raises(Forbidden) as forbidden:
        await _answering(403, {"error": {"code": "forbidden", "message": "no"}}).erase_subject("alice", reason="r")
    assert isinstance(forbidden.value, TaisceError) and forbidden.value.status == 403
    with pytest.raises(ExportTooLarge):
        await _answering(422, {"error": {"code": "export_too_large", "message": "too many"}}).export_subject("alice")
    with pytest.raises(TaisceError) as other:
        await _answering(409, {"error": {"code": "idempotency_conflict", "message": "x"}}).erase_subject("alice", reason="r")
    assert type(other.value) is TaisceError


@pytest.mark.asyncio
async def test_an_erasure_or_export_naming_nobody_is_refused_before_it_is_sent():
    client = _answering(500, {})
    for call in (lambda: client.erase_subject(" ", reason="r"), lambda: client.erase_sources([], reason="r"),
                 lambda: client.erase_sources(["x"] * 10001, reason="r"),
                 lambda: client.export_subject(""), lambda: client.export_sources([])):
        with pytest.raises(ValueError):
            await call()
