# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
"""The client: the v1 contract of a Taisce deployment, and nothing about any agent framework.

The adapters are built on this, so a caller who wants governed memory without a framework does not
acquire one by asking. It is asynchronous because the frameworks it serves are; the sync wrapper
exists for scripts. The credential is a bearer token bound to one project, sent nowhere but the
deployment this client was built for, and never logged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlsplit

import base64
import ipaddress

import httpx

# The service refuses an erasure naming more sources than this; refusing it here says so before the
# credential is spent on a request that cannot succeed.
MAX_ERASURE_SOURCES = 10000


class TaisceError(Exception):
    """A refusal from the deployment, carrying the contract's own code. Branch on ``code``."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message


class Forbidden(TaisceError):
    """The credential's access does not cover the operation: a read-only key changing memory, or an
    agent key erasing a person. Retrying with the same key cannot succeed."""


class ExportTooLarge(TaisceError):
    """The subject holds more rows than one export returns. The deployment refuses rather than
    truncates, because an export that silently stops short is a false answer to an access request."""


@dataclass(frozen=True)
class ErasureReceipt:
    """What an erasure did, as the deployment counted it.

    ``residual`` is the product: a count of what still matches, taken after the delete in the same
    transaction. ``clean`` is that every residual is zero. A receipt that is not clean is never handed
    back as a value; :class:`ErasureNotClean` carries it instead.
    """

    request_id: str
    scope: str
    data_subject_id: str
    source_observation_ids: List[str]
    completed_at: str
    deleted: Dict[str, int]
    residual: Dict[str, int]
    clean: bool

    @staticmethod
    def _from(data: Mapping[str, Any]) -> "ErasureReceipt":
        return ErasureReceipt(
            request_id=data.get("request_id", ""), scope=data.get("scope", ""),
            data_subject_id=data.get("data_subject_id", ""),
            source_observation_ids=list(data.get("source_observation_ids") or []),
            completed_at=data.get("completed_at", ""),
            deleted={k: int(v) for k, v in (data.get("deleted") or {}).items()},
            residual={k: int(v) for k, v in (data.get("residual") or {}).items()},
            clean=bool(data.get("clean")),
        )


class ErasureNotClean(Exception):
    """An erasure completed and rows still match what it named.

    Raised rather than returned so that an application which never reads the receipt cannot tell a
    person their data is gone while it is not. The receipt is on ``receipt``, whole.
    """

    def __init__(self, receipt: ErasureReceipt) -> None:
        left = {k: v for k, v in receipt.residual.items() if v}
        super().__init__(f"the erasure left rows matching what it named: {left}")
        self.receipt = receipt


@dataclass(frozen=True)
class Export:
    """Everything the deployment holds for a person or a set of sources.

    ``sections`` are raw rows keyed by what produced them, unshaped so that a column a migration
    added is in the export without anybody remembering to add it here. ``rows`` counts each section,
    so an export can be compared with an erasure receipt without parsing either.
    """

    scope: str
    data_subject_id: str
    source_observation_ids: List[str]
    sections: Dict[str, List[Any]] = field(default_factory=dict)
    rows: Dict[str, int] = field(default_factory=dict)

    @staticmethod
    def _from(data: Mapping[str, Any]) -> "Export":
        return Export(
            scope=data.get("scope", ""), data_subject_id=data.get("data_subject_id", ""),
            source_observation_ids=list(data.get("source_observation_ids") or []),
            sections={k: list(v or []) for k, v in (data.get("sections") or {}).items()},
            rows={k: int(v) for k, v in (data.get("rows") or {}).items()},
        )


def _require_transport(base_url: str) -> None:
    """Refuses an address the credential would cross the network in clear to reach.

    Every request carries the project credential, and a project holds every end user's memory. One
    mistyped or downgraded address would send it to anyone on the path, readable, until somebody
    revokes it. Plaintext is allowed only to a loopback host, which never leaves the machine: that is
    where a development deployment and every conformance run live. There is no switch to allow more,
    because nothing has needed one, and a switch is the thing that gets left on.
    """
    parts = urlsplit(base_url.strip())
    if parts.scheme == "https" and parts.hostname:
        return
    if parts.scheme == "http" and parts.hostname and _is_loopback(parts.hostname):
        return
    raise ValueError(
        "a deployment address must be https, or http to a loopback host: "
        "the credential travels with every request and would otherwise cross the network in clear")


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class Freshness:
    """How far behind memory is: the highest offset stored, the highest formed, the parked count."""

    scope: str
    stored: Optional[int]
    formed: Optional[int]
    parked: int


class Client:
    """An asynchronous client over the v1 contract."""

    def __init__(self, base_url: str, token: str, *, http: Optional[httpx.AsyncClient] = None, timeout: float = 30.0) -> None:
        if not base_url or not base_url.strip():
            raise ValueError("a deployment address is required")
        if not token or not token.strip():
            raise ValueError("a credential is required")
        _require_transport(base_url)
        self.base_url = base_url.rstrip("/")
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(timeout=timeout)
        self._headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> "Client":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def observe(self, *, idempotency_key: str, messages: Sequence[Mapping[str, Any]],
                      data_subject_id: Optional[str] = None, occurred_at: Optional[str] = None) -> dict:
        """Records a turn. Durable when it returns; formation follows."""
        body: dict = {"idempotency_key": idempotency_key, "messages": list(messages)}
        if data_subject_id:
            body["data_subject_id"] = data_subject_id
        if occurred_at:
            body["occurred_at"] = occurred_at
        return await self._post("/v1/observations", body, 201)

    async def freshness(self) -> Freshness:
        """How far behind memory is."""
        response = await self._http.get(self.base_url + "/v1/freshness", headers=self._headers)
        data = self._read(response, 200)
        return Freshness(scope=data.get("scope", ""), stored=data.get("stored"), formed=data.get("formed"), parked=int(data.get("parked", 0)))

    async def recall(self, *, question: str, data_subject_id: Optional[str] = None, **controls: Any) -> dict:
        """Asks memory a question. ``controls`` are the contract's: max_characters, source_roles, hops, surfaces, themes, as_of, as_known_at."""
        body: dict = {"question": question}
        if data_subject_id:
            body["data_subject_id"] = data_subject_id
        body.update({k: v for k, v in controls.items() if v is not None})
        return await self._post("/v1/recalls", body, 200)

    async def put_artifact(self, *, artifact_id: str, data_subject_id: str, kind: str, content: bytes,
                           name: Optional[str] = None, expected_version: Optional[str] = None) -> dict:
        """Stores an opaque object under a person, replacing the version it names.

        The bytes are the application's and the deployment never interprets them. It holds them
        under a project and a person, expires them with that person's retention and removes them
        with that person's erasure — which is the whole reason to keep session state here rather
        than in an application's own database, where a deletion request would have two places to
        sweep and a counted residual for only one of them.
        """
        body: dict = {"id": artifact_id, "data_subject_id": data_subject_id, "kind": kind,
                      "content": base64.b64encode(content).decode("ascii")}
        if name:
            body["name"] = name
        if expected_version:
            body["expected_version"] = expected_version
        return await self._post("/v1/artifacts/put", body, 200)

    async def get_artifact(self, *, artifact_id: str, data_subject_id: Optional[str] = None) -> dict:
        """Reads one object, optionally requiring it to belong to the person named.

        One credential opens a project, and a project holds every end user's objects. Naming the
        subject makes "this one is theirs" a requirement the deployment enforces rather than a habit
        the application keeps: another person's object is answered exactly as one that is not there.
        """
        body: dict = {"id": artifact_id}
        if data_subject_id:
            body["data_subject_id"] = data_subject_id
        return await self._post("/v1/artifacts/get", body, 200)

    async def delete_artifact(self, *, artifact_id: str, data_subject_id: Optional[str] = None,
                              expected_version: Optional[str] = None) -> dict:
        """Removes one object, optionally requiring it to belong to the person named."""
        body: dict = {"id": artifact_id}
        if data_subject_id:
            body["data_subject_id"] = data_subject_id
        if expected_version:
            body["expected_version"] = expected_version
        return await self._post("/v1/artifacts/delete", body, 200)

    async def search_passages(self, *, question: str, limit: Optional[int] = None,
                              data_subject_id: Optional[str] = None,
                              source_role: Optional[str] = None) -> dict:
        """Searches the stored words themselves, for a question no fact answers.

        This is the other half of retrieval and deliberately separate from :meth:`recall`: recall
        answers from what was inferred, this answers from what was said. A caller that wants
        grounding without adopting the memory model uses only this.

        The answer carries ``approximate`` and ``covered_through_offset`` because a passage search
        reads an embedding generation, and a generation is built up to a point in the log. Treating
        an answer as complete while the build is behind quotes an index rather than the memory.
        """
        body: dict = {"question": question}
        if limit is not None:
            body["limit"] = limit
        if data_subject_id:
            body["data_subject_id"] = data_subject_id
        if source_role:
            body["source_role"] = source_role
        return await self._post("/v1/passages/search", body, 200)

    async def context(self, *, data_subject_id: str, max_characters: Optional[int] = None) -> dict:
        """One subject's history under a budget: the newest turns verbatim and, over the rest, the
        segments the deployment wrote. No model call is made; what the deployment has not rolled up
        yet comes back verbatim and cut from the oldest end."""
        if not data_subject_id or not data_subject_id.strip():
            raise ValueError("a data subject is required: a context is one subject's history")
        body: dict = {"data_subject_id": data_subject_id}
        if max_characters is not None:
            body["max_characters"] = max_characters
        return await self._post("/v1/contexts", body, 200)

    async def resolve_citation(self, *, fact_id: str, limit: Optional[int] = None) -> dict:
        """Resolves a fact to its record."""
        body: dict = {"id": fact_id}
        if limit is not None:
            body["limit"] = limit
        return await self._post("/v1/citations/resolve", body, 200)

    async def erase_subject(self, data_subject_id: str, *, reason: str) -> ErasureReceipt:
        """Erases everything held for one person and returns the deployment's receipt.

        Raises :class:`ErasureNotClean` when rows still match afterwards, and :class:`Forbidden` when
        the credential may not erase. An agent key may not, by design: give this call the key your
        application's erasure path holds, never the one in the agent process.
        """
        if not data_subject_id or not data_subject_id.strip():
            raise ValueError("an erasure names the person it is for; one naming nobody would be a scope-wide delete")
        return await self._erase({"data_subject_id": data_subject_id, "reason": reason})

    async def erase_sources(self, source_observation_ids: Sequence[str], *, reason: str) -> ErasureReceipt:
        """Erases what the named observations said and everything formed from them.

        A separate method from :meth:`erase_subject` because the deployment refuses a request naming
        both a person and sources: the receipt says what was asked, and it was one thing.
        """
        ids = list(source_observation_ids)
        if not ids:
            raise ValueError("an erasure by source names at least one observation")
        if len(ids) > MAX_ERASURE_SOURCES:
            raise ValueError(f"an erasure names at most {MAX_ERASURE_SOURCES} source observations")
        return await self._erase({"source_observation_ids": ids, "reason": reason})

    async def export_subject(self, data_subject_id: str) -> Export:
        """Everything held for one person, for an access request. Raises :class:`ExportTooLarge`
        rather than returning part of it."""
        if not data_subject_id or not data_subject_id.strip():
            raise ValueError("an export names the person it is for")
        return Export._from(await self._post("/v1/exports", {"data_subject_id": data_subject_id}, 200))

    async def export_sources(self, source_observation_ids: Sequence[str]) -> Export:
        """Everything held from the named observations."""
        ids = list(source_observation_ids)
        if not ids:
            raise ValueError("an export by source names at least one observation")
        return Export._from(await self._post("/v1/exports", {"source_observation_ids": ids}, 200))

    async def _erase(self, body: dict) -> ErasureReceipt:
        receipt = ErasureReceipt._from(await self._post("/v1/erasures", body, 200))
        if not receipt.clean:
            raise ErasureNotClean(receipt)
        return receipt

    async def _post(self, path: str, body: dict, success: int) -> dict:
        response = await self._http.post(self.base_url + path, json=body, headers=self._headers)
        return self._read(response, success)

    @staticmethod
    def _read(response: httpx.Response, success: int) -> dict:
        if response.status_code == success:
            return response.json()
        code, message = "unexpected_response", "the deployment answered " + str(response.status_code)
        try:
            error = response.json().get("error") or {}
            code = error.get("code") or code
            message = error.get("message") or message
        except ValueError:
            pass
        if response.status_code == 403 and code == "forbidden":
            raise Forbidden(response.status_code, code, message)
        if response.status_code == 422 and code == "export_too_large":
            raise ExportTooLarge(response.status_code, code, message)
        raise TaisceError(response.status_code, code, message)


def bundle_is_empty(bundle: Mapping[str, Any]) -> bool:
    """Whether a recall returned nothing at all: no facts, no reports, no passages."""
    return not (bundle.get("facts") or bundle.get("reports") or bundle.get("passages"))
