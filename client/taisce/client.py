# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
"""The client: the v1 contract of a Taisce deployment, and nothing about any agent framework.

The adapters are built on this, so a caller who wants governed memory without a framework does not
acquire one by asking. It is asynchronous because the frameworks it serves are; the sync wrapper
exists for scripts. The credential is a bearer token bound to one project, sent nowhere but the
deployment this client was built for, and never logged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import base64

import httpx


class TaisceError(Exception):
    """A refusal from the deployment, carrying the contract's own code. Branch on ``code``."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message


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
        raise TaisceError(response.status_code, code, message)


def bundle_is_empty(bundle: Mapping[str, Any]) -> bool:
    """Whether a recall returned nothing at all: no facts, no reports, no passages."""
    return not (bundle.get("facts") or bundle.get("reports") or bundle.get("passages"))
