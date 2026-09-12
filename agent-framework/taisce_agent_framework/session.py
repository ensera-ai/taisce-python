# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
"""Keeping an agent session where the memory it belongs to already lives.

The framework serializes a session and hands it back; it ships no store, so every application
invents one. Inventing it in the application's own database is the expensive mistake, and the reason
is governance rather than convenience: a session and the memory formed from it are erased by the
same request from the same person. Split across two systems, a deletion has two places to sweep, one
of which can produce a counted residual and one of which cannot, and the honest answer to "is it
gone" becomes "it is gone from the part we can measure".

So a session is an opaque object under the person it belongs to. The deployment never reads it,
expires it with that person's retention, and removes it with that person's erasure.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class Saved:
    """What a save returned: the version the next save should expect to replace."""

    session_id: str
    version: str


@dataclass(frozen=True)
class Loaded:
    """What a load returned: the state the framework will deserialize, and its version."""

    state: Mapping[str, Any]
    version: str


class SessionStore:
    """A session store over one deployment's opaque objects.

    **Whose session it is, is checked by the deployment.** One credential opens a project and a
    project holds every end user's objects, so an application serving many people could hand one
    person's session to another; nothing in the application is a boundary against that, only care.
    Every call names the subject, and the deployment answers another person's session exactly as one
    that does not exist.

    **Concurrency is the caller's to declare.** A save carries the version it expects to replace and
    the deployment refuses a stale one, so two turns of the same session cannot silently lose one of
    their writes. Passing no version is last-write-wins, which is a choice and never the default.
    """

    def __init__(self, client: Any, *, kind: str = "state") -> None:
        if client is None:
            raise ValueError("a deployment is required")
        self._client = client
        self._kind = kind or "state"

    async def save(self, session_id: str, data_subject_id: str, state: Mapping[str, Any], *,
                   expected_version: Optional[str] = None) -> Saved:
        """Stores the framework's serialized session under the person it belongs to."""
        _require(session_id, data_subject_id)
        receipt = await self._client.put_artifact(
            artifact_id=session_id,
            data_subject_id=data_subject_id,
            kind=self._kind,
            name="session",
            content=json.dumps(state, separators=(",", ":")).encode("utf-8"),
            expected_version=expected_version,
        )
        return Saved(session_id=receipt.get("id", session_id), version=receipt.get("version", ""))

    async def load(self, session_id: str, data_subject_id: str) -> Optional[Loaded]:
        """Reads a person's session back, or ``None`` when there is none by that id for that person.

        ``None`` rather than an exception, because "this person has not been here before" is the
        ordinary first turn of every conversation and not an error anybody should have to catch.
        """
        _require(session_id, data_subject_id)
        try:
            stored = await self._client.get_artifact(artifact_id=session_id, data_subject_id=data_subject_id)
        except Exception as exc:  # noqa: BLE001 - only a 404 is an absence; everything else is the caller's
            if getattr(exc, "status", None) == 404:
                return None
            raise
        content = stored.get("content")
        raw = base64.b64decode(content) if isinstance(content, str) else bytes(content or b"")
        return Loaded(state=json.loads(raw.decode("utf-8")), version=stored.get("version", ""))

    async def delete(self, session_id: str, data_subject_id: str, *,
                     expected_version: Optional[str] = None) -> None:
        """Removes a person's session. One that is not there is already removed."""
        _require(session_id, data_subject_id)
        try:
            await self._client.delete_artifact(artifact_id=session_id, data_subject_id=data_subject_id,
                                               expected_version=expected_version)
        except Exception as exc:  # noqa: BLE001
            if getattr(exc, "status", None) != 404:
                raise


def _require(session_id: str, data_subject_id: str) -> None:
    """Both are required, and the subject most of all: without it the deployment cannot tell whose
    session this is, and the check that makes one person's state unreachable to another is the one
    thing this class exists to keep."""
    if not session_id or not session_id.strip():
        raise ValueError("a session needs an identity")
    if not data_subject_id or not data_subject_id.strip():
        raise ValueError("a session belongs to a person: without one, another person's session is reachable")
