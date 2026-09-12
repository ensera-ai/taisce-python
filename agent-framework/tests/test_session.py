# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
#
# A session belongs to a person, and saying so is not optional. One credential opens a project and a
# project holds every end user's objects, so a store that forgets whose session it is makes one
# person's state reachable to another — and no amount of care in the application is a boundary.
import base64
import json

import pytest

from taisce_agent_framework import SessionStore


class Deployment:
    def __init__(self, stored=None, status=None):
        self.calls = []
        self._stored = stored
        self._status = status

    async def put_artifact(self, **kwargs):
        self.calls.append(("put", kwargs))
        return {"id": kwargs["artifact_id"], "version": "v2"}

    async def get_artifact(self, **kwargs):
        self.calls.append(("get", kwargs))
        if self._status is not None:
            raise _Refused(self._status)
        return {"version": "v1", "content": base64.b64encode(json.dumps(self._stored).encode()).decode()}

    async def delete_artifact(self, **kwargs):
        self.calls.append(("delete", kwargs))
        if self._status is not None:
            raise _Refused(self._status)
        return {"deleted": True}


class _Refused(Exception):
    def __init__(self, status):
        super().__init__(str(status))
        self.status = status


@pytest.mark.asyncio
async def test_every_call_names_the_person_and_a_save_declares_what_it_replaces():
    deployment = Deployment(stored={"messages": []})
    store = SessionStore(deployment)

    saved = await store.save("s1", "marta", {"messages": [{"role": "user"}]}, expected_version="v1")
    assert saved.version == "v2"
    kind, put = deployment.calls[0]
    assert kind == "put"
    assert put["data_subject_id"] == "marta" and put["expected_version"] == "v1"
    assert json.loads(put["content"].decode()) == {"messages": [{"role": "user"}]}

    loaded = await store.load("s1", "marta")
    assert loaded.state == {"messages": []} and loaded.version == "v1"
    assert deployment.calls[1][1]["data_subject_id"] == "marta"

    await store.delete("s1", "marta", expected_version="v2")
    assert deployment.calls[2][1]["data_subject_id"] == "marta"


@pytest.mark.asyncio
async def test_a_session_without_a_person_never_reaches_the_deployment():
    deployment = Deployment(stored={})
    store = SessionStore(deployment)
    for call in (store.save("s1", " ", {}), store.save(" ", "marta", {}),
                 store.load("s1", ""), store.delete("s1", "")):
        with pytest.raises(ValueError):
            await call
    assert deployment.calls == []


@pytest.mark.asyncio
async def test_a_person_who_has_not_been_here_before_is_not_an_error():
    store = SessionStore(Deployment(status=404))
    assert await store.load("s1", "marta") is None
    # Nor is deleting what is already gone.
    await store.delete("s1", "marta")
    # Anything else is the caller's to handle.
    with pytest.raises(_Refused):
        await SessionStore(Deployment(status=500)).load("s1", "marta")
