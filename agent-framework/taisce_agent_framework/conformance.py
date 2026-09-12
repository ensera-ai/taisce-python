# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
"""The conformance driver: the adapter wrapped in the subprocess protocol the suite speaks.

One JSON instruction per line on stdin, one report per line on stdout. The agent is the framework's
own ``Agent`` with the provider attached; the model is a stub that records what it was handed and
answers the turn's reply, or fails when the turn says so. Nothing about the deployment is stubbed.

    python -m taisce_agent_framework.conformance
"""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Mapping, Sequence

import httpx
from agent_framework import Agent, BaseChatClient, ChatResponse, Content, InMemoryHistoryProvider, Message

from taisce import Client
from taisce_agent_framework import TaisceCompaction, TaisceContextProvider, UNTRUSTED_PROPERTY


class StubModel(BaseChatClient):
    """Records what it was handed, answers the turn's reply or fails."""

    def __init__(self, turn: dict, report: dict) -> None:
        super().__init__()
        self._turn = turn
        self._report = report

    async def _inner_get_response(self, *, messages: Sequence[Message], stream: bool, options: Mapping[str, Any], **kwargs: Any) -> ChatResponse:
        self._report["model_messages"] = [
            {"role": _role(m), "content": m.text or "",
             "untrusted": bool((getattr(m, "additional_properties", None) or {}).get(UNTRUSTED_PROPERTY))}
            for m in messages
        ]
        if self._turn.get("model_failure"):
            raise RuntimeError("the model failed")
        response = []
        for i, call in enumerate(self._turn.get("tool_calls") or [], start=1):
            call_id = f"call-{i}"
            response.append(Message("assistant", [Content.from_function_call(call_id, "tool", arguments={"call": call["call"]})]))
            response.append(Message("tool", [Content.from_function_result(call_id, result=call["result"])]))
        if self._turn.get("assistant"):
            response.append(Message("assistant", [self._turn["assistant"]]))
        return ChatResponse(messages=response)


def _role(message: Any) -> str:
    role = getattr(message, "role", "")
    return getattr(role, "value", role) if not isinstance(role, str) else role


async def run_turn(instruction: dict) -> dict:
    report: dict = {"model_messages": [], "fatal": False, "observed": False}

    async def saw_request(request: httpx.Request) -> None:
        if request.method == "POST" and request.url.path.endswith("/observations"):
            report["observed"] = True

    http = httpx.AsyncClient(timeout=10.0, event_hooks={"request": [saw_request]})
    client = Client(instruction["api"], instruction["token"], http=http)

    def on_error(stage: str, exc: Exception) -> None:
        if stage == "observe":
            report["store_error"] = str(exc)

    provider = TaisceContextProvider(client, data_subject_id=instruction.get("data_subject_id"), run_id=instruction["case"], on_error=on_error)
    turn = instruction["turn"]
    # The framework's own history provider holds the earlier turns the application kept; the
    # compaction provider, when the turn arms it, is triggered by any history at all.
    history = InMemoryHistoryProvider()
    providers: list = [history, provider]
    if turn.get("compact"):
        providers.insert(1, TaisceCompaction(client, data_subject_id=instruction["data_subject_id"], after_messages=1, on_error=on_error))
    agent = Agent(StubModel(turn, report), context_providers=providers)
    session = agent.create_session()
    if turn.get("history"):
        await history.save_messages(session.session_id, [Message(m["role"], [m["content"]]) for m in turn["history"]],
                                    state=session.state.setdefault(history.source_id, {}))
    messages = [Message(s["role"], [s["content"]]) for s in turn.get("synthetic") or []]
    messages.append(Message("user", [turn["user"]]))
    try:
        await agent.run(messages, session=session)
    except Exception:  # noqa: BLE001 - the run's error reaches the application; which it was is in the report
        report["fatal"] = "store_error" not in report
    finally:
        await http.aclose()
    return report


async def main() -> int:
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            return 0
        if not line.strip():
            continue
        report = await run_turn(json.loads(line))
        sys.stdout.write(json.dumps(report) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
