# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
"""The conformance driver: the middleware wrapped in the subprocess protocol the suite speaks.

The agent is LangGraph's own ``create_agent`` with the middleware attached and one tool; the model
is a scripted chat model that records what it was handed and answers the turn's reply, calling the
tool first when the turn has tool calls, or fails when the turn says so.

    python -m taisce_langgraph.conformance
"""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, List, Optional

import httpx
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import PrivateAttr

from taisce import Client
from taisce_langgraph import UNTRUSTED_KEY, TaisceMemory


@tool
def act(call: str) -> str:
    """Performs the call the turn scripted and answers its result."""
    return _RESULTS.get(call, "")


_RESULTS: dict = {}


class ScriptedModel(BaseChatModel):
    """Records what it was handed on the first call, then follows the turn's script.

    The report is a private attribute rather than a field: a field holding a dict is validated
    into a copy, and a copy is where the record of what the model saw would silently go."""

    turn: dict
    _report: dict = PrivateAttr(default_factory=dict)
    _calls: int = PrivateAttr(default=0)

    def attach(self, report: dict) -> "ScriptedModel":
        self._report = report
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        if self._calls == 0:
            self._report["model_messages"] = [
                {"role": _role(m), "content": _text(m), "untrusted": bool((m.additional_kwargs or {}).get(UNTRUSTED_KEY))}
                for m in messages
            ]
        self._calls += 1
        if self.turn.get("model_failure"):
            raise RuntimeError("the model failed")
        tool_calls = self.turn.get("tool_calls") or []
        if self._calls <= len(tool_calls):
            call = tool_calls[self._calls - 1]
            message = AIMessage(content="", tool_calls=[{"name": "act", "args": {"call": call["call"]}, "id": f"call-{self._calls}"}])
        else:
            message = AIMessage(content=self.turn.get("assistant") or "")
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self


def _role(m: BaseMessage) -> str:
    return {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}.get(m.type, m.type)


def _text(m: BaseMessage) -> str:
    return m.content if isinstance(m.content, str) else "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in m.content)


async def run_turn(instruction: dict) -> dict:
    report: dict = {"model_messages": [], "fatal": False, "observed": False}
    turn = instruction["turn"]
    _RESULTS.clear()
    for call in turn.get("tool_calls") or []:
        _RESULTS[call["call"]] = call["result"]

    async def saw_request(request: httpx.Request) -> None:
        if request.method == "POST" and request.url.path.endswith("/observations"):
            report["observed"] = True

    http = httpx.AsyncClient(timeout=10.0, event_hooks={"request": [saw_request]})
    client = Client(instruction["api"], instruction["token"], http=http)

    def on_error(stage: str, exc: Exception) -> None:
        # Stderr is the driver's own: a swallowed recall failure is still worth a line for a person.
        sys.stderr.write(f"{instruction['case']}: {stage} failed: {exc!r}\n")
        if stage == "observe":
            report["store_error"] = str(exc)

    # When the turn arms compaction, any history at all trips it.
    memory = TaisceMemory(client, data_subject_id=instruction.get("data_subject_id"), run_id=instruction["case"], on_error=on_error,
                          compact_after_messages=1 if turn.get("compact") else None)
    model = ScriptedModel(turn=turn).attach(report)
    agent = create_agent(model, tools=[act], middleware=[memory])
    # The history is the state the application's graph already holds; here it enters with the
    # turn's input, which is where LangGraph puts an existing conversation.
    messages: List[BaseMessage] = [AIMessage(content=m["content"]) if m["role"] == "assistant" else HumanMessage(content=m["content"])
                                   for m in (turn.get("history") or []) + (turn.get("synthetic") or [])]
    messages.append(HumanMessage(content=turn["user"]))
    try:
        await agent.ainvoke({"messages": messages})
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
