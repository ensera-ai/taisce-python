# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
"""The memory middleware: recall before the model call, observe after the agent run.

LangGraph's agent has no context-provider interface; what it has is middleware around the model
call and around the agent run, and that is the seam this maps onto, with nothing of its own.

**The role rule.** Memory is injected into the model request as one ``HumanMessage`` marked
untrusted in its ``additional_kwargs``, never as a ``SystemMessage``: the same bytes as a system
message are instructions the model obeys, and every memory is something somebody said once. It is
injected into the request only, not into the graph's state, so it is never persisted as history
and never stored as though a person said it.

**Failure policy is asymmetric.** A recall that cannot reach the deployment injects nothing and
raises nothing; the agent runs without memory. A failed store raises, because a lost turn is
invisible until a subject access request asks for it.

**Store only on success, and only what people said.** ``aafter_agent`` runs when the agent run
completed; a model that raised never reaches it. What is stored is the last human message and the
assistant messages after it that carry text and no tool calls; tool messages, tool-calling
messages and anything before the human message, which the framework or an earlier turn wrote, are
not this turn's memory.

**Compaction is replacement, in the state, the way the framework's own summarisation middleware
works.** When ``compact_after_messages`` is set and the state holds more non-system messages than
that, ``abefore_model`` asks the deployment for the subject's context and rewrites the state with
the framework's remove-all message: the system messages, one untrusted ``HumanMessage`` carrying
the context unchanged, and the person's current message. Nothing is summarised here; a context
that cannot be fetched leaves the state as it was and reports.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, List, Optional, Sequence

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from taisce import Client, bundle_is_empty
from taisce.memory import is_memory_text, render_context_message, render_memory_message, turn_key

#: The key under which the injected message is marked untrusted in its additional kwargs.
UNTRUSTED_KEY = "taisce.untrusted"


class TaisceMemory(AgentMiddleware):
    """Injects governed memory before each model call and records the turn after the run."""

    def __init__(
        self,
        client: Client,
        *,
        data_subject_id: Optional[str] = None,
        run_id: Optional[str] = None,
        max_characters: Optional[int] = None,
        source_roles: Optional[Sequence[str]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
        compact_after_messages: Optional[int] = None,
    ) -> None:
        super().__init__()
        if compact_after_messages is not None:
            if compact_after_messages <= 0:
                raise ValueError("compact_after_messages must be positive: it is the history length that triggers compaction")
            if not data_subject_id or not data_subject_id.strip():
                raise ValueError("a data subject is required to compact: a context is one subject's history")
        self._client = client
        self._data_subject_id = data_subject_id
        self._run_id = run_id
        self._max_characters = max_characters
        self._source_roles = list(source_roles) if source_roles else None
        self._on_error = on_error
        self._compact_after = compact_after_messages

    async def abefore_model(self, state: Any, runtime: Any) -> Optional[dict]:
        if self._compact_after is None:
            return None
        messages = list(state.get("messages") or [])
        if sum(1 for m in messages if not isinstance(m, SystemMessage)) <= self._compact_after:
            return None
        at = _last_human_index(messages)
        if at >= len(messages):
            return None
        try:
            assembled = await self._client.context(data_subject_id=self._data_subject_id, max_characters=self._max_characters)
        except Exception as exc:  # noqa: BLE001 - swallowed by design, see the module docstring
            self._report("context", exc)
            return None
        memory = HumanMessage(content=render_context_message(assembled), name="taisce", additional_kwargs={UNTRUSTED_KEY: True})
        kept = [m for m in messages[:at] if isinstance(m, SystemMessage)]
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept, memory, *messages[at:]]}

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        question = _last_human_text(request.messages)
        if not question:
            return await handler(request)
        try:
            watermark = await self._client.freshness()
            bundle = await self._client.recall(
                question=question, data_subject_id=self._data_subject_id,
                max_characters=self._max_characters, source_roles=self._source_roles)
        except Exception as exc:  # noqa: BLE001 - swallowed by design, see the module docstring
            self._report("recall", exc)
            return await handler(request)
        if bundle_is_empty(bundle):
            return await handler(request)
        rendered = render_memory_message(
            {"stored": watermark.stored, "formed": watermark.formed, "parked": watermark.parked}, bundle)
        memory = HumanMessage(content=rendered, name="taisce", additional_kwargs={UNTRUSTED_KEY: True})
        # Before the person's last message, so the model reads memory and then the question.
        messages = list(request.messages)
        at = _last_human_index(messages)
        messages.insert(at, memory)
        return await handler(request.override(messages=messages))

    async def aafter_agent(self, state: Any, runtime: Any) -> Optional[dict]:
        messages = turn_messages(state.get("messages") or [])
        if not messages:
            return None
        try:
            await self._client.observe(
                idempotency_key=turn_key(self._data_subject_id, self._run_id, messages),
                messages=messages, data_subject_id=self._data_subject_id,
                occurred_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        except Exception as exc:  # noqa: BLE001 - reported, then raised
            self._report("observe", exc)
            raise
        return None

    def _report(self, stage: str, exc: Exception) -> None:
        if self._on_error is not None:
            self._on_error(stage, exc)


def turn_messages(messages: Sequence[BaseMessage]) -> List[dict]:
    """The turn as the deployment stores it: the last human message and the assistant's own words
    after it. A tool message is the tool talking; an assistant message with tool calls is the
    model talking to a tool; neither is something a person said."""
    at = _last_human_index(list(messages))
    if at >= len(messages):
        return []
    out = [{"role": "user", "content": _text(messages[at])}]
    for m in messages[at + 1:]:
        if isinstance(m, AIMessage) and not m.tool_calls and _text(m).strip():
            out.append({"role": "assistant", "content": _text(m)})
    return out


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)


def _last_human_index(messages: List[BaseMessage]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, HumanMessage) and not is_memory_text(_text(m)) and _text(m).strip():
            return i
    return len(messages)


def _last_human_text(messages: Sequence[BaseMessage]) -> str:
    at = _last_human_index(list(messages))
    return _text(messages[at]).strip() if at < len(messages) else ""
