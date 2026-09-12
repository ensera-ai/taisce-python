# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
"""The context provider: recall before a turn, observe after it.

**The role rule, and why Python makes it sharper.** ``SessionContext`` gives a provider two ways to
add content: ``extend_messages``, which contributes conversation messages, and
``extend_instructions``, which contributes system instructions. The framework validates neither, and
its own documentation names an external source as the path an indirect prompt injection arrives
through. Every memory is something somebody said, so every memory is untrusted input. This provider
uses ``extend_messages`` with role ``user``, always, marked untrusted, and never touches
``extend_instructions``: the same bytes as an instruction are something the model obeys.

**Failure policy is asymmetric.** A recall that cannot reach the deployment contributes nothing and
raises nothing; the agent runs without memory rather than not at all, and ``on_error`` is how an
operator still finds out. A failed store raises, because losing a turn silently is the failure that
stays invisible until a subject access request comes back missing a conversation.

**Store only on success, and only what people said.** The framework runs ``after_run`` on the
success path; the check on ``context.response`` stays because relying on the absence of a call is
relying on a framework internal. What is stored is the person's messages and the assistant's final
text; tool calls, tool results and anything this provider injected are not memory.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Iterable, List, Optional, Sequence

from agent_framework import ContextProvider, Message

from taisce import Client, bundle_is_empty
from taisce.memory import MEMORY_MESSAGE_PREFIX, is_memory_text, render_memory_message, turn_key

#: The key under which the injected message is marked untrusted in its additional properties.
UNTRUSTED_PROPERTY = "taisce.untrusted"
#: The source id this provider reports to the framework; its own messages are keyed by it.
SOURCE_ID = "taisce"


class TaisceContextProvider(ContextProvider):
    """Injects governed memory into an agent's turn, and records the turn afterwards."""

    def __init__(
        self,
        client: Client,
        *,
        data_subject_id: Optional[str] = None,
        run_id: Optional[str] = None,
        max_characters: Optional[int] = None,
        source_roles: Optional[Sequence[str]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
        source_id: str = SOURCE_ID,
    ) -> None:
        super().__init__(source_id=source_id)
        self._client = client
        self._data_subject_id = data_subject_id
        self._run_id = run_id
        self._max_characters = max_characters
        self._source_roles = list(source_roles) if source_roles else None
        self._on_error = on_error

    async def before_run(self, *, agent: Any, session: Any, context: Any, state: dict) -> None:
        question = _last_user_text(context.get_messages(include_input=True))
        if not question:
            return
        try:
            watermark = await self._client.freshness()
            bundle = await self._client.recall(
                question=question,
                data_subject_id=self._data_subject_id,
                max_characters=self._max_characters,
                source_roles=self._source_roles,
            )
        except Exception as exc:  # noqa: BLE001 - swallowed by design, see the module docstring
            self._report("recall", exc)
            return
        if bundle_is_empty(bundle):
            return
        rendered = render_memory_message(
            {"stored": watermark.stored, "formed": watermark.formed, "parked": watermark.parked}, bundle)
        # THE line: extend_messages with role user, never extend_instructions. The text is wrapped
        # in a list: Message takes a sequence of contents and a bare str is a sequence of letters.
        context.extend_messages(self.source_id, [
            Message("user", [rendered], author_name="taisce", additional_properties={UNTRUSTED_PROPERTY: True}),
        ])

    async def after_run(self, *, agent: Any, session: Any, context: Any, state: dict) -> None:
        if context.response is None:
            return
        messages: List[dict] = []
        # The turn's input, not the history a history provider loaded beside it: earlier turns
        # were observed when they happened, and observing them again would store each twice.
        for m in getattr(context, "input_messages", None) or []:
            if _role(m) == "user" and not is_memory_message(m) and (m.text or "").strip():
                messages.append({"role": "user", "content": m.text})
        for m in external_only(getattr(context.response, "messages", None) or []):
            messages.append({"role": "assistant", "content": m.text})
        if not messages:
            return
        try:
            await self._client.observe(
                idempotency_key=turn_key(self._data_subject_id, self._run_id, messages),
                messages=messages,
                data_subject_id=self._data_subject_id,
                occurred_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
        except Exception as exc:  # noqa: BLE001 - reported, then raised
            self._report("observe", exc)
            raise

    def _report(self, stage: str, exc: Exception) -> None:
        if self._on_error is not None:
            self._on_error(stage, exc)


def external_only(messages: Iterable[Any]) -> List[Any]:
    """Only the assistant's own words: a message carrying a function call is the model talking to
    a tool, and a tool's result is the tool talking back. Neither is something a person said."""
    out = []
    for m in messages:
        if _role(m) != "assistant":
            continue
        if any(_content_type(c) in ("function_call", "function_result") for c in (getattr(m, "contents", None) or [])):
            continue
        if not (m.text or "").strip():
            continue
        out.append(m)
    return out


def is_memory_message(message: Any) -> bool:
    """Whether a message is one this provider injected, by its first line."""
    return is_memory_text(getattr(message, "text", "") or "")


def _role(message: Any) -> str:
    role = getattr(message, "role", "")
    return getattr(role, "value", role) if not isinstance(role, str) else role


def _content_type(content: Any) -> str:
    return str(getattr(content, "type", "") or "")


def _last_user_text(messages: Sequence[Any]) -> str:
    for m in reversed(list(messages)):
        if _role(m) == "user" and not is_memory_message(m):
            text = (getattr(m, "text", "") or "").strip()
            if text:
                return text
    return ""
