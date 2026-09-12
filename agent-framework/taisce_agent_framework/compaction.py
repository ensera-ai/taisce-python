# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
"""The compaction provider: when the loaded history is long, replace it with the deployment's context.

The framework's own compaction seam is a strategy that marks loaded messages excluded; a strategy
cannot add a message the run keeps, so this is a context provider of its own that does what the
framework's ``CompactionProvider`` does to the loaded history and then contributes one message. It
decides nothing about what the context holds: it asks ``POST /v1/contexts`` and hands the answer
over unchanged, as one ``user`` message marked untrusted, in place of every loaded message that is
not a system message or a memory message another provider injected.

**Why replacement and not a summary.** A summary written here would be prose the deployment cannot
register, erase or re-derive, in a language-specific way, entering the history as though somebody
said it. The deployment holds every turn already and has rolled the older ones up off the read
path; its answer is the history, and the adapter's only job is fidelity to it.

**Failure policy.** A context that cannot be fetched leaves the loaded history as it was and reports
through ``on_error``; the agent runs with more to read, not less, and the turn is not fatal.

**The trigger** is a count of loaded non-system messages, because that is what this framework's
Python SDK counts by. The stored history is the application's and is not rewritten; once
over the trigger, every run fetches the context once.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from agent_framework import ContextProvider, Message

from taisce import Client
from taisce.memory import render_context_message

from .context import UNTRUSTED_PROPERTY, _role, is_memory_message

#: The source id this provider reports to the framework.
SOURCE_ID = "taisce-compaction"


class TaisceCompaction(ContextProvider):
    """Replaces a long loaded history with the deployment's context, before the model runs."""

    def __init__(
        self,
        client: Client,
        *,
        data_subject_id: str,
        after_messages: int,
        max_characters: Optional[int] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
        source_id: str = SOURCE_ID,
    ) -> None:
        if not data_subject_id or not data_subject_id.strip():
            raise ValueError("a data subject is required: a context is one subject's history")
        if after_messages <= 0:
            raise ValueError("after_messages must be positive: it is the history length that triggers compaction")
        super().__init__(source_id=source_id)
        self._client = client
        self._data_subject_id = data_subject_id
        self._after_messages = after_messages
        self._max_characters = max_characters
        self._on_error = on_error

    async def before_run(self, *, agent: Any, session: Any, context: Any, state: dict) -> None:
        loaded = context.get_messages()
        if sum(1 for m in loaded if not _kept(m)) <= self._after_messages:
            return
        try:
            assembled = await self._client.context(data_subject_id=self._data_subject_id, max_characters=self._max_characters)
        except Exception as exc:  # noqa: BLE001 - swallowed by design, see the module docstring
            self._report("context", exc)
            return
        rendered = render_context_message(assembled)
        for source in list(context.context_messages):
            context.context_messages[source] = [m for m in context.context_messages[source] if _kept(m)]
        context.extend_messages(self.source_id, [
            Message("user", [rendered], author_name="taisce", additional_properties={UNTRUSTED_PROPERTY: True}),
        ])

    def _report(self, stage: str, exc: Exception) -> None:
        if self._on_error is not None:
            self._on_error(stage, exc)


def _kept(message: Any) -> bool:
    """A system message is the application's instructions and a memory message is another
    provider's contribution for this run; neither is history the deployment holds."""
    return _role(message) == "system" or is_memory_message(message)
