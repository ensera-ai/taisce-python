# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
"""What holds without a deployment: the state rewrite a compaction returns. The rest is held by the
conformance suite."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from taisce import Client, MEMORY_MESSAGE_PREFIX
from taisce_langgraph import TaisceMemory, UNTRUSTED_KEY


class _ContextClient:
    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.calls = answer, error, 0

    async def context(self, *, data_subject_id, max_characters=None):
        self.calls += 1
        if self.error:
            raise self.error
        return self.answer


def test_compaction_needs_a_subject_and_a_positive_trigger():
    client = Client("http://127.0.0.1:1", "tsk")
    with pytest.raises(ValueError):
        TaisceMemory(client, compact_after_messages=8)
    with pytest.raises(ValueError):
        TaisceMemory(client, data_subject_id="s", compact_after_messages=0)
    assert TaisceMemory(client, data_subject_id="s", compact_after_messages=8) is not None


@pytest.mark.asyncio
async def test_the_state_is_rewritten_as_system_messages_the_context_and_the_persons_message():
    history = [SystemMessage(content="Be brief."), HumanMessage(content="old one"), AIMessage(content="old reply"), HumanMessage(content="now?")]
    client = _ContextClient(answer={"watermark": {"stored": 3}, "segments": [{"summary": "It began."}], "turns": [], "characters": 9, "truncated": False})
    update = await TaisceMemory(client, data_subject_id="s", compact_after_messages=1).abefore_model({"messages": history}, None)
    messages = update["messages"]
    assert isinstance(messages[0], RemoveMessage) and messages[0].id == REMOVE_ALL_MESSAGES
    assert messages[1].content == "Be brief."
    assert isinstance(messages[2], HumanMessage) and messages[2].content.startswith(MEMORY_MESSAGE_PREFIX) and messages[2].additional_kwargs[UNTRUSTED_KEY] is True
    assert messages[3].content == "now?" and len(messages) == 4
    # Under the trigger, or with no person's message, nothing is rewritten; a failing deployment leaves the state as it was.
    idle = _ContextClient()
    assert await TaisceMemory(idle, data_subject_id="s", compact_after_messages=8).abefore_model({"messages": history}, None) is None
    assert await TaisceMemory(idle, data_subject_id="s", compact_after_messages=1).abefore_model({"messages": [AIMessage(content="a"), AIMessage(content="b")]}, None) is None
    assert idle.calls == 0
    seen = []
    broken = _ContextClient(error=RuntimeError("down"))
    assert await TaisceMemory(broken, data_subject_id="s", compact_after_messages=1, on_error=lambda stage, exc: seen.append(stage)).abefore_model({"messages": history}, None) is None
    assert seen == ["context"]
