# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
"""What holds without a deployment: the refusals at construction, and what a compaction keeps and
replaces in a loaded history. The rest is held by the conformance suite."""
import pytest
from agent_framework import Message, SessionContext

from taisce import Client, MEMORY_MESSAGE_PREFIX
from taisce_agent_framework import TaisceCompaction, UNTRUSTED_PROPERTY


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
        TaisceCompaction(client, data_subject_id="", after_messages=8)
    with pytest.raises(ValueError):
        TaisceCompaction(client, data_subject_id="s", after_messages=0)
    assert TaisceCompaction(client, data_subject_id="s", after_messages=8).source_id == "taisce-compaction"


@pytest.mark.asyncio
async def test_a_loaded_history_over_the_trigger_is_replaced_by_the_context_and_system_and_memory_messages_stay():
    memory = Message("user", [MEMORY_MESSAGE_PREFIX + "\n{}"], additional_properties={UNTRUSTED_PROPERTY: True})
    loaded = {"in_memory": [Message("system", ["Be brief."]), Message("user", ["old one"]), Message("assistant", ["old reply"])], "taisce": [memory]}
    context = SessionContext(input_messages=[Message("user", ["now?"])], context_messages={k: list(v) for k, v in loaded.items()})
    client = _ContextClient(answer={"watermark": {"stored": 3}, "segments": [{"summary": "It began."}], "turns": [], "characters": 9, "truncated": False})
    await TaisceCompaction(client, data_subject_id="s", after_messages=1).before_run(agent=None, session=None, context=context, state={})
    texts = [m.text for m in context.get_messages()]
    assert texts[0] == "Be brief." and texts[1].startswith(MEMORY_MESSAGE_PREFIX) and "old one" not in texts and "old reply" not in texts
    assert texts[-1].startswith(MEMORY_MESSAGE_PREFIX) and '"It began."' in texts[-1]
    assert context.get_messages()[-1].additional_properties[UNTRUSTED_PROPERTY] is True
    # Under the trigger nothing happens; a failing deployment leaves the history as it was.
    untouched = SessionContext(input_messages=[Message("user", ["now?"])], context_messages={"in_memory": [Message("user", ["old one"])]})
    idle = _ContextClient()
    await TaisceCompaction(idle, data_subject_id="s", after_messages=4).before_run(agent=None, session=None, context=untouched, state={})
    assert idle.calls == 0 and [m.text for m in untouched.get_messages()] == ["old one"]
    seen = []
    broken = _ContextClient(error=RuntimeError("down"))
    provider = TaisceCompaction(broken, data_subject_id="s", after_messages=1, on_error=lambda stage, exc: seen.append(stage))
    failing = SessionContext(input_messages=[Message("user", ["now?"])], context_messages={"in_memory": [Message("user", ["old one"]), Message("assistant", ["old reply"])]})
    await provider.before_run(agent=None, session=None, context=failing, state={})
    assert seen == ["context"] and [m.text for m in failing.get_messages()] == ["old one", "old reply"]
