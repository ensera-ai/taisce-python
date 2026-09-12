# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from taisce import Client, MEMORY_MESSAGE_PREFIX, is_memory_text, render_context_message


def test_the_context_message_is_the_prefix_line_and_the_deployments_arrays_unchanged():
    context = {"watermark": {"stored": 12, "formed": 12}, "characters": 140, "truncated": False,
               "segments": [{"level": 1, "summary": "The office moved."}],
               "turns": [{"log_offset": 9, "messages": [{"role": "user", "content": "Book it."}]}]}
    rendered = render_context_message(context)
    assert rendered.startswith(MEMORY_MESSAGE_PREFIX + "\n") and is_memory_text(rendered)
    document = json.loads(rendered.split("\n", 1)[1])
    assert document["watermark"] == {"stored": 12, "formed": 12, "parked": 0}
    assert document["plan"] == {"characters": 140, "truncated": False}
    assert document["segments"] == context["segments"] and document["turns"] == context["turns"]
    # A context with nothing in it renders empty arrays, never a missing field.
    bare = json.loads(render_context_message({}).split("\n", 1)[1])
    assert bare["segments"] == [] and bare["turns"] == [] and bare["watermark"]["stored"] is None


@pytest.mark.asyncio
async def test_a_context_needs_a_subject():
    async with Client("http://127.0.0.1:1", "tsk") as client:
        with pytest.raises(ValueError):
            await client.context(data_subject_id=" ")
