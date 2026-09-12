# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
"""What holds without a deployment: which messages of a run become the turn. Everything the
middleware does against a deployment is held by the conformance suite through
``python -m taisce_langgraph.conformance``."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from taisce.memory import render_memory_message
from taisce_langgraph import turn_messages


def test_the_turn_is_the_last_human_message_and_the_assistants_own_words_after_it():
    memory = render_memory_message({"stored": 1}, {"facts": [{"fact_id": "f1"}]})
    messages = [
        AIMessage(content="Summary of earlier conversation."),
        HumanMessage(content="Book the room for Tuesday."),
        HumanMessage(content=memory),
        AIMessage(content="", tool_calls=[{"name": "act", "args": {"call": "book"}, "id": "c1"}]),
        ToolMessage(content="booked", tool_call_id="c1"),
        AIMessage(content="Room A is booked for Tuesday."),
    ]
    assert turn_messages(messages) == [
        {"role": "user", "content": "Book the room for Tuesday."},
        {"role": "assistant", "content": "Room A is booked for Tuesday."},
    ]


def test_a_run_with_no_human_message_is_no_turn():
    assert turn_messages([AIMessage(content="hello")]) == []
    assert turn_messages([]) == []
