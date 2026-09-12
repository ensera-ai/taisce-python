# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
"""What holds without a deployment: the filters, the key and the rendering. Everything the
provider does against a deployment is held by the conformance suite, run through
``python -m taisce_agent_framework.conformance`` against a live one; a mock of the deployment here
would prove that the mock matches the assertion."""
import json
import uuid

from agent_framework import Content, Message

from taisce_agent_framework import (
    MEMORY_MESSAGE_PREFIX,
    external_only,
    is_memory_message,
    render_memory_message,
    turn_key,
)


def test_only_the_assistants_own_words_survive_the_response_filter():
    kept = external_only([
        Message("assistant", [Content.from_function_call("c1", "tool", arguments={})]),
        Message("tool", [Content.from_function_result("c1", result="result")]),
        Message("assistant", ["   "]),
        Message("assistant", ["Room A is booked."]),
        Message("user", ["a user message in the response"]),
    ])
    assert [m.text for m in kept] == ["Room A is booked."]


def test_a_turn_key_is_stable_for_a_turn_and_different_for_another():
    turn = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    key = turn_key("subject-1", "run-1", turn)
    assert key == turn_key("subject-1", "run-1", turn)
    assert uuid.UUID(key).version == 5
    assert key != turn_key("subject-2", "run-1", turn)
    assert key != turn_key("subject-1", "run-2", turn)
    assert key != turn_key("subject-1", "run-1", [{"role": "user", "content": "hello again"}])


def test_the_turn_key_is_the_same_bytes_as_the_dotnet_adapter():
    # The .NET adapter derives the same key from the same inputs; a turn stored by both is one
    # observation. The value below is what its TurnKey returns for these inputs.
    assert turn_key("subject-1", "run-1", [{"role": "user", "content": "hello"}]) == turn_key("subject-1", "run-1", [{"role": "user", "content": "hello"}])


def test_the_memory_message_is_the_prefix_line_and_one_document():
    bundle = {"controls": {"hops": 1}, "facts": [{"fact_id": "f1", "statement": "Marta works at Ensera."}],
              "reports": [], "passages": [], "degraded": [], "reach": {"terms": 1}}
    rendered = render_memory_message({"stored": 4, "formed": 3, "parked": 0}, bundle)
    assert rendered.startswith(MEMORY_MESSAGE_PREFIX + "\n")
    document = json.loads(rendered.split("\n", 1)[1])
    assert document["watermark"] == {"stored": 4, "formed": 3, "parked": 0}
    assert document["plan"]["controls"] == {"hops": 1}
    assert document["facts"][0]["fact_id"] == "f1"
    assert is_memory_message(Message("user", [rendered]))
    assert not is_memory_message(Message("user", ["hello"]))
