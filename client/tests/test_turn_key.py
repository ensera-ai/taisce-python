# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
"""The turn key is the conformance suite's derivation, byte for byte.

testdata/turn-keys.json is the suite's vector file, copied unchanged. Agreement with it is what makes
a turn retried through this adapter fold into the observation its first attempt made, and what makes
every adapter's keys the same keys.
"""
import json
import pathlib

from taisce import turn_key
from taisce.memory import TURN_KEY_DERIVATION

VECTORS = json.loads((pathlib.Path(__file__).parent / "testdata" / "turn-keys.json").read_text(encoding="utf-8"))


def test_every_published_vector_derives_its_key():
    assert VECTORS["derivation"] == TURN_KEY_DERIVATION
    for v in VECTORS["vectors"]:
        assert turn_key(v["data_subject_id"], v["run"], v["messages"]) == v["key"], v["name"]


def test_two_turns_whose_line_joined_text_is_identical_get_different_keys():
    first, second = VECTORS["vectors"][0], VECTORS["vectors"][1]
    joined = lambda v: "".join(m["role"] + "\n" + m["content"] + "\n" for m in v["messages"])
    assert joined(first) == joined(second)
    assert turn_key(first["data_subject_id"], first["run"], first["messages"]) != turn_key(second["data_subject_id"], second["run"], second["messages"])


def test_an_absent_subject_or_run_is_the_empty_string():
    turn = [{"role": "user", "content": "hello"}]
    assert turn_key(None, None, turn) == turn_key("", "", turn)
