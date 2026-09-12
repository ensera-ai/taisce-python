# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
"""What every adapter renders the same way: the memory message and the turn key.

These live in the client so that the two Python adapters share one implementation, and so that
the bytes agree with the .NET adapter's, which derives them the same way: a turn stored by two
adapters is one observation, and a memory message from either parses the same.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Mapping, Optional, Sequence

#: The first line of every injected memory message; the conformance suite parses it.
MEMORY_MESSAGE_PREFIX = "taisce-memory/v1 untrusted"


def render_memory_message(watermark: Mapping[str, object], bundle: Mapping[str, object]) -> str:
    """The prefix line, then a JSON document with the watermark, the plan (what the recall did)
    and the recall's own arrays unchanged."""
    document = {
        "watermark": {"stored": watermark.get("stored"), "formed": watermark.get("formed"), "parked": watermark.get("parked", 0)},
        "plan": {"controls": bundle.get("controls"), "degraded": bundle.get("degraded") or [], "reach": bundle.get("reach")},
        "facts": bundle.get("facts") or [],
        "reports": bundle.get("reports") or [],
        "passages": bundle.get("passages") or [],
    }
    return MEMORY_MESSAGE_PREFIX + "\n" + json.dumps(document, separators=(",", ":"))


def render_context_message(context: Mapping[str, object]) -> str:
    """The prefix line, then a JSON document with the watermark, the assembly's cost as the plan,
    and the context's segments and turns unchanged: what replaces a history is the deployment's
    answer and nothing an adapter wrote."""
    watermark = context.get("watermark") or {}
    if not isinstance(watermark, Mapping):
        watermark = {}
    document = {
        "watermark": {"stored": watermark.get("stored"), "formed": watermark.get("formed"), "parked": watermark.get("parked", 0)},
        "plan": {"characters": context.get("characters") or 0, "truncated": bool(context.get("truncated"))},
        "segments": context.get("segments") or [],
        "turns": context.get("turns") or [],
    }
    return MEMORY_MESSAGE_PREFIX + "\n" + json.dumps(document, separators=(",", ":"))


def is_memory_text(text: Optional[str]) -> bool:
    """Whether a message's text is one an adapter injected, by its first line."""
    return bool(text) and text.startswith(MEMORY_MESSAGE_PREFIX)


def turn_key(data_subject_id: Optional[str], run_id: Optional[str], messages: Sequence[Mapping[str, str]]) -> str:
    """A turn's key: a version-5 UUID over the subject, the run and the messages, so a retried
    store is one observation and a different turn is another."""
    text = (data_subject_id or "") + "\n" + (run_id or "") + "\n"
    for m in messages:
        text += m["role"] + "\n" + m["content"] + "\n"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = bytearray(digest[:16])
    raw[6] = (raw[6] & 0x0F) | 0x50
    raw[8] = (raw[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(raw)))
