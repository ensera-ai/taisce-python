# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
"""Microsoft Agent Framework adapter for Taisce."""
from .compaction import TaisceCompaction
from .session import Loaded, Saved, SessionStore
from .text_search import APPROXIMATE_NOTICE, SearchResult, map_results, search
from .context import (
    MEMORY_MESSAGE_PREFIX,
    UNTRUSTED_PROPERTY,
    TaisceContextProvider,
    external_only,
    is_memory_message,
    render_memory_message,
    turn_key,
)

__all__ = ["TaisceCompaction",
    "APPROXIMATE_NOTICE",
    "SearchResult",
    "SessionStore",
    "Saved",
    "Loaded",
    "map_results",
    "search",
    
    "MEMORY_MESSAGE_PREFIX",
    "UNTRUSTED_PROPERTY",
    "TaisceContextProvider",
    "external_only",
    "is_memory_message",
    "render_memory_message",
    "turn_key",
]
