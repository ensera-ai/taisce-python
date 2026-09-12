# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
"""The on-ramp that asks nothing of a developer's beliefs.

The context provider is the product, and it asks something: that a memory service should be forming
facts from conversations. Many developers do not accept that yet, and saying it louder is not a
strategy. What they do want is grounding — a search over what was actually said, returned with
enough identity to cite — and the deployment already serves exactly that.

So this maps a passage search onto the framework's search seam and adds nothing: no recall, no
storing, no memory message. A developer who starts here and later wants entities changes one line.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence

APPROXIMATE_NOTICE = (
    "These passages come from an index that is still being built, so they are some of what was "
    "said and not necessarily all of it."
)


@dataclass(frozen=True)
class SearchResult:
    """One retrieved passage, with what a citation needs and nothing a caller must interpret."""

    text: str
    source_name: str
    raw: Mapping[str, Any]


def map_results(results: Mapping[str, Any], *, say_when_approximate: bool = True) -> List[SearchResult]:
    """Turns what the deployment returned into results the framework can consume.

    Separated from the call so it can be held to its rules without a deployment: a mapping tested
    through an HTTP stub is a test of the stub.

    The source is named by what it is in this system — a turn and a position in it — rather than by
    a document title this deployment does not have. A caller that wants the surrounding words
    resolves the citation through the same contract.
    """
    passages: Sequence[Mapping[str, Any]] = results.get("passages") or []
    mapped = [
        SearchResult(
            text=p.get("preview", ""),
            source_name="turn {} message {}".format(p.get("source_id", ""), p.get("ordinal", 0)),
            raw=p,
        )
        for p in passages
    ]
    # A model handed an incomplete answer as though it were complete answers confidently from it.
    # A caveat with no results, on the other hand, is itself a result, so it is only ever appended
    # to something.
    if say_when_approximate and results.get("approximate") and mapped:
        mapped.append(SearchResult(text=APPROXIMATE_NOTICE, source_name="taisce", raw={}))
    return mapped


def search(client: Any, *, limit: Optional[int] = None, data_subject_id: Optional[str] = None,
           source_role: Optional[str] = None,
           say_when_approximate: bool = True) -> Callable[[str], Any]:
    """Builds the search function the framework calls, over one deployment.

    Returns an async callable taking the query and returning the mapped results, which is the shape
    every search seam in this ecosystem expects; a framework that wants its own type wraps these
    three fields without asking the deployment anything further.
    """

    async def run(query: str) -> List[SearchResult]:
        results = await client.search_passages(
            question=query, limit=limit, data_subject_id=data_subject_id, source_role=source_role
        )
        return map_results(results, say_when_approximate=say_when_approximate)

    return run
