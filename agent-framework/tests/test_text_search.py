# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
#
# The on-ramp: a search over what was said, mapped onto the framework's seam, adding nothing. Each
# result keeps enough identity to cite, and an answer from an index still being built says so — a
# model handed an incomplete answer as a complete one answers confidently from it.
import pytest

from taisce_agent_framework import APPROXIMATE_NOTICE, map_results, search

ANSWER = {
    "generation_id": "g1",
    "through_offset": 9,
    "covered_through_offset": 4,
    "build_state": "building",
    "approximate": True,
    "passages": [
        {"chunk_id": "c1", "source_id": "o1", "ordinal": 2, "role": "user",
         "preview": "I work at Ensera.", "similarity": 0.81,
         "occurred_at": "2026-01-01T00:00:00Z", "preview_complete": True},
    ],
}


def test_every_passage_becomes_one_citable_result_and_an_incomplete_index_says_so():
    mapped = map_results(ANSWER)
    assert len(mapped) == 2
    assert mapped[0].text == "I work at Ensera."
    assert mapped[0].source_name == "turn o1 message 2"
    assert mapped[0].raw is ANSWER["passages"][0]
    assert mapped[1].text == APPROXIMATE_NOTICE

    assert len(map_results(ANSWER, say_when_approximate=False)) == 1
    assert len(map_results({**ANSWER, "approximate": False})) == 1
    # A caveat with no results is itself a result.
    assert map_results({**ANSWER, "passages": []}) == []


@pytest.mark.asyncio
async def test_the_search_function_asks_the_deployment_what_it_was_configured_to_ask():
    asked = {}

    class Deployment:
        async def search_passages(self, **kwargs):
            asked.update(kwargs)
            return ANSWER

    run = search(Deployment(), limit=3, data_subject_id="marta", source_role="user")
    mapped = await run("where does marta work")
    assert asked == {"question": "where does marta work", "limit": 3,
                     "data_subject_id": "marta", "source_role": "user"}
    assert mapped[0].source_name == "turn o1 message 2"
