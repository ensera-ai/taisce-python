# Copyright 2026 The Taisce Authors
# SPDX-License-Identifier: Apache-2.0
import uuid

import pytest

from taisce import Client, Forbidden, turn_key


async def _observe(client: Client, subject: str, words: str) -> str:
    messages = [{"role": "user", "content": words}, {"role": "assistant", "content": "Noted."}]
    receipt = await client.observe(idempotency_key=turn_key(subject, "live", messages), messages=messages,
                                   data_subject_id=subject)
    return receipt["id"]


@pytest.mark.asyncio
async def test_an_erasure_by_subject_returns_a_clean_receipt_and_leaves_nothing_to_export(api, token):
    subject = "live-" + uuid.uuid4().hex
    async with Client(api, token) as client:
        await _observe(client, subject, "My badge number is 4471.")
        receipt = await client.erase_subject(subject, reason="live test")
        assert receipt.clean and receipt.data_subject_id == subject
        assert sum(receipt.deleted.values()) > 0
        assert all(n == 0 for n in receipt.residual.values())
        left = await client.export_subject(subject)
        assert sum(left.rows.values()) == 0


@pytest.mark.asyncio
async def test_an_erasure_by_sources_erases_the_observation_it_names_and_not_its_neighbour(api, token):
    subject = "live-" + uuid.uuid4().hex
    async with Client(api, token) as client:
        doomed = await _observe(client, subject, "Delete this sentence.")
        await _observe(client, subject, "Keep this one.")
        receipt = await client.erase_sources([doomed], reason="live test")
        assert receipt.clean and receipt.source_observation_ids == [doomed]
        assert sum((await client.export_sources([doomed])).rows.values()) == 0
        assert sum((await client.export_subject(subject)).rows.values()) > 0
        await client.erase_subject(subject, reason="live test cleanup")


@pytest.mark.asyncio
async def test_a_read_only_key_is_refused_an_erasure_and_allowed_an_export(api, token, reader_token):
    subject = "live-" + uuid.uuid4().hex
    async with Client(api, token) as writer, Client(api, reader_token) as reader:
        await _observe(writer, subject, "Readers may look and not delete.")
        with pytest.raises(Forbidden):
            await reader.erase_subject(subject, reason="live test")
        exported = await reader.export_subject(subject)
        assert sum(exported.rows.values()) > 0
        await writer.erase_subject(subject, reason="live test cleanup")


@pytest.mark.asyncio
async def test_an_exports_row_counts_match_its_sections(api, token):
    subject = "live-" + uuid.uuid4().hex
    async with Client(api, token) as client:
        await _observe(client, subject, "Count what you hand back.")
        exported = await client.export_subject(subject)
        assert exported.data_subject_id == subject
        assert set(exported.rows) == set(exported.sections)
        for name, rows in exported.sections.items():
            assert exported.rows[name] == len(rows), name
        assert exported.rows.get("message", 0) >= 2
        await client.erase_subject(subject, reason="live test cleanup")
