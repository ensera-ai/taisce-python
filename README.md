<!-- Copyright 2026 The Taisce Authors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Taisce for Python

One repository, one distribution per package, released on PyPI's cadence:

| Directory | Package | What it is |
|---|---|---|
| `client/` | `taisce` | The v1 contract of a Taisce deployment, asynchronous, `httpx` as its only dependency |
| `agent-framework/` | `taisce-agent-framework` | The Microsoft Agent Framework adapter: a context provider |
| `langgraph/` | `taisce-langgraph` | The LangGraph adapter |

```python
from taisce import Client
from taisce_agent_framework import TaisceContextProvider
from agent_framework import Agent

client = Client("https://memory.example", token)
agent = Agent(chat_client, context_providers=[TaisceContextProvider(client, data_subject_id="alice")])
```

## What the provider does, and holds to

- **Memory arrives as exactly one `user` message, marked untrusted.** Never `system`, never
  `assistant`; `extend_messages`, never `extend_instructions`. The message is the line
  `taisce-memory/v1 untrusted` and a JSON document carrying the freshness watermark, what the recall
  did, and the recall's facts, reports and passages unchanged.
- **Recall failure is not fatal.** The agent runs without memory rather than not at all; `on_error`
  is how an operator still finds out.
- **Write failure is never silent.** A failed store raises.
- **A failed turn is not a memory.** The store runs only when the turn produced a response.
- **Only what people said becomes memory by default.** The person's messages and the assistant's
  final text; tool calls, tool results and the provider's own message are dropped.
- **A retried store is one observation, and a different turn is another.** The idempotency key is
  derived from the subject, the run and the turn, every field length-prefixed so that content cannot
  move a boundary between them: the conformance suite's derivation (`taisce-turn-key/2`), checked
  against its published vectors, and the same bytes as every other adapter's. A turn whose first
  attempt was stored by 0.1.x and whose retry is sent by 0.2.0 carries two keys and is stored twice;
  that is the one cost of the change, once per turn in flight across the upgrade.
- **Erasure and export are the application's, never the model's.** The client has typed
  `erase_subject`, `erase_sources`, `export_subject` and `export_sources`; no provider, tool or
  middleware offers them to a model.
- **Compaction is replacement, never a summary.** `TaisceCompaction` is a second context provider:
  when the history a history provider loaded holds more than `after_messages` non-system messages,
  it asks the deployment for the subject's context and replaces that history with one untrusted
  `user` message carrying the segments and the verbatim turns unchanged. A context that cannot be
  fetched leaves the history as it was. The LangGraph middleware does the same in `abefore_model`
  with `compact_after_messages`, rewriting the state the way the framework's own summarisation
  middleware does.

```python
history = InMemoryHistoryProvider()
agent = Agent(chat_client, context_providers=[
    history,
    TaisceCompaction(client, data_subject_id="alice", after_messages=40),
    TaisceContextProvider(client, data_subject_id="alice"),
])
```

Where the framework's Python surface lacks a seam the .NET one has, the feature is omitted here
and said so, never emulated: an adapter that decides things locally is how behaviour forks per
language.

## How it is proved

The adapter conformance suite from the service repository runs against a live deployment with
`python -m taisce_agent_framework.conformance` as the driver: the framework's own `Agent` with the
provider attached and a stubbed model.

```sh
TAISCE_TOKEN=… taisce conformance --api https://memory.example --cases ../taisce/conformance/cases.json \
  --driver "python -m taisce_agent_framework.conformance"
```

Every case passes for each adapter. `pytest client/tests agent-framework/tests langgraph/tests tests`
holds what needs no deployment; `client/tests_live` holds the client's erasure and export against a
deployment, with `TAISCE_API`, `TAISCE_TOKEN` (read_write) and `TAISCE_READER_TOKEN` (read_only)
set, and fails rather than skips without them. CI runs the deployment-free tests twice, at the newest
dependencies and at the declared floors, and audits both installs for published advisories.

## Versions

Tested on Python 3.12 at the newest dependencies and at each package's declared floors
(`agent-framework-core` 1.18.0; `langchain` 1.3.9, `langchain-core` 1.3.3 and `langgraph` 1.0.10, the
first lines clear of their published advisories). The packages declare Python 3.10 as their floor.
