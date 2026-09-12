# taisce-agent-framework

The Microsoft Agent Framework has a context-provider seam, and `TaisceContextProvider` fills it: before
each model call it fetches the deployment's context for the subject and injects it as one `user`
message marked untrusted, and after the turn it records the person's message and the assistant's final
reply. Tool calls, tool results and the injected message itself are never stored.

Memory arrives as a user message rather than a system message on purpose. What the deployment returns
was written by somebody else, and a model that reads it as instruction can be steered by anything
anyone ever said to it.

The trigger is a count of loaded non-system messages, because that is what this framework's Python SDK
counts by. The stored history is the application's and is not rewritten; once over the trigger, every
run fetches the context once.

Recall failure is not fatal — the agent runs with more to read, not less. A failed store raises. Held
to the conformance suite through `python -m taisce_agent_framework.conformance`.

Built on [`taisce`](https://pypi.org/project/taisce/), the client for the v1 contract.
