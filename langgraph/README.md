# taisce-langgraph

LangGraph has no context-provider interface; its seam is agent middleware. `TaisceMemory` is an
`AgentMiddleware` for `create_agent`: before each model call it injects memory as one human message
marked untrusted, and after the agent run it records the person's message and the assistant's final
reply. Tool calls, tool results and the injected message are never stored. Recall failure is not
fatal; a failed store raises. Held to the conformance suite through
`python -m taisce_langgraph.conformance`.
