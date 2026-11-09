# taisce

The client for a Taisce memory deployment over the v1 contract: `observe`, `freshness`, `recall`,
`context`, `search_passages`, the artifact calls, `resolve_citation`, and erasure and export;
asynchronous, with `httpx` as its only dependency. The agent framework adapters in this repository
are built on it.

## The address must be https

The client refuses an address that is not `https`, unless its host is loopback (`localhost`,
`127.0.0.0/8`, `::1`). The project credential travels with every request, and one mistyped scheme
would send it across the network in clear. A development deployment on the same machine is reached
over plain `http://localhost`.

## Erasure and export

```python
from taisce import Client, ErasureNotClean, Forbidden

async with Client("https://memory.example", erasure_token) as client:
    receipt = await client.erase_subject("alice", reason="right to erasure")
    print(receipt.deleted, receipt.residual)        # what went, and what still matches: nothing
    everything = await client.export_subject("bob")  # sections of rows, and a count per section
```

- `erase_subject` and `erase_sources` are separate because the deployment refuses a request naming
  both a person and sources. Each returns an `ErasureReceipt`.
- **A receipt that is not clean is raised, not returned**, as `ErasureNotClean` carrying the whole
  receipt on `.receipt`. An application that never reads the return value cannot tell a person their
  data is gone while rows still match.
- `Forbidden` is a credential whose access does not cover the call. A `read_only` key cannot erase,
  and neither can an `agent` key on a deployment that issues them; give these calls the `read_write`
  key your erasure path holds, never the one in the agent process. `ExportTooLarge` is a subject holding more rows than one export returns; the deployment
  refuses rather than returning part of it.
- No adapter offers erasure or export to a model. Whether a request to forget is genuine is the
  application's to decide, in code, before it calls.
