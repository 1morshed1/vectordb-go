# Learning Guide 07 — The gRPC API

**Packages:** `api/proto`, `api/gen`, `internal/server`, `cmd/vectordb-server`

So far everything is a Go library. This guide puts it on the network so a RAG
application in any language can talk to it. We use **gRPC** — the same RPC
framework Qdrant, Milvus, and most infrastructure services expose.

## 1. Why gRPC (and not REST/JSON)

- **A schema, checked at compile time.** The `.proto` file is the single source
  of truth; client and server are generated from it, so a field rename is a
  compile error, not a 3am production surprise.
- **Efficient binary encoding.** Protocol Buffers are compact and fast to
  parse — it matters when you're shipping thousands of 1536-float vectors.
- **Streaming, deadlines, cancellation, and typed errors** come built in.

The cost is that you can't `curl` it as easily as JSON. For an
infrastructure service consumed by other services, that's the right trade.

## 2. The contract: `api/proto/vectordb.proto`

The `.proto` file defines the `VectorDB` service and every message. A few
choices worth calling out:

- **`MetadataValue` is a `oneof`** of string/number/bool. `oneof` is a tagged
  union: the type is explicit on the wire, mirroring our metadata value types
  exactly. No "is this JSON number an int or a float" ambiguity.
- **Enums for `DistanceMetric` and `FilterStrategy`** mirror the internal Go
  enums, so the wire API and the engine speak the same vocabulary.
- **`go_package` option** tells `protoc` where the generated Go lands
  (`api/gen`, package `vectordbpb`).
- **`SearchResponse` returns `strategy_used` and `matched`**, so a client that
  asked for `AUTO` can see which strategy actually ran — observability baked into
  the API.

## 3. Code generation (`make proto`)

`protoc` + two plugins turn the `.proto` into Go:

```
protoc --go_out=... --go-grpc_out=... api/proto/vectordb.proto
```

- `protoc-gen-go` → `vectordb.pb.go` (the message structs + marshaling).
- `protoc-gen-go-grpc` → `vectordb_grpc.pb.go` (the client + server interfaces).

**We check the generated code into `api/gen/`.** That means an everyday
`go build` needs *no* protoc — only someone editing the `.proto` runs
`make proto`. Checking in generated code is a deliberate, common choice for Go
projects: it keeps the build hermetic and reviewable. `make tools` installs the
plugins; the `Makefile` documents the whole flow.

## 4. The server is a thin adapter (`internal/server`)

`Service` implements the generated `VectorDBServer` interface, and it contains
**no search logic** — that all lives in the packages we already built and tested.
Its only jobs are:

1. **Validate** requests (non-empty vector, positive dim, …).
2. **Convert** between wire types and domain types (`metadataFromProto`,
   `metricFromProto`, `strategyFromProto`, and their inverses).
3. **Map errors to gRPC status codes** (`toStatus`).

Keeping the transport layer this thin means the hard logic stayed unit-testable
without a network, and the network layer is trivial to reason about.

### Embedding `UnimplementedVectorDBServer`

`Service` embeds `vectordbpb.UnimplementedVectorDBServer`. This is a
forward-compatibility trick: if we later add an RPC to the `.proto`, existing
servers still compile (they inherit a "not implemented" stub) instead of
breaking. gRPC-Go requires it.

### Errors as status codes (`toStatus`)

A good API returns *machine-readable* failures. `toStatus` maps our domain
errors to standard gRPC codes:

| Domain error | gRPC code |
|--------------|-----------|
| collection/id not found | `NotFound` |
| duplicate collection/id | `AlreadyExists` |
| dimension mismatch, bad filter | `InvalidArgument` |
| anything else | `Internal` |

`TestErrorCodes` asserts each of these, so clients can branch on `codes.NotFound`
rather than string-matching messages.

## 5. The server binary (`cmd/vectordb-server`)

`main.go` wires it together:

1. `storage.Open(--data)` — recover collections from disk.
2. Register the service with a `grpc.Server`, installing the metrics interceptor
   (guide 08).
3. Serve, and on `SIGINT`/`SIGTERM` do a **graceful shutdown**: stop accepting
   new RPCs, drain in-flight ones (with a 10s timeout so a stuck stream can't
   hang forever), **snapshot every collection**, then close the store. Snapshot-
   on-shutdown means a clean restart is fast (small WAL to replay).

## 6. Testing over a real gRPC connection (`bufconn`)

`server_test.go` doesn't logic-test the service directly — it stands up the
*actual* gRPC server and talks to it through a client, over an **in-memory
`bufconn`** listener (no TCP port, no flakiness). This exercises the real
serialization path: proto marshaling, the interface dispatch, error-code
propagation. `TestEndToEndFilteredSearch` even confirms that a 10%-selective
filter comes back with `strategy_used = PRE` across the wire.

This is the right level for transport tests: real enough to catch
serialization/wiring bugs, fast and hermetic enough to run on every commit.

## Takeaways

- The `.proto` is the contract; generate both sides from it and check the
  generated code in so builds don't need protoc.
- Keep the server a thin validate/convert/map-errors adapter over the
  already-tested engine.
- Map domain errors to gRPC status codes so clients get typed failures.
- Snapshot on graceful shutdown for fast restarts.
- Test the transport over an in-memory `bufconn` to exercise real serialization
  without a network.

Next: `08-metrics-and-benchmarks.md` — measuring recall, QPS, build time, and
memory.
