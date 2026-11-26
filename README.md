# vectordb-go

[![CI](https://github.com/1morshed1/vectordb-go/actions/workflows/ci.yml/badge.svg)](https://github.com/1morshed1/vectordb-go/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **vector search database** built from scratch in Go — a simplified Qdrant /
Weaviate. It stores high-dimensional embeddings, builds an **HNSW** index for
fast approximate nearest-neighbor search, supports **metadata filtering**, adds
**int8 quantization** and **crash-safe persistence**, and exposes everything over
a **gRPC** API with **Prometheus** metrics.

Every package has a companion design document in [`docs/learning/`](docs/learning/) that explains *why* the code is
the way it is and which trade-offs were chosen. If you've used pgvector, FAISS,
or ChromaDB but never looked inside one, start with
[`docs/learning/00-overview.md`](docs/learning/00-overview.md).

## What's inside

```
  RAG app ──gRPC──▶  VectorDB server
                     ├─ Collection manager   (named, schema'd vector sets)
                     ├─ HNSW index           (approximate NN search, O(log N))
                     ├─ Metadata filter      (pre / post / auto strategies)
                     ├─ int8 quantization    (4× less memory)
                     ├─ Storage: WAL + snapshots (crash recovery)
                     └─ Prometheus metrics
```

| Package | What it does | Learning doc |
|---------|--------------|--------------|
| `internal/distance` | cosine / dot / euclidean, normalize-then-dot | [01](docs/learning/01-distance-metrics.md) |
| `internal/hnsw` | the HNSW index (the core algorithm) | [02](docs/learning/02-hnsw-index.md) |
| `internal/collection` | collections + metadata + manager | [03](docs/learning/03-collections.md) |
| `internal/filter` | filter DSL, evaluator, strategies | [04](docs/learning/04-metadata-filtering.md) |
| `internal/quantize` | int8 scalar quantization | [05](docs/learning/05-quantization.md) |
| `internal/storage` | write-ahead log + snapshots | [06](docs/learning/06-persistence-and-wal.md) |
| `api/`, `internal/server`, `cmd/vectordb-server` | gRPC API + server | [07](docs/learning/07-grpc-api.md) |
| `internal/metrics`, `cmd/vectordb-bench` | Prometheus + benchmarks | [08](docs/learning/08-metrics-and-benchmarks.md) |

## Requirements

- **Go 1.24+** — to build and test everything.
- **protoc + plugins** — *only* if you edit `api/proto/vectordb.proto`. The
  generated code is checked into `api/gen/`, so ordinary builds don't need it.
  Install the plugins with `make tools`.

## Quick start

```bash
# Run all tests (unit + gRPC integration)
make test        # or: go test ./...

# Build the binaries into ./bin
make build

# Start the server (data persisted in ./data, metrics on :9090)
make run         # or: go run ./cmd/vectordb-server --addr :50051 --data ./data

# Run the benchmark (recall@k, QPS, build time, memory)
make bench       # or: go run ./cmd/vectordb-bench --n 20000 --dim 128
```

## Using the API

The service is defined in [`api/proto/vectordb.proto`](api/proto/vectordb.proto).
Generate a client in any language with `protoc`, or use the Go client. Core flow:

1. `CreateCollection{name, dim, metric}` — e.g. dim 1536, metric cosine.
2. `Insert{collection, id, vector, metadata}` — one vector at a time.
3. `Search{collection, vector, top_k, ef_search, filter, strategy}` — nearest
   neighbors, optionally filtered by an expression like:

   ```
   category == "finance" AND year >= 2024 AND region in ["us", "eu"]
   ```

4. `Snapshot{collection}` — compact the WAL into a fresh snapshot.

`Search` returns each hit's `id`, `distance`, and `metadata`, plus which filter
`strategy` actually ran and how many vectors matched the filter.

### Tuning knobs

- **`metric`** — `cosine` (default, for text embeddings), `dot`, or `euclidean`.
- **`M`** / **`ef_construction`** — index build quality vs build time/memory.
- **`ef_search`** — per-query recall vs latency. This is the dial you'll use
  most; see the benchmark output and [guide 08](docs/learning/08-metrics-and-benchmarks.md).
- **filter `strategy`** — `auto` (recommended), `pre` (exact, for selective
  filters), or `post` (approximate, for broad filters).

## Durability

Every insert/delete is written to a **write-ahead log** and `fsync`ed before it's
acknowledged, so an acknowledged write survives a crash. On restart the server
recovers each collection from its **snapshot + WAL replay**. A clean shutdown
snapshots every collection first for fast restarts. See
[guide 06](docs/learning/06-persistence-and-wal.md).

## Observability

The server exposes Prometheus metrics at `http://<metrics-addr>/metrics`:
request latency histograms (for p50/p95/p99), request/error counts, live vector
counts, and index build times. See [guide 08](docs/learning/08-metrics-and-benchmarks.md).

## What this is not

This is a learning-grade implementation. Compared to production engines it
deliberately omits: distributed sharding/replication, an mmap'd on-disk graph
(we rebuild the index on load), secondary metadata indexes, product
quantization, and fine-grained locking. Each omission is called out — with the
production alternative — in the relevant learning doc, so you know exactly what
the next step would be.

## Development

```bash
make fmt      # gofmt
make vet      # go vet
make test     # tests
make proto    # regenerate gRPC code (needs protoc + `make tools`)
```

## License

MIT — see [LICENSE](LICENSE).
