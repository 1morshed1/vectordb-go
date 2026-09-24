# Learning Guide 00 — Overview & How to Read This Repo

Welcome. This repository builds a **vector search database** in Go from
scratch — a simplified [Qdrant](https://qdrant.tech/) /
[Weaviate](https://weaviate.io/). If you have used pgvector, FAISS, or
ChromaDB in RAG projects but never looked *inside* one, this repo is the
inside.

The goal is not just "working code" — it is **understanding**. Every package
has a matching learning document in `docs/learning/` that explains *why* the
code looks the way it does, what the alternatives were, and which trade-off we
picked. Read the code and the doc side by side.

## The mental model

A vector database does exactly three things well:

1. **Store** high-dimensional vectors (embeddings) plus some metadata.
2. **Search** for the vectors closest to a query vector — fast, even with
   millions of vectors — using *approximate* nearest neighbor (ANN) search.
3. **Filter** results by metadata (e.g. `category == "finance" AND year >= 2024`).

Everything else (gRPC, persistence, quantization, metrics) is plumbing around
those three jobs.

## The reading order

Read the learning docs in numeric order. Each one maps to a Go package and a
git commit, so you can `git log` and follow the project as it grew.

| Doc | Package | What you learn |
|-----|---------|----------------|
| `00-overview.md` | — | This file: the map. |
| `01-distance-metrics.md` | `internal/distance` | Why cosine works, the normalize-then-dot trick, SIMD-friendly loops. |
| `02-hnsw-index.md` | `internal/hnsw` | The core ANN algorithm behind FAISS/Qdrant. The heart of the project. |
| `03-collections.md` | `internal/collection` | Tying vectors, index, and metadata into a named unit. |
| `04-metadata-filtering.md` | `internal/filter` | Pre-filter vs post-filter vs auto, and selectivity estimation. |
| `05-quantization.md` | `internal/quantize` | int8 quantization: 4× less memory using your ML intuition. |
| `06-persistence-and-wal.md` | `internal/storage` | Crash recovery, why a write-ahead log, snapshot format. |
| `07-grpc-api.md` | `api/`, `cmd/vectordb-server` | The wire protocol and server wiring. |
| `08-metrics-and-benchmarks.md` | `internal/metrics`, `cmd/vectordb-bench` | recall@k, QPS, and how we measure them. |

## The design principles we hold throughout

1. **Correctness before speed.** We start with a brute-force exact search we
   can trust, then prove HNSW matches it (that is what `recall@k` measures).
2. **Small, testable packages.** Each `internal/` package compiles and tests
   on its own. No package imports "up" toward the server.
3. **Comment the intent, not the syntax.** Comments say *why*, because the
   *what* is in the code.
4. **The standard library first.** We reach for third-party deps only when the
   standard library genuinely cannot do the job (e.g. gRPC).

## Prerequisites

- Go 1.24+
- (Later) `protoc` for regenerating gRPC stubs — but generated code is checked
  in, so you can build without it.

Let's begin. Open `01-distance-metrics.md`.
