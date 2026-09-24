# Learning Guide 09 — Decisions Log & Scope

**No package** — this is the capstone. It collects, in one place, the major
decisions made across the project and maps the original `plan.md` spec to what
was actually built (and what was deliberately left out, and why). Read it after
the others as a review, or first if you want the executive summary.

## How to read the decision entries

Each entry is: **the decision**, then **why**, then the **alternative** we passed
on. The alternatives aren't "worse" — many are what production systems do — they
were just a bigger complexity step than the lesson at hand needed.

---

## Architecture-shaping decisions

### D1. Distance functions return a *distance*, never a similarity
Every metric returns "smaller = closer," converting cosine/dot similarities at
the leaf. **Why:** the index can then use one uniform min-heap model and never
reason about sign. **Alternative:** carry a "higher is better?" flag through the
index — more branching, more bugs. *(Guide 01)*

### D2. Cosine = normalize at insert + run a dot-product index
A cosine collection normalizes vectors on the way in and runs the index on the
dot metric. **Why:** `dot(unit, unit) == cosine similarity`, but with no
per-comparison square roots — and that loop runs billions of times.
**Alternative:** compute full cosine (two norms + a division) per comparison.
*(Guides 01, 03)*

### D3. HNSW uses the paper's *diversity heuristic* for neighbor selection
Neighbors are kept only if closer to the query than to already-selected
neighbors, not just "the M nearest." **Why:** raw-nearest clusters links in one
direction and leaves graph holes that trap search; diversity is the difference
between ~0.90 and ~0.999 recall. **Alternative:** simple top-M — simpler, worse.
*(Guide 02)*

### D4. Deletion is soft (tombstones)
Deleted nodes stay in the graph but are excluded from results. **Why:** physical
removal means expensive edge repair across every node that links to it.
**Alternative:** hard delete + edge repair, or periodic segment rebuild (what
production does as a background "optimizer"). *(Guide 02)*

### D5. Filtering separates *meaning* from *strategy*
`internal/filter` parses/evaluates predicates and knows nothing about HNSW; the
collection layer decides pre/post/auto. **Why:** two small testable concerns
instead of one tangled one. *(Guide 04)*

### D6. Filtered graph search *traverses* rejected nodes, only *excludes* them
The admission predicate controls the result set, not traversal. **Why:** matches
can sit on the far side of non-matching nodes; blocking traversal strands them.
**Alternative:** naive post-filter (search then drop) — can return zero results
when matches exist. *(Guide 04)*

### D7. Auto strategy measures selectivity, then picks
Scan metadata to count matches; ≤10% (or ≤2·topK) → exact pre-filter, else
approximate post-filter. **Why:** pre-filter is exact and cheap when selective;
post-filter wins when broad. **Alternative:** always one strategy — bad on the
other end of the selectivity range. *(Guide 04)*

### D8. Quantization is symmetric, per-vector, scalar int8
One scale per vector, codes in [-127,127], reconstruct as `code*scale`.
**Why:** matches embeddings' zero-centered distribution, gives 4× memory savings
at ~0.99 recall, and makes the dot product an integer multiply-add scaled once.
**Alternative:** product quantization (learned codebooks, 8–16×) — the natural
sequel, much more complex. *(Guide 05)*

### D9. Durability via write-ahead log + snapshots, with fsync before ack
Mutations hit the WAL (and fsync) before being applied or acknowledged.
**Why:** no acknowledged write can be lost to a crash. **Alternative:** periodic
flush only — a crash loses recent "successful" writes. *(Guide 06)*

### D10. Snapshot before truncate; replay is idempotent
Compaction writes the new snapshot durably, *then* truncates the WAL, and replay
ignores duplicate-insert / missing-delete. **Why:** a crash between the two only
replays the same mutations twice, which is safe; the other order loses data.
*(Guide 06)*

### D11. Recover by rebuilding the graph from persisted vectors
We persist vectors + metadata, not graph edges, and re-insert on load.
**Why:** keeps the guide about *durability*, not serialization format; the
recovered graph is guaranteed consistent. **Alternative (production):** persist
the adjacency lists in an mmap'd file so load is a mmap + demand paging, no
rebuild. This is the single biggest gap vs a real engine, and it's called out
here on purpose. *(Guide 06)*

### D12. The gRPC server is a thin adapter
It validates, converts wire↔domain types, and maps errors to status codes — no
search logic. **Why:** the hard logic stayed unit-testable without a network;
the transport is trivial to audit. *(Guide 07)*

### D13. Generated gRPC code is checked in
`api/gen/*.pb.go` is committed. **Why:** everyday `go build` needs no protoc;
the build is hermetic and the generated code is reviewable. `make proto`
regenerates it only when the `.proto` changes. *(Guide 07)*

### D14. Metrics via one interceptor + histograms
A single unary interceptor instruments every RPC; latency is a histogram.
**Why:** instrument-once, and histograms expose p99 tail latency that averages
hide. *(Guide 08)*

### D15. Recall is always measured against exact brute force
`hnsw.BruteForce` and the exact pre-filter stay in the codebase forever as
ground truth. **Why:** an approximate index is only trustworthy if you can
measure how approximate it is. *(Guides 02, 04, 08)*

---

## `plan.md` spec → what was built

| Spec item | Status | Notes |
|-----------|--------|-------|
| Store high-dim vectors | ✅ | `internal/hnsw` (flat `[][]float32`) |
| HNSW index | ✅ | full algorithm, recall-tested |
| Metadata filtering | ✅ | DSL + pre/post/auto |
| gRPC API (Insert/Search/Delete/CreateCollection) | ✅ | + Drop/List/Snapshot |
| Collection manager with config | ✅ | `internal/collection` |
| WAL for crash recovery | ✅ | fsync, CRC, torn-tail handling |
| Vectors as mmap'd flat file | ⛔️ scoped out | rebuild-on-load instead (D11) |
| HNSW graph as mmap'd adjacency | ⛔️ scoped out | rebuilt on load (D11) |
| Metadata in embedded KV (BoltDB/Badger) | ↔️ substituted | in-memory map persisted via WAL/snapshot |
| Prometheus metrics | ✅ | `internal/metrics` |
| int8 quantization | ✅ | `internal/quantize` (standalone module) |
| Benchmarks (recall/QPS/build/memory) | ✅ | `cmd/vectordb-bench` |
| Normalize-then-dot optimization | ✅ | D2 |
| Distributed sharding / replication | ⛔️ scoped out | a whole project on its own |

Legend: ✅ built · ↔️ built differently · ⛔️ deliberately out of scope.

## If you want to keep going

In rough order of value-for-effort:

1. **mmap the vectors and graph** (D11) — the biggest step toward "real,"
   removes rebuild-on-load, enables datasets larger than RAM.
2. **Integrate quantization into the index** — store `QuantizedVector`s, add
   exact re-ranking of the top candidates for full recall.
3. **Pooled `visited` bitset** in `searchLayer` — kills the per-query
   allocation that dominates build/search time (guide 02).
4. **Secondary metadata indexes** (roaring bitmaps) — sub-linear selectivity
   estimation and pre-filtering (guide 04).
5. **Sharding + replication** — partition collections across nodes, replicate
   for availability. The big one.

Each has a pointer into the learning docs where the current simplification and
its production alternative are already described. That's the whole point of this
repo: you now know not just how it works, but exactly where the edges are.
