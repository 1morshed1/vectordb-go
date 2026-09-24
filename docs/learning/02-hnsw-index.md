# Learning Guide 02 — The HNSW Index

**Package:** `internal/hnsw`

This is the algorithm the whole project exists to teach. HNSW (Hierarchical
Navigable Small World) is what powers FAISS's `IndexHNSWFlat`, Qdrant, Weaviate,
and pgvector's `hnsw` index. If you understand this file, you understand the
core of every modern vector database.

## 1. The problem: exact search is too slow

To find the nearest neighbor exactly, you compare the query to *every* stored
vector — O(N) distance computations per query. That is `BruteForce` in the code,
and at 10 million vectors it is hopeless for real-time search.

We accept a deal: give up *exactness* for *speed*. An **approximate** nearest
neighbor (ANN) index returns the true top-k *almost* always. "Almost" is
measured by **recall@k** = (fraction of the true top-k that we actually
returned). Our test shows recall@10 = 0.999 — we miss the true answer roughly
1 in 1000 times, and in exchange we touch a tiny fraction of the vectors.

> This is the single most important idea in vector search: **it is
> probabilistic.** That is why `BruteForce` stays in the codebase forever — it
> is the ground truth that lets us *measure* how approximate we are.

## 2. The intuition: a skip-list for geometry

Start with a plain graph where each vector is a node linked to its nearest
neighbors ("navigable small world"). To search, walk greedily: from your current
node, jump to whichever neighbor is closest to the query, repeat until no
neighbor is closer. This works but can get stuck taking many small steps across
the whole dataset.

HNSW's insight: **add express lanes.** Build multiple layers:

```
Layer 2:  ●─────────────────●          few nodes, long jumps
Layer 1:  ●──────●──────●────●          more nodes, medium jumps
Layer 0:  ●●●●●●●●●●●●●●●●●●●●●          every node, short jumps
```

- **Layer 0** holds *every* vector with short-range links — for fine-grained
  local search.
- Each **higher layer** holds an exponentially smaller random sample with
  longer-range links — an express highway.

Search enters at the top, takes big greedy jumps to get into the right region
fast, then drops down a layer and refines, until layer 0 where it does a careful
beam search. This is a **skip list generalized to metric space**, and it gives
O(log N) search.

## 3. The level distribution (`randomLevel`)

When we insert a node, we roll its top layer from a geometric distribution:

```go
level = floor(-ln(U) * ml)   where U ~ Uniform(0,1], ml = 1/ln(M)
```

- P(node at layer 0) ≈ 1 − 1/M
- P(node at layer 1) ≈ 1/M
- P(node at layer 2) ≈ 1/M²  … and so on.

With M=16, ~94% of nodes live only at layer 0, ~6% reach layer 1, ~0.4% reach
layer 2. That exponential thinning is exactly what makes the upper layers a
sparse express network. `ml = 1/ln(M)` is the normalization from the paper that
makes the expected number of layers ≈ log_M(N).

## 4. The two priority queues (`heap.go`)

`searchLayer` — the heart of both insert and search — juggles two heaps:

| Queue | Type | Why |
|-------|------|-----|
| **candidates** (frontier "C") | min-heap | always expand the *closest* unexplored node — most promising direction |
| **results** ("W") | max-heap | keep only the `ef` best; when full, evict the *farthest*, which must be at the top |

The **early-termination** check is what makes it fast:

```go
if results.Len() >= ef && nearest.dist > results.peek().dist {
    break
}
```

If the closest thing left to explore is already farther than the *worst* result
we're keeping, nothing remaining can improve the answer — stop. Without this,
beam search would wander the whole graph.

`ef` (the beam width) is the master speed/accuracy knob:
- `efConstruction` during insert → graph *quality*.
- `efSearch` during query → query *recall*. Our test shows ef=10 → 0.76 recall,
  ef=200 → 1.00. Higher ef = more nodes visited = higher recall, slower query.

## 5. Insertion, step by step (`Insert`)

1. Roll `newLevel` for the node.
2. **Phase 1 — descend (greedy, beam width 1):** from the global entry point,
   walk down layers `maxLevel … newLevel+1`, at each layer greedily hopping to
   the closest neighbor. We do *not* add links here; we're just finding a good
   entry point for the layers where we *will* link. (`greedyDescend`.)
3. **Phase 2 — connect:** from `min(newLevel, maxLevel)` down to 0:
   - `searchLayer` with width `efConstruction` to gather candidates.
   - `selectNeighborsHeuristic` picks the best M.
   - Wire **bidirectional** edges (new↔selected).
   - If a neighbor now exceeds its degree budget (`M`, or `2M` at layer 0),
     `pruneNeighbors` re-selects its best links.
4. If `newLevel > maxLevel`, the new node becomes the global entry point.

Two degree budgets matter: `M` on upper layers, `mMax0 = 2M` at layer 0. Layer 0
has all the nodes and needs denser connectivity, so the paper gives it double.

## 6. Neighbor selection: diversity beats "just the closest"

The naive choice is "keep the M nearest candidates." It performs surprisingly
*badly*: the M nearest are often clustered together in one direction, so the
node has no links toward other regions, and the graph develops holes that trap
greedy search.

`selectNeighborsHeuristic` (paper Algorithm 4) instead keeps a candidate only
if it is closer to the query than to any already-selected neighbor:

```
keep c  ⟺  for all selected s:  dist(c, query) < dist(c, s)
```

This spreads links out in *different directions*, keeping the graph navigable.
It is the difference between ~0.90 and ~0.999 recall at low M, and it's why the
prune step reuses the same heuristic instead of trimming the farthest edges.

## 7. Deletion is "soft" (`Delete`)

Physically removing a node means finding every node that links to it and
repairing those edges — expensive and lock-heavy. So we **tombstone**: mark the
node `deleted`, exclude it from result sets, and unmap its id. The node still
participates in graph *navigation* (its edges still route traffic) but never
appears in results. This is exactly what Qdrant and Weaviate do; a background
optimizer later rebuilds segments to reclaim the space (guide 06).

## 8. Thread safety

A single `sync.RWMutex` guards the whole index. `Search` and `BruteForce` take
the read lock (many concurrent readers); `Insert` and `Delete` take the write
lock. This is coarse but correct. Finer-grained locking (per-node, or lock-free
reads with a copy-on-write graph) is a real optimization vector databases use,
but it is a large complexity jump — correctness first.

## 9. What we deliberately left for later / did simply

- **`visited` as a map.** Each `searchLayer` allocates a map. A production index
  keeps a pooled visited-bitset (a `[]uint32` of "visit generation" stamps) to
  avoid per-query allocation. Clear, then fast.
- **Vectors stored in a `[][]float32`.** Simple and cache-hostile (pointer
  chasing). Guide 06's storage layer moves toward a flat mmap'd buffer.
- **No incremental entry-point repair on delete.** Fine at our scale.

## 10. How to convince yourself it works

Run `go test ./internal/hnsw/ -v`. The tests that matter:

- `TestRecallAgainstBruteForce` — HNSW's top-10 vs the *exact* top-10 over 100
  queries. This is the proof the approximation is trustworthy.
- `TestRecallImprovesWithEfSearch` — turns the efSearch knob and shows recall
  climb, demonstrating the speed/accuracy trade-off you now control.

## Takeaways

- ANN trades exactness for speed; always keep a brute-force ground truth to
  measure recall.
- Layers = express lanes → O(log N). The level distribution is exponential.
- Beam search with early termination + two heaps is the core loop.
- Neighbor *diversity* (the heuristic), not raw closeness, makes the graph work.
- `efConstruction` tunes build quality; `efSearch` tunes query recall.

Next: `03-collections.md` — wrapping the index, vectors, and metadata into a
named, dimension-checked unit.
