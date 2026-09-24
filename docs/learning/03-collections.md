# Learning Guide 03 — Collections & the Manager

**Package:** `internal/collection`

The index (guide 02) only knows about `uint64 id → vector`. A real vector
database works with **collections**: named, schema'd sets of vectors that also
carry *metadata* and enforce a *dimension* and *distance metric*. This package
is that layer, plus a `Manager` that owns many collections at once.

## 1. What a collection guarantees

A `Collection` keeps three things consistent under one lock:

1. **The index** — the HNSW graph over the vectors.
2. **The metadata** — `map[uint64]Metadata`, one payload per id, used for
   filtering (guide 04) and returned with results.
3. **The config** — dimension and metric, fixed at creation. You cannot insert
   a 512-d vector into a 1536-d collection, and you cannot change the metric
   after vectors exist (the graph was built assuming it).

Every mutation (`Insert`, `Delete`) updates the index *and* the metadata map
together, so they can never drift apart.

## 2. Where the cosine optimization actually happens

Guide 01 explained the theory: normalize vectors once, then cosine similarity
becomes a plain dot product. Guide 02's index just takes a metric. **This is the
package where the two meet.**

In `New`, if the requested metric is `Cosine`:

```go
indexMetric = distance.DotProduct   // run the index on dot product
normalize   = true                  // normalize every vector on the way in
```

So a "cosine collection" is really a **dot-product index fed normalized
vectors**. `prepareVector` normalizes on insert; `Search` normalizes the query
the same way. The index never knows cosine exists — it just does fast dot
products. This is exactly how production engines implement cosine.

> **Decision:** Keep the index metric-agnostic and put the normalize-then-dot
> policy in the collection layer. The index stays simple; the ML-specific choice
> lives in one obvious place.

## 3. We copy vectors, and we never mutate the caller's slice

`prepareVector` always `copy`s the input before normalizing. If we normalized
the caller's slice in place, we'd silently rescale a slice they still hold — a
nasty, hard-to-find bug. `TestCosineDoesNotMutateCaller` guards this. The cost
is one allocation per insert, which is negligible next to the graph work.

## 4. Metadata is deliberately a small type

```go
type Metadata map[string]any   // values: string | float64 | bool
```

We restrict values to three scalar types. Why not "anything"?

- The **filter engine** (guide 04) must define comparisons like `>=` and `==`.
  Those are only well-defined for a known, small set of types.
- Numbers are always `float64` (JSON's number type), so there's a single numeric
  path — no int-vs-float comparison surprises.

Nested objects and arrays are intentionally out of scope; they'd complicate the
filter grammar without teaching anything new.

## 5. The Manager: many collections, one server

`Manager` is the object the gRPC server holds. It maps `name → *Collection` and
offers `Create`, `Get`, `Drop`, `List`, and `Register`. Two design notes:

- **`List` sorts its output.** Deterministic ordering makes API responses and
  tests stable. Ranging over a Go map is deliberately randomized, so we sort.
- **`Register`** exists for the storage layer (guide 06): at startup we rebuild
  collections from disk and register the finished objects, rather than creating
  empty ones and replaying inserts through the public API.

## 6. Locking model

Two levels of `sync.RWMutex`:

- The **Manager** lock guards the name→collection map (create/drop/list).
- Each **Collection** has its own lock guarding its index + metadata.

They nest cleanly: the Manager hands out a `*Collection` under its read lock,
then callers operate on that collection under *its* lock. Reads (search) share;
writes (insert/delete) are exclusive. Coarse but correct — and correctness is
the rule until a profiler says otherwise.

## Takeaways

- A collection keeps index + metadata + config in lockstep under one lock.
- "Cosine collection" = dot-product index + normalized inputs; the policy lives
  here, not in the index.
- Always copy before normalizing so you never corrupt a caller's slice.
- Metadata is a small typed map precisely so filtering can define comparisons.

Next: `04-metadata-filtering.md` — deciding *which* vectors are even eligible,
and the pre-filter/post-filter/auto strategy choice.
