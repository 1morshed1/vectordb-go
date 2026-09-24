# Learning Guide 01 — Distance Metrics

**Package:** `internal/distance`

This is the smallest package in the repo and the most important to get right,
because it runs in the innermost loop of every insert and every search. If it
is wrong, *nothing* above it can be correct; if it is slow, *everything* above
it is slow.

## 1. What "close" means

An embedding model maps a piece of text (or an image, or audio) to a point in a
high-dimensional space — typically 384, 768, or 1536 dimensions. The model is
trained so that **semantically similar inputs land near each other**. So
"nearest neighbor search" is literally "find the stored points closest to my
query point."

But "closest" needs a definition. We support three:

### Cosine — the angle between vectors

```
cosine_similarity(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

Cosine ignores magnitude and looks only at *direction*. This is the default for
text embeddings, and here is the intuition: a long document and a short
document about the same topic point in the same direction, even though the long
one may have a larger magnitude. We care that they are about the same thing,
not that one is longer.

- Range: `[-1, 1]`. `1` = identical direction, `0` = orthogonal (unrelated),
  `-1` = opposite.

### Dot product — the raw inner product

```
dot(a, b) = Σ aᵢ · bᵢ
```

No normalization. Grows without bound. Useful on its own for some models (e.g.
those trained with a dot-product objective), but its real role here is as the
**fast path for cosine** — see §3.

### Euclidean (L2) — straight-line distance

```
euclidean(a, b) = ‖a - b‖ = √(Σ (aᵢ - bᵢ)²)
```

The everyday notion of distance. Magnitude matters here. Good when the length
of a vector carries meaning, or for non-text data.

## 2. The one convention that keeps the rest of the codebase sane

The HNSW index (next guide) is written entirely in terms of **"smaller is
closer."** It never wants to think about whether a metric is a similarity
(bigger = better) or a distance (smaller = better).

So we normalize that here, once:

| Metric | Similarity range | We return (a *distance*) | Range |
|--------|------------------|--------------------------|-------|
| Cosine | sim ∈ [-1, 1] | `1 - sim` | `[0, 2]` |
| Dot | dot ∈ (-∞, ∞) | `-dot` | `(-∞, ∞)` |
| Euclidean | already a distance | unchanged | `[0, ∞)` |

Every function in `distance.go` returns a value where **smaller means more
similar.** This is the single most important line to remember when reading the
index code.

> **Decision:** Convert similarity → distance at the leaf, not at every call
> site. One place to reason about sign, and the entire rest of the codebase
> gets a uniform "min-heap of distances" model.

## 3. The normalize-then-dot optimization (your ML intuition paying off)

Look at the cosine formula again:

```
cosine_similarity(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

If we **normalize every vector to unit length** (`‖v‖ = 1`) *once, at insertion
time*, then the denominator becomes `1 · 1 = 1`, and:

```
cosine_similarity(a, b) = a · b       (when ‖a‖ = ‖b‖ = 1)
```

Cosine similarity collapses to a plain dot product. A dot product has **no
square roots** and touches each element once. Over the life of an index the
distance function runs *billions* of times, so removing two `sqrt` calls and a
division per comparison is a large, free win.

The test `TestNormalizeThenDotEqualsCosine` proves this identity holds. The
collection layer (guide 03) uses it: for a cosine collection, it normalizes
every incoming vector and then runs the index using the dot-product metric.

> **Decision:** Push normalization to insert time and switch cosine collections
> to the dot metric internally. Pay the normalization cost once per vector
> instead of once per comparison.

## 4. Why one pass over the data

`cosineDistance` computes `a·b`, `‖a‖²`, and `‖b‖²` in a *single* loop rather
than three. Each element of `a` and `b` is loaded from memory once and reused
for all three accumulators. On modern CPUs, for vectors that don't fit in L1
cache, memory bandwidth — not arithmetic — is the bottleneck, so a single pass
is meaningfully faster than three passes.

## 5. The bounds-check hint

Go inserts a bounds check on every slice index by default (memory safety). In a
loop over `a` that also reads `b[i]`, the compiler can prove `a[i]` is safe (we
iterate `range a`), but not `b[i]`. The line:

```go
_ = b[len(a)-1:]
```

is a single check up front that `b` is at least as long as `a`. After that, the
compiler can prove every `b[i]` inside the loop is in range and elides the
per-iteration check. This is a standard Go micro-optimization pattern for
paired-slice loops.

We also keep a `if len(b) < len(a) { a = a[:len(b)] }` guard: callers are
*supposed* to pass equal-length vectors (validated at the collection layer), but
a distance function should never read out of bounds even if someone upstream
has a bug.

## 6. Handling the zero vector

A zero vector has no direction, so cosine similarity is undefined (`0/0`). We
define its distance as the maximum (`2` for cosine) so it can never win a
search. Returning `1` (what `1 - 0` would give) would be a silent bug — it
would rank a zero vector as "medium similarity" to everything.

## 7. What we deliberately did *not* do (yet)

- **Hand-written SIMD / assembly.** Go's compiler auto-vectorizes these simple
  loops acceptably, and hand-rolled AVX would hurt readability. If profiling
  later shows the distance function dominates, that is the place to optimize —
  but "make it correct and clear first."
- **Squared-Euclidean-only ranking.** We pay the `sqrt` in Euclidean so the API
  returns a real distance. HNSW ranking would be correct without it (sqrt is
  monotonic), but the simplicity is worth more than the micro-optimization here.

## Takeaways

- Similarity vs distance is a sign convention; fix it once, at the leaf.
- Normalize at insert time so cosine becomes a dot product.
- The hot loop is memory-bound: one pass, reuse loaded values, hint away bounds
  checks.
- Correct-and-clear beats clever-and-fast until a profiler says otherwise.

Next: `02-hnsw-index.md`, the algorithm this whole project exists to teach.
