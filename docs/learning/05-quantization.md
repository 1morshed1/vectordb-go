# Learning Guide 05 — int8 Quantization

**Package:** `internal/quantize`

This is the module where knowing *what embeddings are* pays off as an
infrastructure win. The idea: store each vector's components as 1-byte integers
instead of 4-byte floats. That's a **4× memory reduction**, and because
embeddings are smooth and robust, it costs almost no search quality.

## 1. Why memory is the constraint

A `text-embedding-3-small` vector is 1536 dimensions of float32:

```
1536 × 4 bytes = 6,144 bytes per vector
```

At 10 million vectors that's ~61 GB *just for the raw vectors*, before the graph
edges. Memory — not CPU — is usually what decides how many vectors fit on a
machine. If we can store each component in 1 byte:

```
1536 × 1 byte + 4 bytes (scale) = 1,540 bytes per vector   →  ~4× smaller
```

`TestMemoryIsRoughly4xSmaller` confirms the 3.99× ratio. Four times as many
vectors per box is four times cheaper infrastructure for the same corpus.

## 2. Why it barely hurts quality (the ML intuition)

Embedding components are trained to be **smoothly and roughly symmetrically
distributed around zero**, and the *pattern* across dimensions carries the
meaning, not the exact 7th decimal of any one dimension. Rounding each component
to one of 256 levels perturbs every dimension by a tiny, roughly independent
amount. Those perturbations mostly cancel in the dot product (errors of opposite
sign), so the *ranking* of nearest neighbors is almost unchanged — and ranking
is all search cares about.

`TestQuantizationPreservesRanking` measures exactly this: the top-10 computed on
quantized vectors vs the top-10 on the originals. The run shows **recall ≈ 0.99**
— we keep essentially the same answers at a quarter of the memory.

## 3. The scheme: symmetric per-vector scalar quantization

For each vector independently:

1. `maxAbs = max_i |v_i|` — the biggest-magnitude component.
2. `Scale = maxAbs / 127`.
3. `code_i = clamp(round(v_i / Scale), -127, 127)` — an int8.

To reconstruct: `v_i ≈ code_i × Scale`.

Three words in that name, each a decision:

- **Symmetric** — centered on zero (range ±127), no separate offset. Embeddings
  are ~zero-centered, so a symmetric range wastes no codes, and it keeps the dot
  product clean (§4). We use ±127 rather than the full int8 [-128, 127] so the
  mapping is exactly symmetric; giving up one code is worth the simplicity.
- **Per-vector** — one `Scale` per vector, chosen from *that vector's* own
  maxAbs. A vector with small components gets a fine scale; a vector with large
  components gets a coarse one. This adapts to each vector instead of forcing a
  single global scale that would clip the big ones or waste precision on the
  small ones.
- **Scalar** — each dimension quantized independently. The more advanced
  alternative, **product quantization (PQ)**, splits the vector into sub-vectors
  and quantizes each against a learned codebook, reaching 8–16× compression. PQ
  is what FAISS's `IndexIVFPQ` uses; it's a natural next module but a big step up
  in complexity, so we stop at scalar here.

## 4. The payoff: integer dot products

Symmetric quantization makes the operation we care about most — the dot product
— factor cleanly:

```
dot(a, b) ≈ Σ (codeA_i · ScaleA)(codeB_i · ScaleB)
          = ScaleA · ScaleB · Σ (codeA_i · codeB_i)
```

The inner `Σ (codeA_i · codeB_i)` is **pure integer multiply-add** accumulated in
an `int32`, and we multiply by the two scales just *once* at the end. Integer
int8 dot products are extremely fast on modern CPUs (dedicated VNNI / dp4a
instructions); our `Dot` keeps a plain portable loop, but the structure is what
those instructions accelerate. This is `quantize.Dot`.

## 5. Measuring the error honestly

`ReconstructionError` reports the mean absolute difference between a vector and
its dequantized form — the raw precision you gave up.

A subtlety worth internalizing from `TestQuantizedDotApproximatesFloatDot`: we
normalize the dot-product error by `‖a‖·‖b‖` (the largest possible |dot|), **not**
by the dot value itself. Two random high-dimensional vectors are nearly
orthogonal, so their true dot product is near zero — dividing a small absolute
error by a near-zero value produces a meaningless "900% error." Relative-to-
ceiling is the fair measure, and by it the error is ~1%. *Choosing the right
denominator is itself a lesson: a metric that looks catastrophic can be an
artifact of normalizing by something near zero.*

## 6. How this would plug into the index

We built quantization as a standalone, well-tested module rather than threading
it through the HNSW storage, to keep each concept isolated. To use it in
production you would:

- Store `QuantizedVector`s instead of `[]float32` in the index's vector array.
- Use `quantize.Dot` as the distance function (negated, per guide 01's
  convention) inside `searchLayer`.
- Optionally keep full-precision vectors on disk and **re-rank** the top few
  hundred quantized candidates with exact distances — the standard
  "quantized search, exact re-rank" pattern that recovers the last bit of recall.

## Takeaways

- Memory is the real constraint; int8 gives 4× more vectors per box.
- Embeddings are robust, so rounding to 256 levels keeps recall ≈ 0.99.
- Symmetric per-vector scalar quantization makes the dot product an integer
  multiply-add scaled once at the end.
- Product quantization (learned codebooks) is the higher-compression sequel.
- Normalize error by the magnitude ceiling, not by a near-zero dot product.

Next: `06-persistence-and-wal.md` — surviving a crash and restarting with the
same data.
