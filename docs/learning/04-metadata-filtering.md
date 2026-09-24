# Learning Guide 04 — Metadata Filtering

**Packages:** `internal/filter`, plus `internal/collection/filtered_search.go`
and the `SearchFiltered`/`BruteForceSubset` methods in `internal/hnsw`.

Real queries are rarely "find vectors like this." They are "find vectors like
this **that are also** in the finance category, from 2024 or later, and public."
That extra clause is a **metadata filter**, and combining it with ANN search is
one of the genuinely hard parts of a vector database.

## 1. Two problems, kept separate

- **What does the filter mean?** — parse and evaluate a predicate over a
  metadata map. That's `internal/filter`.
- **When do we apply it relative to the vector search?** — the pre/post/auto
  strategy. That's the collection layer, because it owns *both* the index and
  the metadata.

Keeping these apart means the filter language knows nothing about HNSW, and the
strategy code knows nothing about parsing.

## 2. The filter language (`internal/filter`)

We support a small, readable DSL:

```
category == "finance" AND year >= 2024 AND NOT archived == true
region in ["us", "eu"] OR priority > 8
```

- **Operators:** `== != > >= < <=` and `in [ ... ]`.
- **Combinators:** `AND`, `OR`, `NOT`, parentheses.
- **Values:** strings (`'...'` or `"..."`), numbers, `true`/`false`.

### The AST (`filter.go`)

Everything is a `Filter` — an interface with `Match(md) bool`. The concrete
types are `Condition` (a leaf), `And`, `Or`, `Not`. Evaluation is a plain tree
walk. Two deliberate decisions:

- **A missing field never matches — even for `!=`.** `year >= 2024` on a
  document with no `year` is `false`, and so is `year != 2024`. Treating absence
  as "condition unsatisfiable" is far less surprising than silently passing.
- **Cross-type comparisons are `false`.** A string is never `<` a number.
  Numbers are all coerced to `float64` (via `toFloat`) so `int` metadata from Go
  callers and JSON's `float64` compare on one path.

### The parser (`parse.go`)

A hand-written **lexer + recursive-descent parser** — the same shape as a small
programming-language front end, small enough to read in one sitting:

```
expr    := orExpr
orExpr  := andExpr ("OR"  andExpr)*
andExpr := unary   ("AND" unary  )*
unary   := "NOT" unary | primary
primary := "(" expr ")" | condition
```

The grammar encodes **precedence**: `OR` is parsed *outermost* (binds loosest),
then `AND`, then `NOT`, then parentheses/conditions (bind tightest). So
`a OR b AND c` parses as `a OR (b AND c)` — the standard boolean precedence,
verified by `TestPrecedenceAndOverOr`. Writing precedence into the grammar's
shape (one function per level) is the classic trick; there is no precedence
table anywhere.

## 3. The strategy problem: pre vs post vs auto

Now the hard part. You have a filter that matches some subset, and you want the
top-k *nearest* vectors *within that subset*. Three ways:

### Pre-filter (exact) — `StrategyPre`

Apply the filter first to get the matching ids, then compute exact distances to
just those vectors and take the top-k (`hnsw.BruteForceSubset`).

- **Cost:** O(matched) distance computations.
- **Accuracy:** exact (recall 1.0).
- **Great when** the filter is *selective* (few matches). Scoring 50 vectors
  exactly beats navigating a million-node graph.
- **Terrible when** the filter matches most vectors — it degrades to full brute
  force.

### Post-filter (approximate) — `StrategyPost`

Run the HNSW graph search, but only let matching ids into the result set. The
key subtlety, implemented in `hnsw.searchLayer`'s `admit` predicate: **we still
*traverse* non-matching nodes** — they're bridges to matching ones — we just
don't *keep* them as results.

- **Cost:** roughly a normal graph search (grows as the filter gets more
  selective, because the result set fills slower and we explore more).
- **Accuracy:** approximate, but high for broad filters.
- **Great when** the filter matches a large fraction.
- **Weak when** the filter is very selective: the graph may have to wander far
  to find enough matches, and can miss some (lower recall). That's exactly the
  case pre-filter handles perfectly — hence auto.

> A naive "post-filter" — search top-k then drop non-matches — is *worse*: if
> none of the top-k match, you get zero results even though matches exist. Our
> predicate-integrated search avoids that by continuing until it has k matches
> or exhausts the reachable graph.

### Auto — `StrategyAuto`

Measure selectivity, then choose. Because metadata is in memory, we scan it once
to count matches (and reuse the collected ids if we pre-filter). The rule:

```go
if matched/total <= 0.10 || matched <= 2*topK {
    use Pre    // selective => exact subset scan wins
} else {
    use Post   // broad => graph search wins
}
```

The `matched <= 2*topK` guard handles tiny collections and very tight filters
where even 10% is only a handful of vectors. `TestAutoChoosesPreForSelective`
(5% filter → Pre) and `TestAutoChoosesPostForBroad` (95% filter → Post) pin this
behavior down.

## 4. How we know it's correct

- `TestFilteredResultsAllMatchFilter` — every returned result actually satisfies
  the filter, under all three strategies.
- `TestPostFilterMatchesPreFilterRecall` — the approximate post-filter's top-k
  matches the *exact* pre-filter's top-k (recall 1.0 in the run). Pre-filter is
  our ground truth here, exactly as brute force was for plain search.

## 5. What real systems add that we don't

- **Metadata indexes.** We scan the metadata map to evaluate filters. Qdrant and
  Weaviate build secondary indexes (inverted indexes, roaring bitmaps) so
  selectivity estimation and pre-filtering are sub-linear. Our O(N) scan is fine
  for teaching and for moderate N.
- **Cardinality statistics.** Production auto-strategy uses per-field histograms
  to estimate selectivity *without* scanning. We just scan — simpler and exact.

## Takeaways

- Separate "what the filter means" (parse/evaluate) from "when to apply it"
  (strategy).
- Precedence lives in the grammar's shape, one parse function per level.
- Pre-filter is exact and wins when selective; post-filter is approximate and
  wins when broad; auto measures and picks.
- Filtered graph search must *traverse* rejected nodes, only *exclude* them from
  results — otherwise you strand matches behind a wall of non-matches.

Next: `05-quantization.md` — spending your ML intuition to cut memory 4× with
int8 vectors.
