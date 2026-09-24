// Package distance implements the vector similarity/distance functions that
// power nearest-neighbor search.
//
// A vector database's whole job starts here: given two vectors, how "far
// apart" are they? Every index (HNSW included) is just a clever way to avoid
// computing this function against *every* stored vector. So the distance
// function must be (a) correct and (b) fast, because it runs in the innermost
// loop of every insert and every search.
//
// See docs/learning/01-distance-metrics.md for the theory.
package distance

import (
	"fmt"
	"math"
)

// Metric identifies a distance/similarity function. It is stored on a
// collection at creation time and never changes, because the HNSW graph is
// built assuming one specific notion of "close".
type Metric uint8

const (
	// Cosine measures the angle between two vectors, ignoring their
	// magnitude. This is the default for text embeddings: two documents about
	// the same topic point the same direction even if one is longer (larger
	// magnitude) than the other.
	Cosine Metric = iota

	// DotProduct is the raw inner product. For vectors that are already
	// L2-normalized, dot product == cosine similarity, but without the
	// per-call normalization cost. This is our fast path (see the package doc).
	DotProduct

	// Euclidean is straight-line (L2) distance. Useful for embeddings where
	// magnitude carries meaning, or for non-text data.
	Euclidean
)

func (m Metric) String() string {
	switch m {
	case Cosine:
		return "cosine"
	case DotProduct:
		return "dot"
	case Euclidean:
		return "euclidean"
	default:
		return fmt.Sprintf("metric(%d)", uint8(m))
	}
}

// ParseMetric converts a human string (as sent over the API) into a Metric.
func ParseMetric(s string) (Metric, error) {
	switch s {
	case "cosine", "":
		return Cosine, nil
	case "dot", "dotproduct", "dot_product", "ip", "inner_product":
		return DotProduct, nil
	case "euclidean", "l2":
		return Euclidean, nil
	default:
		return 0, fmt.Errorf("unknown distance metric %q", s)
	}
}

// -----------------------------------------------------------------------------
// The distance functions.
//
// IMPORTANT CONVENTION: every function in this file returns a *distance*, i.e.
// a value where SMALLER MEANS MORE SIMILAR. The HNSW index is written entirely
// in terms of "smaller is closer", so we convert similarities (where larger is
// better) into distances here, once, and never think about it again upstream.
//
//   - Cosine similarity is in [-1, 1] (1 == identical direction). We return
//     the cosine *distance* = 1 - similarity, which is in [0, 2].
//   - Dot product similarity grows without bound; we return -dot so that a
//     larger dot product becomes a smaller distance.
//   - Euclidean is already a distance.
// -----------------------------------------------------------------------------

// Distance returns the distance between a and b under the given metric.
// The two vectors must have the same length; callers are expected to have
// validated dimensions already (this is the hot path, so we skip the check).
func Distance(m Metric, a, b []float32) float32 {
	switch m {
	case Cosine:
		return cosineDistance(a, b)
	case DotProduct:
		return -dot(a, b)
	case Euclidean:
		return euclideanDistance(a, b)
	default:
		// Should be unreachable: metrics are validated at collection creation.
		return cosineDistance(a, b)
	}
}

// dot computes the inner product a·b.
//
// This loop is deliberately simple: a single accumulator, indexed access, no
// bounds-check elimination tricks. Go's compiler auto-vectorizes tight loops
// like this reasonably well, and keeping it readable matters more than a few
// percent. The `_ = b[len(a)-1]` hint tells the compiler both slices are at
// least len(a) long, which removes per-iteration bounds checks on b.
func dot(a, b []float32) float32 {
	if len(b) < len(a) {
		// Defensive: callers guarantee equal length, but never index OOB.
		a = a[:len(b)]
	}
	_ = b[len(a)-1:] // bounds-check hint
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// cosineDistance = 1 - cosine similarity.
//
// cosine similarity = (a·b) / (||a|| * ||b||).
//
// We compute all three sums (dot, ||a||², ||b||²) in one pass over the data,
// because a single pass is cache-friendly: each element of a and b is loaded
// once. If the vectors were pre-normalized at insert time, prefer the
// DotProduct metric instead — it skips the two norm computations entirely.
func cosineDistance(a, b []float32) float32 {
	if len(b) < len(a) {
		a = a[:len(b)]
	}
	_ = b[len(a)-1:]
	var dotp, na, nb float32
	for i := range a {
		dotp += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		// A zero vector has no direction; define it as maximally distant so it
		// never wins a search. (1 - 0 == 1 would be misleadingly "medium".)
		return 2
	}
	sim := dotp / float32(math.Sqrt(float64(na))*math.Sqrt(float64(nb)))
	return 1 - sim
}

// euclideanDistance returns the L2 distance ||a - b||.
//
// Note we return the true distance (with the sqrt). For pure ranking you could
// skip the sqrt (squared distance preserves order and is cheaper), but the
// value is exposed to API clients as a real distance, so we pay the sqrt once
// at the leaves. HNSW's internal comparisons still benefit because sqrt is
// monotonic — but we keep it simple and correct here.
func euclideanDistance(a, b []float32) float32 {
	if len(b) < len(a) {
		a = a[:len(b)]
	}
	_ = b[len(a)-1:]
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return float32(math.Sqrt(float64(sum)))
}

// Norm returns the L2 norm (Euclidean length) ||v|| of a vector.
func Norm(v []float32) float32 {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	return float32(math.Sqrt(float64(sum)))
}

// Normalize scales v in place to unit length (||v|| == 1) and returns it.
//
// This is the key optimization from the ML side: if we normalize every vector
// once at insertion time, then cosine similarity collapses to a plain dot
// product (because the denominator ||a||*||b|| becomes 1*1). Dot product is
// cheaper — no per-comparison square roots — and that inner loop runs billions
// of times over the life of an index.
//
// A zero vector is left unchanged (there is no meaningful unit direction).
func Normalize(v []float32) []float32 {
	n := Norm(v)
	if n == 0 {
		return v
	}
	inv := 1 / n
	for i := range v {
		v[i] *= inv
	}
	return v
}
