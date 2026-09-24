package quantize

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

func TestQuantizeDequantizeRoundTrip(t *testing.T) {
	v := []float32{0.5, -0.25, 1.0, -1.0, 0.0, 0.75}
	q := Quantize(v)
	rec := q.Dequantize()
	// The largest-magnitude components (±1.0) should reconstruct almost exactly,
	// since they map to the extreme codes.
	if math.Abs(float64(rec[2]-1.0)) > 0.01 {
		t.Errorf("1.0 reconstructed as %v", rec[2])
	}
	if math.Abs(float64(rec[3]-(-1.0))) > 0.01 {
		t.Errorf("-1.0 reconstructed as %v", rec[3])
	}
}

func TestZeroVector(t *testing.T) {
	q := Quantize([]float32{0, 0, 0})
	if q.Scale != 0 {
		t.Errorf("zero vector scale = %v, want 0", q.Scale)
	}
	for _, c := range q.Dequantize() {
		if c != 0 {
			t.Error("zero vector did not round-trip to zeros")
		}
	}
}

func TestMemoryIsRoughly4xSmaller(t *testing.T) {
	dim := 1536
	v := make([]float32, dim)
	for i := range v {
		v[i] = rand.Float32()
	}
	q := Quantize(v)
	f32 := Float32Bytes(dim)
	ratio := float64(f32) / float64(q.MemoryBytes())
	t.Logf("float32 = %d bytes, int8 = %d bytes, ratio = %.2fx", f32, q.MemoryBytes(), ratio)
	if ratio < 3.9 {
		t.Errorf("memory ratio %.2f, expected ~4x", ratio)
	}
}

func TestQuantizedDotApproximatesFloatDot(t *testing.T) {
	// We normalize the error by ‖a‖·‖b‖ (the maximum possible |dot|), NOT by the
	// dot value itself. Two random Gaussian vectors are nearly orthogonal, so
	// their dot product is near zero — dividing by it would make even a tiny
	// absolute error look enormous. Error relative to the magnitude ceiling is
	// the meaningful measure of quantization quality.
	r := rand.New(rand.NewSource(1))
	dim := 256
	norm := func(v []float32) float64 {
		var s float64
		for _, x := range v {
			s += float64(x) * float64(x)
		}
		return math.Sqrt(s)
	}
	var worst float64
	for trial := 0; trial < 100; trial++ {
		a := make([]float32, dim)
		b := make([]float32, dim)
		for i := range a {
			a[i] = float32(r.NormFloat64())
			b[i] = float32(r.NormFloat64())
		}
		var exact float32
		for i := range a {
			exact += a[i] * b[i]
		}
		approx := Dot(Quantize(a), Quantize(b))
		normErr := math.Abs(float64(exact-approx)) / (norm(a) * norm(b))
		if normErr > worst {
			worst = normErr
		}
	}
	t.Logf("worst dot error over 100 pairs (normalized by |a||b|) = %.4f", worst)
	if worst > 0.02 {
		t.Errorf("quantized dot normalized error %.4f too high", worst)
	}
}

// TestQuantizationPreservesRanking is the key quality test: for a query, the
// nearest-neighbor ranking computed on quantized vectors should closely match
// the ranking on the original float32 vectors. This is what "minimal recall
// loss" means in practice.
func TestQuantizationPreservesRanking(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	dim, n, k := 128, 2000, 10
	vecs := make([][]float32, n)
	qvecs := make([]QuantizedVector, n)
	for i := range vecs {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(r.NormFloat64())
		}
		vecs[i] = v
		qvecs[i] = Quantize(v)
	}

	type scored struct {
		id  int
		val float32
	}
	topK := func(scores []scored) map[int]bool {
		sort.Slice(scores, func(i, j int) bool { return scores[i].val > scores[j].val })
		out := map[int]bool{}
		for i := 0; i < k; i++ {
			out[scores[i].id] = true
		}
		return out
	}

	var totalRecall float64
	trials := 50
	for q := 0; q < trials; q++ {
		query := vecs[q]
		qquery := Quantize(query)

		exact := make([]scored, n)
		approx := make([]scored, n)
		for i := range vecs {
			var d float32
			for j := range query {
				d += query[j] * vecs[i][j]
			}
			exact[i] = scored{i, d}
			approx[i] = scored{i, Dot(qquery, qvecs[i])}
		}
		et := topK(exact)
		at := topK(approx)
		hits := 0
		for id := range at {
			if et[id] {
				hits++
			}
		}
		totalRecall += float64(hits) / float64(k)
	}
	avg := totalRecall / float64(trials)
	t.Logf("quantized ranking recall@%d = %.4f", k, avg)
	if avg < 0.90 {
		t.Errorf("quantized recall %.4f too low (< 0.90)", avg)
	}
}
