package hnsw

import (
	"math/rand"
	"testing"

	"github.com/1morshed1/vectordb-go/internal/distance"
)

// randomVectors generates n deterministic random vectors of the given dim.
func randomVectors(n, dim int, seed int64) [][]float32 {
	r := rand.New(rand.NewSource(seed))
	out := make([][]float32, n)
	for i := range out {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(r.NormFloat64())
		}
		out[i] = v
	}
	return out
}

func TestInsertAndLen(t *testing.T) {
	idx := New(DefaultConfig(8, distance.Euclidean))
	vecs := randomVectors(50, 8, 1)
	for i, v := range vecs {
		if err := idx.Insert(uint64(i), v); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}
	if idx.Len() != 50 {
		t.Fatalf("Len = %d, want 50", idx.Len())
	}
}

func TestDimMismatch(t *testing.T) {
	idx := New(DefaultConfig(4, distance.Euclidean))
	if err := idx.Insert(1, []float32{1, 2, 3}); err != ErrDimMismatch {
		t.Fatalf("want ErrDimMismatch, got %v", err)
	}
}

func TestDuplicateID(t *testing.T) {
	idx := New(DefaultConfig(4, distance.Euclidean))
	_ = idx.Insert(1, []float32{1, 2, 3, 4})
	if err := idx.Insert(1, []float32{5, 6, 7, 8}); err != ErrDuplicateID {
		t.Fatalf("want ErrDuplicateID, got %v", err)
	}
}

func TestSearchFindsExactMatch(t *testing.T) {
	idx := New(DefaultConfig(8, distance.Euclidean))
	vecs := randomVectors(200, 8, 2)
	for i, v := range vecs {
		_ = idx.Insert(uint64(i), v)
	}
	// Querying with a stored vector should return that vector first, distance ~0.
	res, err := idx.Search(vecs[42], 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 {
		t.Fatalf("want 1 result, got %d", len(res))
	}
	if res[0].ID != 42 {
		t.Errorf("nearest to stored vector 42 was %d (dist %v)", res[0].ID, res[0].Distance)
	}
}

func TestSearchEmptyIndex(t *testing.T) {
	idx := New(DefaultConfig(4, distance.Euclidean))
	res, err := idx.Search([]float32{1, 2, 3, 4}, 5, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 0 {
		t.Fatalf("empty index should return no results, got %d", len(res))
	}
}

func TestDeleteExcludesFromResults(t *testing.T) {
	idx := New(DefaultConfig(8, distance.Euclidean))
	vecs := randomVectors(100, 8, 3)
	for i, v := range vecs {
		_ = idx.Insert(uint64(i), v)
	}
	// Delete id 10, then query with its vector: it must not come back.
	if !idx.Delete(10) {
		t.Fatal("delete returned false")
	}
	if idx.Len() != 99 {
		t.Fatalf("Len after delete = %d, want 99", idx.Len())
	}
	res, _ := idx.Search(vecs[10], 5, 50)
	for _, r := range res {
		if r.ID == 10 {
			t.Error("deleted id 10 came back in results")
		}
	}
}

// recallAtK returns the fraction of the exact top-k that HNSW also found.
func recallAtK(exact, approx []Result, k int) float64 {
	truth := make(map[uint64]bool)
	for i := 0; i < k && i < len(exact); i++ {
		truth[exact[i].ID] = true
	}
	hits := 0
	for i := 0; i < k && i < len(approx); i++ {
		if truth[approx[i].ID] {
			hits++
		}
	}
	return float64(hits) / float64(k)
}

// TestRecallAgainstBruteForce is the most important test in the repo: it proves
// the approximate index actually finds the right answers. We build an index,
// then for many queries compare HNSW's top-10 to the exact top-10 from brute
// force. With these parameters we expect very high recall.
func TestRecallAgainstBruteForce(t *testing.T) {
	const (
		n   = 2000
		dim = 32
		k   = 10
		nQ  = 100
	)
	cfg := DefaultConfig(dim, distance.Euclidean)
	cfg.EfConstruction = 200
	cfg.EfSearch = 100
	idx := New(cfg)

	vecs := randomVectors(n, dim, 7)
	for i, v := range vecs {
		if err := idx.Insert(uint64(i), v); err != nil {
			t.Fatal(err)
		}
	}

	queries := randomVectors(nQ, dim, 99)
	var total float64
	for _, q := range queries {
		exact, _ := idx.BruteForce(q, k)
		approx, _ := idx.Search(q, k, cfg.EfSearch)
		total += recallAtK(exact, approx, k)
	}
	avg := total / float64(nQ)
	t.Logf("average recall@%d over %d queries = %.4f", k, nQ, avg)
	if avg < 0.90 {
		t.Errorf("recall@%d = %.4f, want >= 0.90", k, avg)
	}
}

// TestRecallImprovesWithEfSearch verifies the fundamental speed/accuracy knob:
// a larger efSearch should never reduce recall and generally increases it.
func TestRecallImprovesWithEfSearch(t *testing.T) {
	const (
		n   = 2000
		dim = 32
		k   = 10
		nQ  = 50
	)
	idx := New(DefaultConfig(dim, distance.Euclidean))
	vecs := randomVectors(n, dim, 11)
	for i, v := range vecs {
		_ = idx.Insert(uint64(i), v)
	}
	queries := randomVectors(nQ, dim, 13)

	measure := func(ef int) float64 {
		var total float64
		for _, q := range queries {
			exact, _ := idx.BruteForce(q, k)
			approx, _ := idx.Search(q, k, ef)
			total += recallAtK(exact, approx, k)
		}
		return total / float64(nQ)
	}
	low := measure(10)
	high := measure(200)
	t.Logf("recall@10: ef=10 -> %.4f, ef=200 -> %.4f", low, high)
	if high < low {
		t.Errorf("higher efSearch reduced recall: ef10=%.4f ef200=%.4f", low, high)
	}
}

func TestCosineViaNormalizedDot(t *testing.T) {
	// Mimic what the collection layer does: normalize vectors and use DotProduct.
	dim := 16
	idx := New(DefaultConfig(dim, distance.DotProduct))
	vecs := randomVectors(300, dim, 21)
	for i, v := range vecs {
		distance.Normalize(v)
		_ = idx.Insert(uint64(i), v)
	}
	q := make([]float32, dim)
	copy(q, vecs[5])
	res, _ := idx.Search(q, 1, 50)
	if len(res) == 0 || res[0].ID != 5 {
		t.Errorf("expected id 5 as nearest, got %+v", res)
	}
}
