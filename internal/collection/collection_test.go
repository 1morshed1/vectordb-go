package collection

import (
	"math/rand"
	"testing"

	"github.com/1morshed1/vectordb-go/internal/distance"
)

func newTestCollection(t *testing.T, metric distance.Metric) *Collection {
	t.Helper()
	c, err := New(Config{Name: "test", Dim: 16, Metric: metric})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func randVec(r *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = float32(r.NormFloat64())
	}
	return v
}

func TestInsertSearchDelete(t *testing.T) {
	c := newTestCollection(t, distance.Cosine)
	r := rand.New(rand.NewSource(1))
	vecs := make([][]float32, 100)
	for i := range vecs {
		vecs[i] = randVec(r, 16)
		if err := c.Insert(uint64(i), vecs[i], Metadata{"n": float64(i)}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}
	if c.Len() != 100 {
		t.Fatalf("Len = %d, want 100", c.Len())
	}

	res, err := c.Search(vecs[7], 1, 50)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].ID != 7 {
		t.Fatalf("expected id 7, got %+v", res)
	}
	if res[0].Metadata["n"].(float64) != 7 {
		t.Errorf("metadata not attached: %+v", res[0].Metadata)
	}

	if err := c.Delete(7); err != nil {
		t.Fatal(err)
	}
	if c.Len() != 99 {
		t.Fatalf("Len after delete = %d, want 99", c.Len())
	}
	if _, err := c.GetMetadata(7); err != ErrNotFound {
		t.Errorf("expected ErrNotFound after delete, got %v", err)
	}
}

func TestCosineDoesNotMutateCaller(t *testing.T) {
	c := newTestCollection(t, distance.Cosine)
	v := []float32{3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	orig := append([]float32(nil), v...)
	_ = c.Insert(1, v, nil)
	for i := range v {
		if v[i] != orig[i] {
			t.Fatalf("caller vector mutated at %d: %v vs %v", i, v[i], orig[i])
		}
	}
}

func TestDuplicateAndDimErrors(t *testing.T) {
	c := newTestCollection(t, distance.Euclidean)
	if err := c.Insert(1, randVec(rand.New(rand.NewSource(1)), 16), nil); err != nil {
		t.Fatal(err)
	}
	if err := c.Insert(1, randVec(rand.New(rand.NewSource(2)), 16), nil); err != ErrDuplicateID {
		t.Errorf("want ErrDuplicateID, got %v", err)
	}
	if err := c.Insert(2, []float32{1, 2, 3}, nil); err != ErrDimMismatch {
		t.Errorf("want ErrDimMismatch, got %v", err)
	}
}

func TestManagerLifecycle(t *testing.T) {
	m := NewManager()
	if _, err := m.Create(Config{Name: "docs", Dim: 8, Metric: distance.Cosine}); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Create(Config{Name: "docs", Dim: 8, Metric: distance.Cosine}); err != ErrCollectionExists {
		// wrapped; compare via errors.Is-like check
		if err == nil {
			t.Fatal("expected duplicate create to fail")
		}
	}
	if _, err := m.Get("docs"); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Get("missing"); err == nil {
		t.Fatal("expected not found")
	}
	if got := m.List(); len(got) != 1 || got[0] != "docs" {
		t.Fatalf("List = %v", got)
	}
	if err := m.Drop("docs"); err != nil {
		t.Fatal(err)
	}
	if got := m.List(); len(got) != 0 {
		t.Fatalf("List after drop = %v", got)
	}
}
