package collection

import (
	"math/rand"
	"testing"

	"github.com/1morshed1/vectordb-go/internal/distance"
	"github.com/1morshed1/vectordb-go/internal/filter"
)

// buildFilterCollection inserts n vectors, tagging each with a "category" that
// is "rare" for every 20th id and "common" otherwise, plus a numeric "year".
func buildFilterCollection(t *testing.T, n int) (*Collection, [][]float32) {
	t.Helper()
	c, err := New(Config{Name: "f", Dim: 24, Metric: distance.Cosine, EfSearch: 200})
	if err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewSource(5))
	vecs := make([][]float32, n)
	for i := 0; i < n; i++ {
		v := randVec(r, 24)
		vecs[i] = v
		cat := "common"
		if i%20 == 0 {
			cat = "rare"
		}
		year := float64(2020 + i%5)
		if err := c.Insert(uint64(i), v, Metadata{"category": cat, "year": year}); err != nil {
			t.Fatal(err)
		}
	}
	return c, vecs
}

// exactFilteredTopK computes the ground-truth filtered nearest neighbors by
// brute force over the whole collection, applying the filter directly.
func exactFilteredTopK(c *Collection, q []float32, k int, f filter.Filter) []uint64 {
	res, _ := c.SearchFiltered(q, k, 0, f, StrategyPre) // Pre is exact
	ids := make([]uint64, len(res.Results))
	for i, r := range res.Results {
		ids[i] = r.ID
	}
	return ids
}

func TestFilteredResultsAllMatchFilter(t *testing.T) {
	c, vecs := buildFilterCollection(t, 1000)
	f, _ := filter.Parse(`category == "rare"`)
	for _, strat := range []FilterStrategy{StrategyPre, StrategyPost, StrategyAuto} {
		res, err := c.SearchFiltered(vecs[0], 10, 200, f, strat)
		if err != nil {
			t.Fatal(err)
		}
		if len(res.Results) == 0 {
			t.Fatalf("%s: no results", strat)
		}
		for _, r := range res.Results {
			if r.Metadata["category"].(string) != "rare" {
				t.Errorf("%s: result %d has category %v, want rare", strat, r.ID, r.Metadata["category"])
			}
		}
	}
}

func TestAutoChoosesPreForSelectiveFilter(t *testing.T) {
	c, vecs := buildFilterCollection(t, 1000)
	// "rare" matches 5% of vectors => below the pre-filter threshold.
	f, _ := filter.Parse(`category == "rare"`)
	res, err := c.SearchFiltered(vecs[0], 5, 200, f, StrategyAuto)
	if err != nil {
		t.Fatal(err)
	}
	if res.Strategy != StrategyPre {
		t.Errorf("auto chose %s for a 5%% filter, want pre", res.Strategy)
	}
}

func TestAutoChoosesPostForBroadFilter(t *testing.T) {
	c, vecs := buildFilterCollection(t, 1000)
	// "common" matches 95% => above the threshold.
	f, _ := filter.Parse(`category == "common"`)
	res, err := c.SearchFiltered(vecs[1], 5, 200, f, StrategyAuto)
	if err != nil {
		t.Fatal(err)
	}
	if res.Strategy != StrategyPost {
		t.Errorf("auto chose %s for a 95%% filter, want post", res.Strategy)
	}
}

// TestPostFilterMatchesPreFilterRecall checks the approximate post-filter finds
// (almost) the same top-k as the exact pre-filter for a broad filter.
func TestPostFilterMatchesPreFilterRecall(t *testing.T) {
	c, vecs := buildFilterCollection(t, 1000)
	f, _ := filter.Parse(`year >= 2021`) // matches 80%
	k := 10
	var totalRecall float64
	trials := 20
	for i := 0; i < trials; i++ {
		q := vecs[i*10]
		exact := exactFilteredTopK(c, q, k, f)
		post, _ := c.SearchFiltered(q, k, 200, f, StrategyPost)
		truth := map[uint64]bool{}
		for _, id := range exact {
			truth[id] = true
		}
		hits := 0
		for _, r := range post.Results {
			if truth[r.ID] {
				hits++
			}
		}
		totalRecall += float64(hits) / float64(len(exact))
	}
	avg := totalRecall / float64(trials)
	t.Logf("post-filter recall@%d vs exact = %.4f", k, avg)
	if avg < 0.85 {
		t.Errorf("post-filter recall %.4f too low", avg)
	}
}

func TestEmptyFilterEqualsUnfiltered(t *testing.T) {
	c, vecs := buildFilterCollection(t, 500)
	res, err := c.SearchFiltered(vecs[3], 1, 200, filter.MatchAll(), StrategyPost)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Results) == 0 || res.Results[0].ID != 3 {
		t.Errorf("expected id 3 nearest with match-all filter, got %+v", res.Results)
	}
}
