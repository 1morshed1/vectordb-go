package collection

import (
	"github.com/1morshed1/vectordb-go/internal/distance"
	"github.com/1morshed1/vectordb-go/internal/filter"
)

// FilterStrategy selects how a metadata filter is combined with the vector
// search. See docs/learning/04 for the full reasoning.
type FilterStrategy uint8

const (
	// StrategyAuto estimates the filter's selectivity and picks Pre or Post.
	StrategyAuto FilterStrategy = iota
	// StrategyPre applies the filter first, then does an EXACT scan over only
	// the matching vectors. Best when the filter is very selective.
	StrategyPre
	// StrategyPost runs the (approximate) HNSW graph search with the filter as an
	// admission predicate. Best when the filter matches a large fraction.
	StrategyPost
)

func (s FilterStrategy) String() string {
	switch s {
	case StrategyPre:
		return "pre"
	case StrategyPost:
		return "post"
	default:
		return "auto"
	}
}

// preFilterMaxSelectivity is the fraction of the collection at or below which
// StrategyAuto chooses the exact pre-filter. If a filter matches <= 10% of
// vectors, scanning just those exactly is both faster and more accurate than
// navigating the graph while rejecting the other 90%.
const preFilterMaxSelectivity = 0.10

// FilteredSearchResult is a SearchResult plus which strategy actually ran, so
// callers (and the benchmark) can see the auto decision.
type FilteredSearchResult struct {
	Results  []SearchResult
	Strategy FilterStrategy // the concrete strategy used (never Auto)
	// Matched is the number of vectors the filter matched, when known (pre-filter
	// and auto compute it; pure post leaves it -1).
	Matched int
}

// SearchFiltered runs a metadata-filtered nearest-neighbor search.
//
//	query    the query vector (normalized internally for cosine collections)
//	topK     how many results to return
//	efSearch beam width; <=0 uses the collection default
//	f        the metadata filter (use filter.MatchAll() for none)
//	strategy Pre, Post, or Auto
func (c *Collection) SearchFiltered(query []float32, topK, efSearch int, f filter.Filter, strategy FilterStrategy) (FilteredSearchResult, error) {
	if len(query) != c.cfg.Dim {
		return FilteredSearchResult{}, ErrDimMismatch
	}
	q := query
	if c.normalizeInput {
		q = make([]float32, len(query))
		copy(q, query)
		distance.Normalize(q)
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	total := len(c.meta)
	if total == 0 {
		return FilteredSearchResult{Strategy: strategy, Matched: 0}, nil
	}

	// Resolve Auto -> a concrete strategy by measuring selectivity. Because the
	// metadata lives in memory, we scan it once to both count matches and, if we
	// end up pre-filtering, reuse the collected ids. This scan is O(N) over
	// metadata (cheap map lookups), not over vectors.
	chosen := strategy
	var matchedIDs []uint64
	matched := -1
	if strategy == StrategyAuto || strategy == StrategyPre {
		matchedIDs = c.matchingIDs(f)
		matched = len(matchedIDs)
		if strategy == StrategyAuto {
			if float64(matched)/float64(total) <= preFilterMaxSelectivity || matched <= 2*topK {
				chosen = StrategyPre
			} else {
				chosen = StrategyPost
			}
		}
	}

	switch chosen {
	case StrategyPre:
		if matchedIDs == nil {
			matchedIDs = c.matchingIDs(f)
			matched = len(matchedIDs)
		}
		hits, err := c.index.BruteForceSubset(q, matchedIDs, topK)
		if err != nil {
			return FilteredSearchResult{}, err
		}
		return FilteredSearchResult{Results: c.attach(hits), Strategy: StrategyPre, Matched: matched}, nil

	default: // StrategyPost
		allow := func(id uint64) bool {
			md, ok := c.meta[id]
			return ok && f.Match(md)
		}
		hits, err := c.index.SearchFiltered(q, topK, efSearch, allow)
		if err != nil {
			return FilteredSearchResult{}, err
		}
		return FilteredSearchResult{Results: c.attach(hits), Strategy: StrategyPost, Matched: matched}, nil
	}
}

// matchingIDs returns the ids whose metadata satisfies the filter. Caller must
// hold at least the read lock.
func (c *Collection) matchingIDs(f filter.Filter) []uint64 {
	ids := make([]uint64, 0, len(c.meta)/4+1)
	for id, md := range c.meta {
		if f.Match(md) {
			ids = append(ids, id)
		}
	}
	return ids
}
