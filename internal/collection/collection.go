// Package collection groups a set of vectors, their HNSW index, and their
// metadata under a single named unit with a fixed schema (dimension + metric).
//
// A collection is the user-facing concept in every vector DB: "the documents
// collection", "the images collection". It owns three things that must stay in
// sync: the vectors (inside the index), the per-id metadata (for filtering),
// and the configuration. This package is the seam where the ML-facing decisions
// live — chiefly, normalizing vectors for cosine so the index can use the fast
// dot-product path (see docs/learning/01 and 03).
package collection

import (
	"errors"
	"fmt"
	"sync"

	"github.com/1morshed1/vectordb-go/internal/distance"
	"github.com/1morshed1/vectordb-go/internal/hnsw"
)

// Metadata is the per-vector payload used for filtering and returned with
// results. Values are restricted to the JSON-ish scalar types the filter engine
// understands: string, float64 (all numbers), and bool. Keeping the value set
// small keeps the filter engine simple and its comparisons well-defined.
type Metadata map[string]any

// Config describes a collection's fixed schema and index tuning.
type Config struct {
	Name           string
	Dim            int
	Metric         distance.Metric
	M              int
	EfConstruction int
	EfSearch       int
}

// SearchResult is one hit: id, distance, and the stored metadata.
type SearchResult struct {
	ID       uint64
	Distance float32
	Metadata Metadata
}

// Collection is a thread-safe named set of vectors + metadata + index.
type Collection struct {
	cfg Config

	// normalizeInput is true for cosine collections: we normalize every vector
	// on the way in and run the underlying index with the DotProduct metric,
	// because dot(normalized) == cosine similarity but is cheaper per compare.
	normalizeInput bool

	mu    sync.RWMutex
	index *hnsw.Index
	meta  map[uint64]Metadata
}

var (
	// ErrDimMismatch is returned when a vector length != the collection dim.
	ErrDimMismatch = errors.New("collection: vector dimension mismatch")
	// ErrDuplicateID is returned when inserting an id that already exists.
	ErrDuplicateID = errors.New("collection: duplicate id")
	// ErrNotFound is returned when an id is not present.
	ErrNotFound = errors.New("collection: id not found")
)

// New creates an empty collection from the given config.
func New(cfg Config) (*Collection, error) {
	if cfg.Dim <= 0 {
		return nil, fmt.Errorf("collection: dimension must be positive, got %d", cfg.Dim)
	}
	if cfg.Name == "" {
		return nil, errors.New("collection: name must not be empty")
	}

	// Cosine is implemented as normalize-then-dot. Everything else runs the
	// requested metric directly.
	indexMetric := cfg.Metric
	normalize := false
	if cfg.Metric == distance.Cosine {
		indexMetric = distance.DotProduct
		normalize = true
	}

	hc := hnsw.DefaultConfig(cfg.Dim, indexMetric)
	if cfg.M > 0 {
		hc.M = cfg.M
	}
	if cfg.EfConstruction > 0 {
		hc.EfConstruction = cfg.EfConstruction
	}
	if cfg.EfSearch > 0 {
		hc.EfSearch = cfg.EfSearch
	}

	return &Collection{
		cfg:            cfg,
		normalizeInput: normalize,
		index:          hnsw.New(hc),
		meta:           make(map[uint64]Metadata),
	}, nil
}

// Config returns a copy of the collection configuration.
func (c *Collection) Config() Config { return c.cfg }

// Len returns the number of live vectors.
func (c *Collection) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.meta)
}

// prepareVector validates the dimension and returns a copy that has been
// normalized if this is a cosine collection. We copy so the caller's slice is
// never mutated by our normalization.
func (c *Collection) prepareVector(vector []float32) ([]float32, error) {
	if len(vector) != c.cfg.Dim {
		return nil, ErrDimMismatch
	}
	v := make([]float32, len(vector))
	copy(v, vector)
	if c.normalizeInput {
		distance.Normalize(v)
	}
	return v, nil
}

// Insert adds a vector with optional metadata. Metadata may be nil.
func (c *Collection) Insert(id uint64, vector []float32, md Metadata) error {
	v, err := c.prepareVector(vector)
	if err != nil {
		return err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, exists := c.meta[id]; exists {
		return ErrDuplicateID
	}
	if err := c.index.Insert(id, v); err != nil {
		return err
	}
	if md == nil {
		md = Metadata{}
	}
	c.meta[id] = md
	return nil
}

// Delete removes a vector and its metadata. Returns ErrNotFound if absent.
func (c *Collection) Delete(id uint64) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.meta[id]; !ok {
		return ErrNotFound
	}
	c.index.Delete(id)
	delete(c.meta, id)
	return nil
}

// GetMetadata returns the metadata for an id (a copy is not made; treat as
// read-only). Returns ErrNotFound if absent.
func (c *Collection) GetMetadata(id uint64) (Metadata, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	md, ok := c.meta[id]
	if !ok {
		return nil, ErrNotFound
	}
	return md, nil
}

// Search returns the topK nearest neighbors, attaching metadata to each hit.
// efSearch<=0 uses the collection default. This is the unfiltered path; filtered
// search is added in the filter guide.
func (c *Collection) Search(query []float32, topK, efSearch int) ([]SearchResult, error) {
	if len(query) != c.cfg.Dim {
		return nil, ErrDimMismatch
	}
	q := query
	if c.normalizeInput {
		q = make([]float32, len(query))
		copy(q, query)
		distance.Normalize(q)
	}

	c.mu.RLock()
	defer c.mu.RUnlock()
	hits, err := c.index.Search(q, topK, efSearch)
	if err != nil {
		return nil, err
	}
	return c.attach(hits), nil
}

// Item is a full stored entry: id, its (possibly normalized) vector, and its
// metadata. Returned by Items for snapshotting.
type Item struct {
	ID       uint64
	Vector   []float32
	Metadata Metadata
}

// Items returns every live entry in the collection. The vectors are copies, so
// the caller may keep them. Used by the storage layer to snapshot state.
func (c *Collection) Items() []Item {
	c.mu.RLock()
	defer c.mu.RUnlock()
	items := make([]Item, 0, len(c.meta))
	c.index.ForEach(func(id uint64, vector []float32) {
		v := make([]float32, len(vector))
		copy(v, vector)
		items = append(items, Item{ID: id, Vector: v, Metadata: c.meta[id]})
	})
	return items
}

// attach converts index results into collection results by looking up metadata.
// Caller must hold at least the read lock.
func (c *Collection) attach(hits []hnsw.Result) []SearchResult {
	out := make([]SearchResult, 0, len(hits))
	for _, h := range hits {
		out = append(out, SearchResult{
			ID:       h.ID,
			Distance: h.Distance,
			Metadata: c.meta[h.ID],
		})
	}
	return out
}
