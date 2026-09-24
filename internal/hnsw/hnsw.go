// Package hnsw implements a Hierarchical Navigable Small World (HNSW) index for
// approximate nearest-neighbor (ANN) search.
//
// HNSW is the algorithm behind FAISS's IndexHNSWFlat, Qdrant, Weaviate, and
// pgvector's hnsw index. It is the single most important idea in this whole
// project, so the code is written to be *read*, not just run. Pair it with
// docs/learning/02-hnsw-index.md.
//
// The one-paragraph summary: build a multi-layer graph. The bottom layer (0)
// contains every vector and has short-range links. Each higher layer contains
// an exponentially smaller random sample of nodes with longer-range links.
// To search, enter at the top and greedily walk toward the query, dropping down
// a layer whenever you can't get closer, until you reach layer 0 where you do a
// wider beam search. The upper layers act like an express highway that gets you
// into the right neighborhood in O(log N) hops; layer 0 does the fine-grained
// search. Insertion is a search followed by wiring up bidirectional edges.
//
// Reference: Malkov & Yashunin, "Efficient and robust approximate nearest
// neighbor search using Hierarchical Navigable Small World graphs" (2016).
package hnsw

import (
	"container/heap"
	"errors"
	"math"
	"math/rand"
	"sort"
	"sync"

	"github.com/1morshed1/vectordb-go/internal/distance"
)

// Config holds the tunable knobs of an HNSW index. The defaults (see
// DefaultConfig) match what Qdrant and pgvector ship with.
type Config struct {
	// Dim is the dimensionality of every vector. Fixed at creation.
	Dim int

	// Metric is the distance metric. For cosine collections the layer above
	// normalizes vectors and passes DotProduct here (see guide 01).
	Metric distance.Metric

	// M is the number of bidirectional links created for each node on insertion
	// (per layer, above layer 0). Higher M => better recall, more memory, slower
	// build. Typical range 8–48; 16 is a good default.
	M int

	// EfConstruction is the size of the dynamic candidate list during
	// insertion. Higher => better graph quality (higher recall later) but slower
	// builds. It has no effect on query-time memory. Typical 100–400.
	EfConstruction int

	// EfSearch is the default size of the dynamic candidate list during search.
	// Higher => better recall, slower queries. Can be overridden per-query.
	// Must be >= topK to return topK results. Typical 50–200.
	EfSearch int

	// Seed makes level assignment (and therefore the whole graph) deterministic
	// in tests. Zero means "use a fixed default seed" so builds are
	// reproducible; callers who want randomness pass a varying seed.
	Seed int64
}

// DefaultConfig returns sensible defaults for the given dimension and metric.
func DefaultConfig(dim int, metric distance.Metric) Config {
	return Config{
		Dim:            dim,
		Metric:         metric,
		M:              16,
		EfConstruction: 200,
		EfSearch:       100,
		Seed:           42,
	}
}

// Result is one hit from a search: the caller-facing id and its distance.
type Result struct {
	ID       uint64
	Distance float32
}

// Index is a thread-safe HNSW graph.
//
// Internally, every node is referred to by a dense uint32 "internal index" into
// the parallel slices below. The caller-facing uint64 id is mapped through
// idToIdx / ids. Using a dense internal index keeps the adjacency lists compact
// (uint32 instead of uint64) and lets us store vectors in a flat slice.
type Index struct {
	cfg Config

	mu  sync.RWMutex
	rng *rand.Rand
	// ml is the level-generation normalization factor, 1/ln(M). See randomLevel.
	ml float64
	// mMax0 is the max number of links a node may keep at layer 0. The paper
	// uses 2*M at the base layer (it has the most nodes and needs denser links).
	mMax0 int

	// Parallel arrays indexed by internal index.
	vectors [][]float32 // the stored vectors (already normalized if cosine)
	ids     []uint64    // internal index -> external id
	// links[internalIdx][level] = neighbor internal indices at that level.
	links   [][][]uint32
	deleted []bool // soft-delete tombstones

	idToIdx map[uint64]uint32

	entryPoint uint32 // internal index of the current top-level entry node
	maxLevel   int    // the highest layer that currently has any node
	count      int    // number of live (non-deleted) vectors
}

var (
	// ErrDimMismatch is returned when a vector's length != the index dimension.
	ErrDimMismatch = errors.New("hnsw: vector dimension mismatch")
	// ErrDuplicateID is returned by Insert when the id already exists.
	ErrDuplicateID = errors.New("hnsw: duplicate id")
)

// New creates an empty index with the given configuration.
func New(cfg Config) *Index {
	if cfg.M <= 0 {
		cfg.M = 16
	}
	if cfg.EfConstruction <= 0 {
		cfg.EfConstruction = 200
	}
	if cfg.EfSearch <= 0 {
		cfg.EfSearch = 100
	}
	if cfg.Seed == 0 {
		cfg.Seed = 42
	}
	return &Index{
		cfg:        cfg,
		rng:        rand.New(rand.NewSource(cfg.Seed)),
		ml:         1.0 / math.Log(float64(cfg.M)),
		mMax0:      cfg.M * 2,
		idToIdx:    make(map[uint64]uint32),
		entryPoint: 0,
		maxLevel:   -1, // -1 means "empty; no entry point yet"
	}
}

// Len returns the number of live vectors in the index.
func (h *Index) Len() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.count
}

// randomLevel draws the top layer for a new node from a geometric distribution.
//
// The probability of a node reaching layer L decays exponentially:
// P(level >= L) = exp(-L / ml)^... — concretely we sample
// floor(-ln(U) * ml) where U ~ Uniform(0,1]. With ml = 1/ln(M), a node lands
// at layer 0 with probability ~1 - 1/M, at layer 1 with ~1/M, layer 2 with
// ~1/M², and so on. This is what makes the upper layers exponentially sparse —
// the "express highway" property that gives O(log N) search.
func (h *Index) randomLevel() int {
	r := h.rng.Float64()
	// r is in [0,1); avoid ln(0) by nudging.
	if r <= 0 {
		r = 1e-12
	}
	return int(-math.Log(r) * h.ml)
}

// distTo computes the distance from an arbitrary query vector to the stored
// vector at internal index idx.
func (h *Index) distTo(query []float32, idx uint32) float32 {
	return distance.Distance(h.cfg.Metric, query, h.vectors[idx])
}

// Insert adds a vector under the given caller id. The vector is copied, so the
// caller may reuse its slice. For cosine collections the caller is responsible
// for having normalized the vector first (the collection layer does this).
func (h *Index) Insert(id uint64, vector []float32) error {
	if len(vector) != h.cfg.Dim {
		return ErrDimMismatch
	}
	h.mu.Lock()
	defer h.mu.Unlock()

	if _, exists := h.idToIdx[id]; exists {
		return ErrDuplicateID
	}

	// Copy the vector so later caller mutation cannot corrupt the index.
	v := make([]float32, len(vector))
	copy(v, vector)

	newLevel := h.randomLevel()

	// Allocate the new node's slots in every parallel array.
	idx := uint32(len(h.vectors))
	h.vectors = append(h.vectors, v)
	h.ids = append(h.ids, id)
	h.deleted = append(h.deleted, false)
	h.idToIdx[id] = idx
	// Pre-size the per-level link slices: levels 0..newLevel.
	nodeLinks := make([][]uint32, newLevel+1)
	h.links = append(h.links, nodeLinks)
	h.count++

	// First node ever: it becomes the entry point and we're done.
	if h.maxLevel == -1 {
		h.entryPoint = idx
		h.maxLevel = newLevel
		return nil
	}

	// Phase 1: greedy descent from the top down to newLevel+1. On these upper
	// layers we do NOT insert links; we only walk closer to refine the entry
	// point for the lower layers. Beam width 1 (pure greedy) is enough here.
	curr := h.entryPoint
	currDist := h.distTo(v, curr)
	for lvl := h.maxLevel; lvl > newLevel; lvl-- {
		curr, currDist = h.greedyDescend(v, curr, currDist, lvl)
	}

	// Phase 2: from min(newLevel, maxLevel) down to 0, run a full beam search to
	// collect efConstruction candidates, pick M of them as neighbors, and wire
	// up bidirectional links (pruning over-connected neighbors).
	entryPoints := []candidate{{node: curr, dist: currDist}}
	startLevel := newLevel
	if startLevel > h.maxLevel {
		startLevel = h.maxLevel
	}
	for lvl := startLevel; lvl >= 0; lvl-- {
		candidates := h.searchLayer(v, entryPoints, h.cfg.EfConstruction, lvl, nil)

		mMax := h.cfg.M
		if lvl == 0 {
			mMax = h.mMax0
		}
		selected := h.selectNeighborsHeuristic(v, candidates, h.cfg.M, lvl)

		// Link the new node -> selected.
		h.links[idx][lvl] = selected

		// Link selected -> new node (bidirectional), pruning if a neighbor now
		// exceeds its max degree.
		for _, nb := range selected {
			h.links[nb][lvl] = append(h.links[nb][lvl], idx)
			if len(h.links[nb][lvl]) > mMax {
				h.links[nb][lvl] = h.pruneNeighbors(nb, lvl, mMax)
			}
		}

		// The candidates found here seed the search on the next lower level.
		entryPoints = candidates
	}

	// If this node reached higher than any existing node, it becomes the new
	// entry point (the top of the express highway).
	if newLevel > h.maxLevel {
		h.entryPoint = idx
		h.maxLevel = newLevel
	}
	return nil
}

// greedyDescend performs a beam-width-1 greedy walk on a single layer: from the
// current node, repeatedly jump to the neighbor closest to the query, until no
// neighbor is closer than where we are. Returns the local minimum and its
// distance. Used on the upper layers during both insert and search, where we
// just want to refine the entry point cheaply.
func (h *Index) greedyDescend(query []float32, curr uint32, currDist float32, level int) (uint32, float32) {
	changed := true
	for changed {
		changed = false
		for _, nb := range h.links[curr][level] {
			d := h.distTo(query, nb)
			if d < currDist {
				currDist = d
				curr = nb
				changed = true
			}
		}
	}
	return curr, currDist
}

// searchLayer is the workhorse: a best-first beam search within a single layer.
//
// It maintains two structures (see heap.go for why two):
//   - candidates (min-heap): the frontier of nodes to explore, closest first.
//   - results   (max-heap): the ef best nodes found so far, farthest at top so
//     we can evict it when we overflow.
//
// The loop expands the closest candidate, and stops early when even the closest
// remaining candidate is farther than the worst result we already have — at
// that point nothing left to explore can improve the result set.
//
// admit decides whether a node is eligible to enter the *result* set. It does
// NOT affect traversal: non-admitted nodes are still explored as bridges toward
// admitted ones (this is what makes filtered search over the graph work). Pass
// nil to admit every live node (the ordinary unfiltered case).
//
// Returns up to ef candidates (unsorted; caller sorts or heuristically selects).
func (h *Index) searchLayer(query []float32, entryPoints []candidate, ef, level int, admit func(uint32) bool) []candidate {
	// visited guards against re-expanding a node. A map is simple and fine; a
	// production system would use a reusable visited-bitset pool to avoid
	// allocation, which guide 02 discusses.
	visited := make(map[uint32]struct{}, ef*2)

	admits := func(node uint32) bool {
		return !h.deleted[node] && (admit == nil || admit(node))
	}

	cands := &minHeap{}
	results := &maxHeap{}
	heap.Init(cands)
	heap.Init(results)

	for _, ep := range entryPoints {
		if _, seen := visited[ep.node]; seen {
			continue
		}
		visited[ep.node] = struct{}{}
		heap.Push(cands, ep)
		if admits(ep.node) {
			heap.Push(results, ep)
		}
	}

	for cands.Len() > 0 {
		nearest := heap.Pop(cands).(candidate)
		// Early termination: if the closest thing left to explore is farther
		// than our current worst kept result (and we already have ef of them),
		// we can never improve — stop. When filtering, results fills more slowly
		// (fewer nodes are admitted), so we naturally explore more of the graph
		// before this triggers.
		if results.Len() >= ef && nearest.dist > results.peek().dist {
			break
		}

		for _, nb := range h.links[nearest.node][level] {
			if _, seen := visited[nb]; seen {
				continue
			}
			visited[nb] = struct{}{}
			d := h.distTo(query, nb)

			// Explore nb (push to frontier) if it could still be useful, i.e. we
			// don't yet have ef results or nb beats our current worst result.
			if results.Len() < ef || d < results.peek().dist {
				heap.Push(cands, candidate{node: nb, dist: d})
				if admits(nb) {
					heap.Push(results, candidate{node: nb, dist: d})
					if results.Len() > ef {
						heap.Pop(results) // evict the farthest
					}
				}
			}
		}
	}

	out := make([]candidate, results.Len())
	copy(out, *results)
	return out
}

// selectNeighborsHeuristic chooses up to m neighbors from the candidate set,
// using the paper's heuristic (Algorithm 4) rather than a naive "m closest".
//
// The naive approach (just take the m nearest) tends to create clusters of
// mutually-close nodes and leaves "gaps" in the graph, hurting reachability.
// The heuristic instead keeps a candidate only if it is closer to the query
// than it is to any already-selected neighbor. This favors a *diverse* spread
// of neighbors in different directions, which keeps the graph well-connected and
// improves recall — especially at low M.
func (h *Index) selectNeighborsHeuristic(query []float32, candidates []candidate, m, level int) []uint32 {
	if len(candidates) <= m {
		// Nothing to prune; still sort so the nearest is first (used as the
		// entry point for the next layer during insert).
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].dist < candidates[j].dist
		})
		out := make([]uint32, len(candidates))
		for i, c := range candidates {
			out[i] = c.node
		}
		return out
	}

	// Work through candidates nearest-first.
	sorted := make([]candidate, len(candidates))
	copy(sorted, candidates)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].dist < sorted[j].dist })

	selected := make([]uint32, 0, m)
	for _, c := range sorted {
		if len(selected) >= m {
			break
		}
		good := true
		for _, s := range selected {
			// distToQuery is c.dist; distToSelected is dist(candidate, selected).
			distToSelected := distance.Distance(h.cfg.Metric, h.vectors[c.node], h.vectors[s])
			if distToSelected < c.dist {
				// c is closer to an already-selected neighbor than to the query:
				// it would be redundant, skip it to preserve diversity.
				good = false
				break
			}
		}
		if good {
			selected = append(selected, c.node)
		}
	}
	return selected
}

// pruneNeighbors re-selects the best mMax neighbors for a node that has grown
// over its degree budget. It reuses the diversity heuristic so pruning keeps the
// graph well spread rather than just trimming the farthest links.
func (h *Index) pruneNeighbors(node uint32, level, mMax int) []uint32 {
	current := h.links[node][level]
	cands := make([]candidate, 0, len(current))
	for _, nb := range current {
		cands = append(cands, candidate{node: nb, dist: h.distTo(h.vectors[node], nb)})
	}
	return h.selectNeighborsHeuristic(h.vectors[node], cands, mMax, level)
}

// Search returns the topK nearest neighbors of the query vector. efSearch<=0
// uses the index's configured default. The returned results are sorted nearest
// first.
func (h *Index) Search(query []float32, topK, efSearch int) ([]Result, error) {
	if len(query) != h.cfg.Dim {
		return nil, ErrDimMismatch
	}
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.maxLevel == -1 || h.count == 0 {
		return nil, nil // empty index
	}
	if efSearch <= 0 {
		efSearch = h.cfg.EfSearch
	}
	if efSearch < topK {
		efSearch = topK // must explore at least topK to return topK
	}

	// Phase 1: greedy descent through the upper layers to find a good entry
	// point into layer 0.
	curr := h.entryPoint
	currDist := h.distTo(query, curr)
	for lvl := h.maxLevel; lvl > 0; lvl-- {
		curr, currDist = h.greedyDescend(query, curr, currDist, lvl)
	}

	// Phase 2: wide beam search at layer 0.
	found := h.searchLayer(query, []candidate{{node: curr, dist: currDist}}, efSearch, 0, nil)

	sort.Slice(found, func(i, j int) bool { return found[i].dist < found[j].dist })
	if len(found) > topK {
		found = found[:topK]
	}
	out := make([]Result, len(found))
	for i, c := range found {
		out[i] = Result{ID: h.ids[c.node], Distance: c.dist}
	}
	return out, nil
}

// SearchFiltered runs the same graph search as Search, but only ids for which
// allow(id) returns true are eligible to appear in the results. Non-matching
// nodes are still traversed as bridges, so matches on the far side of a filtered
// region are still reachable. This is the "filtered HNSW" approach Qdrant uses
// for medium-selectivity filters (see docs/learning/04).
//
// Because the result set fills more slowly when the filter is selective, the
// search explores more of the graph; callers should raise efSearch for filters,
// and fall back to an exact pre-filter scan when the filter is very selective.
func (h *Index) SearchFiltered(query []float32, topK, efSearch int, allow func(id uint64) bool) ([]Result, error) {
	if len(query) != h.cfg.Dim {
		return nil, ErrDimMismatch
	}
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.maxLevel == -1 || h.count == 0 {
		return nil, nil
	}
	if efSearch <= 0 {
		efSearch = h.cfg.EfSearch
	}
	if efSearch < topK {
		efSearch = topK
	}

	// Translate the caller's id predicate into an internal-index predicate once.
	admit := func(idx uint32) bool { return allow(h.ids[idx]) }

	curr := h.entryPoint
	currDist := h.distTo(query, curr)
	for lvl := h.maxLevel; lvl > 0; lvl-- {
		curr, currDist = h.greedyDescend(query, curr, currDist, lvl)
	}
	found := h.searchLayer(query, []candidate{{node: curr, dist: currDist}}, efSearch, 0, admit)

	sort.Slice(found, func(i, j int) bool { return found[i].dist < found[j].dist })
	if len(found) > topK {
		found = found[:topK]
	}
	out := make([]Result, len(found))
	for i, c := range found {
		out[i] = Result{ID: h.ids[c.node], Distance: c.dist}
	}
	return out, nil
}

// BruteForceSubset computes the exact topK nearest neighbors among only the
// given ids. This is the "pre-filter" path: when a metadata filter is very
// selective (matches few vectors), it is faster and *exact* to score just those
// vectors directly than to navigate the graph avoiding everything else. Unknown
// or deleted ids are skipped.
func (h *Index) BruteForceSubset(query []float32, ids []uint64, topK int) ([]Result, error) {
	if len(query) != h.cfg.Dim {
		return nil, ErrDimMismatch
	}
	h.mu.RLock()
	defer h.mu.RUnlock()

	all := make([]Result, 0, len(ids))
	for _, id := range ids {
		idx, ok := h.idToIdx[id]
		if !ok || h.deleted[idx] {
			continue
		}
		all = append(all, Result{ID: id, Distance: h.distTo(query, idx)})
	}
	sort.Slice(all, func(i, j int) bool { return all[i].Distance < all[j].Distance })
	if len(all) > topK {
		all = all[:topK]
	}
	return all, nil
}

// Delete soft-deletes a vector by id. The node stays in the graph (removing it
// would require expensive edge repair), but it is excluded from result sets and
// its id is unmapped so the id can be reused. Returns false if id was unknown.
//
// Soft deletion is what Qdrant/Weaviate do too; a background "optimizer" later
// rebuilds segments to physically reclaim space. We discuss this in guide 06.
func (h *Index) Delete(id uint64) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	idx, ok := h.idToIdx[id]
	if !ok {
		return false
	}
	h.deleted[idx] = true
	delete(h.idToIdx, id)
	h.count--
	return true
}

// BruteForce computes the exact topK nearest neighbors by scanning every live
// vector. It is O(N) per query — far too slow for production — but it is the
// *ground truth* against which we measure HNSW's recall. Never delete this: a
// probabilistic index is only trustworthy if you can compare it to an exact
// answer. Used by tests and the benchmark tool.
func (h *Index) BruteForce(query []float32, topK int) ([]Result, error) {
	if len(query) != h.cfg.Dim {
		return nil, ErrDimMismatch
	}
	h.mu.RLock()
	defer h.mu.RUnlock()

	all := make([]Result, 0, h.count)
	for idx := range h.vectors {
		if h.deleted[idx] {
			continue
		}
		all = append(all, Result{
			ID:       h.ids[idx],
			Distance: h.distTo(query, uint32(idx)),
		})
	}
	sort.Slice(all, func(i, j int) bool { return all[i].Distance < all[j].Distance })
	if len(all) > topK {
		all = all[:topK]
	}
	return all, nil
}

// Config returns a copy of the index configuration.
func (h *Index) Config() Config { return h.cfg }
