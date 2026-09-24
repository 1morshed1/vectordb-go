package hnsw

import "container/heap"

// candidate is a (node, distance) pair used throughout the HNSW search
// routines. `node` is an *internal* index into the index's flat arrays, not the
// caller-facing uint64 id. `dist` is a distance (smaller == closer), following
// the convention established in internal/distance.
type candidate struct {
	node uint32
	dist float32
}

// The HNSW search algorithm needs two priority queues at once:
//
//   - A *min-heap* over the candidate frontier ("C" in the paper): we always
//     want to expand the closest unexplored candidate next, because that is the
//     most promising direction to find even-closer nodes.
//
//   - A *max-heap* over the dynamic result set ("W" in the paper): we keep only
//     the `ef` closest results found so far, so when the set overflows we must
//     be able to drop the *farthest* one in O(log n). That means the farthest
//     must be at the top — a max-heap.
//
// We implement both over a []candidate using container/heap, differing only in
// the Less comparison. Keeping them as distinct concrete types (rather than a
// bool flag) makes the call sites read clearly: `minFrontier` vs `maxResults`.

// minHeap orders candidates so the *closest* (smallest dist) is at index 0.
type minHeap []candidate

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return h[i].dist < h[j].dist }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)        { *h = append(*h, x.(candidate)) }
func (h *minHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

// peek returns the closest candidate without removing it.
func (h minHeap) peek() candidate { return h[0] }

// maxHeap orders candidates so the *farthest* (largest dist) is at index 0.
type maxHeap []candidate

func (h maxHeap) Len() int           { return len(h) }
func (h maxHeap) Less(i, j int) bool { return h[i].dist > h[j].dist }
func (h maxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *maxHeap) Push(x any)        { *h = append(*h, x.(candidate)) }
func (h *maxHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

// peek returns the farthest candidate without removing it.
func (h maxHeap) peek() candidate { return h[0] }

// Compile-time assertions that our types satisfy heap.Interface.
var (
	_ heap.Interface = (*minHeap)(nil)
	_ heap.Interface = (*maxHeap)(nil)
)
