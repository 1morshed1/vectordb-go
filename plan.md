## Project E: Embedding & Vector Search Infrastructure (Go)

### Why This Project

You use pgvector, FAISS, and ChromaDB in every RAG project. But have you ever thought about what FAISS actually does internally? How an HNSW index works? How vector databases shard and replicate embedding data?

This project builds a standalone vector search service in Go — a simplified Qdrant or Weaviate. It stores high-dimensional vectors, builds an HNSW index for approximate nearest neighbor search, supports metadata filtering, and exposes a gRPC API. It's the infrastructure that powers every RAG system you've ever built.

This is the perfect "intersection" project: you understand the ML side (what embeddings are, why cosine similarity works, what dimensionality matters), and you build the infrastructure side (the index data structure, disk persistence, distributed sharding) in Go.

### Architecture

```
  RAG Applications                                                             
   │                                                                           
   │  gRPC: Insert(vector, metadata)                                          
   │  gRPC: Search(query_vector, top_k, filters)                              
   ▼                                                                           
┌──────────────────────────────────────────────────────────────────────────────┐
│  Go Vector Search Server                                                     │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  gRPC API                                                            │   │
│  │                                                                      │   │
│  │  rpc Insert(InsertRequest) returns (InsertResponse)                  │   │
│  │  rpc Search(SearchRequest) returns (SearchResponse)                  │   │
│  │  rpc Delete(DeleteRequest) returns (DeleteResponse)                  │   │
│  │  rpc CreateCollection(CollectionConfig) returns (Collection)         │   │
│  └──────────────────────────────────┬───────────────────────────────────┘    │
│                                     │                                        │
│                                     ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  Collection Manager                                                  │   │
│  │                                                                      │   │
│  │  Collection "documents" {                                            │   │
│  │    dimension: 1536,                                                  │   │
│  │    distance_metric: cosine,                                          │   │
│  │    index_type: HNSW,                                                 │   │
│  │    hnsw_config: { M: 16, ef_construction: 200, ef_search: 100 }     │   │
│  │  }                                                                   │   │
│  └──────────────────────────────────┬───────────────────────────────────┘    │
│                                     │                                        │
│                                     ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  HNSW Index Engine                                                   │   │
│  │                                                                      │   │
│  │  Multi-layered navigable small world graph:                          │   │
│  │                                                                      │   │
│  │  Layer 3:  ●───────────────────●  (few nodes, long-range links)     │   │
│  │                                                                      │   │
│  │  Layer 2:  ●───────●───────────●──────●                              │   │
│  │                                                                      │   │
│  │  Layer 1:  ●──●──●──●──●──●──●──●──●──●                             │   │
│  │                                                                      │   │
│  │  Layer 0:  ●●●●●●●●●●●●●●●●●●●●●●●●●  (all nodes, local links)    │   │
│  │                                                                      │   │
│  │  Search: enter at top layer, greedily descend to nearest neighbor,  │   │
│  │  then drop to next layer and repeat. At layer 0, return top-K.      │   │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  Storage Layer                                                       │   │
│  │                                                                      │   │
│  │  - Vectors: memory-mapped flat file (fast sequential access)         │   │
│  │  - HNSW graph: memory-mapped adjacency lists                         │   │
│  │  - Metadata: embedded key-value store (BoltDB or Badger)             │   │
│  │  - WAL: write-ahead log for crash recovery                           │   │
│  │                                                                      │   │
│  │  On-disk layout:                                                     │   │
│  │  collection-abc/                                                     │   │
│  │  ├── vectors.dat        # raw float32 vectors, mmap'd               │   │
│  │  ├── hnsw.dat           # graph edges, mmap'd                        │   │
│  │  ├── metadata.db        # BoltDB key-value store                     │   │
│  │  └── wal/               # write-ahead log segments                   │   │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  Metadata Filter Engine                                              │   │
│  │                                                                      │   │
│  │  Search(vector, top_k=10, filter="category == 'finance' AND          │   │
│  │                                    year >= 2024")                     │   │
│  │                                                                      │   │
│  │  Strategy:                                                           │   │
│  │  - Pre-filter: apply metadata filter first, then search only         │   │
│  │    matching vectors (fast when filter is very selective)              │   │
│  │  - Post-filter: search all vectors, then filter results              │   │
│  │    (fast when filter is not selective)                                │   │
│  │  - Auto: estimate selectivity, choose the better strategy            │   │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Prometheus: query_latency, vectors_total, index_build_time, memory_usage   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### The HNSW Algorithm (the core of the project)

HNSW (Hierarchical Navigable Small World) is the algorithm behind FAISS's `IndexHNSWFlat`, Qdrant, Weaviate, and pgvector. It builds a multi-layered graph where each layer is a "small world" — any node can reach any other node in O(log N) hops.

```go
type HNSWIndex struct {
    layers       []map[uint64][]uint64    // layer -> node_id -> neighbor_ids
    vectors      *VectorStore              // mmap'd vector storage
    maxLayer     int
    M            int                       // max connections per node per layer
    efConstruct  int                       // beam width during construction
    efSearch     int                       // beam width during search
    entryPoint   uint64                    // top-level entry node
    mu           sync.RWMutex
}

func (h *HNSWIndex) Insert(id uint64, vector []float32) {
    h.mu.Lock()
    defer h.mu.Unlock()
    
    // Determine which layer this node enters at
    // (random, with exponential decay — most nodes only at layer 0)
    nodeLayer := h.randomLevel()
    
    // Find nearest neighbors at each layer (top-down search)
    currentNode := h.entryPoint
    
    // Traverse from top layer down to nodeLayer+1 (greedy search, no insertion)
    for layer := h.maxLayer; layer > nodeLayer; layer-- {
        currentNode = h.searchLayerGreedy(vector, currentNode, layer)
    }
    
    // From nodeLayer down to 0, find neighbors and connect
    for layer := min(nodeLayer, h.maxLayer); layer >= 0; layer-- {
        neighbors := h.searchLayer(vector, currentNode, layer, h.efConstruct)
        
        // Select M best neighbors (closest by distance)
        selected := h.selectNeighbors(vector, neighbors, h.M)
        
        // Add bidirectional edges
        h.layers[layer][id] = selected
        for _, neighborID := range selected {
            h.layers[layer][neighborID] = append(h.layers[layer][neighborID], id)
            
            // Prune if neighbor has too many connections
            if len(h.layers[layer][neighborID]) > h.M*2 {
                h.layers[layer][neighborID] = h.pruneConnections(
                    neighborID, h.layers[layer][neighborID], h.M, layer,
                )
            }
        }
        
        currentNode = neighbors[0]  // use nearest neighbor as entry for next layer
    }
    
    // Update entry point if this node is at a higher layer
    if nodeLayer > h.maxLayer {
        h.entryPoint = id
        h.maxLayer = nodeLayer
    }
}

func (h *HNSWIndex) Search(query []float32, topK int) []SearchResult {
    h.mu.RLock()
    defer h.mu.RUnlock()
    
    // Start at top layer, greedy search down to layer 1
    currentNode := h.entryPoint
    for layer := h.maxLayer; layer > 0; layer-- {
        currentNode = h.searchLayerGreedy(query, currentNode, layer)
    }
    
    // At layer 0, do a beam search with ef_search width
    candidates := h.searchLayer(query, currentNode, 0, h.efSearch)
    
    // Return top-K by distance
    sort.Slice(candidates, func(i, j int) bool {
        return candidates[i].Distance < candidates[j].Distance
    })
    
    if len(candidates) > topK {
        candidates = candidates[:topK]
    }
    return candidates
}

func (h *HNSWIndex) randomLevel() int {
    // Exponential decay: P(layer L) = (1/M)^L
    level := 0
    for rand.Float64() < 1.0/float64(h.M) {
        level++
    }
    return level
}
```

### What Your ML Background Gives You Here

- You know that cosine similarity is `1 - cosine_distance`, and that for normalized vectors, dot product = cosine similarity. So you can optimize: normalize all vectors at insertion time, then use dot product (which is faster — no square root) for search.
- You know that 1536-dimensional embeddings from `text-embedding-3-small` can be quantized to int8 with minimal quality loss, reducing memory by 4x. Build quantization support.
- You know that different embedding models produce different distributions. You can write a benchmark comparing search quality (recall@10) of your HNSW index vs FAISS vs pgvector on the same dataset with the same embeddings.

### Benchmarks

- **Recall@10 at various ef_search values:** how accuracy trades off with speed.
- **Queries per second:** at 100K, 1M, 10M vectors. Compare with FAISS.
- **Index build time:** at various M and ef_construction values.
- **Memory usage:** bytes per vector (lower is better). Compare with and without quantization.

---
---