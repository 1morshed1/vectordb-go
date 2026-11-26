// Command vectordb-bench measures the four numbers that decide whether an ANN
// index is any good, exactly the ones listed in the original spec:
//
//   - Recall@k at various ef_search — how accuracy trades off against speed.
//   - Queries per second (QPS)      — throughput at a given recall.
//   - Index build time              — how long construction takes.
//   - Memory usage (bytes/vector)   — float32 vs int8 quantized.
//
// It builds an index over random vectors, then reports each metric. Recall is
// always measured against the exact brute-force answer, which is the only
// trustworthy ground truth for an approximate index.
//
// Usage:
//
//	go run ./cmd/vectordb-bench --n 50000 --dim 128 --k 10 --queries 500
//
// See docs/learning/08-metrics-and-benchmarks.md.
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"text/tabwriter"
	"time"

	"github.com/1morshed1/vectordb-go/internal/distance"
	"github.com/1morshed1/vectordb-go/internal/hnsw"
	"github.com/1morshed1/vectordb-go/internal/quantize"
)

func main() {
	n := flag.Int("n", 20000, "number of vectors to index")
	dim := flag.Int("dim", 128, "vector dimensionality")
	k := flag.Int("k", 10, "top-k for recall measurement")
	queries := flag.Int("queries", 300, "number of query vectors")
	m := flag.Int("m", 16, "HNSW M (max links per layer)")
	efc := flag.Int("ef-construction", 200, "HNSW efConstruction")
	seed := flag.Int64("seed", 42, "random seed")
	flag.Parse()

	fmt.Printf("vectordb-bench: n=%d dim=%d k=%d queries=%d M=%d efConstruction=%d\n\n",
		*n, *dim, *k, *queries, *m, *efc)

	rng := rand.New(rand.NewSource(*seed))
	data := randomVectors(rng, *n, *dim)
	qs := randomVectors(rng, *queries, *dim)
	// Normalize everything and use the dot-product metric throughout. This makes
	// the benchmark a cosine benchmark (guide 01), and — crucially — lets the
	// int8 quantized recall below be compared against the *same* dot-product
	// ground truth, an honest apples-to-apples comparison.
	for _, v := range data {
		distance.Normalize(v)
	}
	for _, v := range qs {
		distance.Normalize(v)
	}

	// ---- Build time ----
	cfg := hnsw.DefaultConfig(*dim, distance.DotProduct)
	cfg.M = *m
	cfg.EfConstruction = *efc
	cfg.Seed = *seed
	idx := hnsw.New(cfg)

	buildStart := time.Now()
	for i, v := range data {
		if err := idx.Insert(uint64(i), v); err != nil {
			fmt.Fprintln(os.Stderr, "insert:", err)
			os.Exit(1)
		}
	}
	buildDur := time.Since(buildStart)
	fmt.Printf("Index build:   %v  (%.0f vectors/sec)\n\n",
		buildDur.Round(time.Millisecond), float64(*n)/buildDur.Seconds())

	// ---- Ground truth (exact top-k for every query) ----
	truth := make([]map[uint64]bool, len(qs))
	for i, q := range qs {
		exact, _ := idx.BruteForce(q, *k)
		set := make(map[uint64]bool, *k)
		for _, r := range exact {
			set[r.ID] = true
		}
		truth[i] = set
	}

	// ---- Recall & QPS across ef_search ----
	fmt.Println("Recall@k and QPS vs ef_search:")
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "ef_search\trecall@"+itoa(*k)+"\tQPS\tavg latency")
	fmt.Fprintln(tw, "---------\t---------\t---\t-----------")
	for _, ef := range []int{10, 20, 50, 100, 200, 400} {
		recall, qps, avgLat := measure(idx, qs, truth, *k, ef)
		fmt.Fprintf(tw, "%d\t%.4f\t%.0f\t%v\n", ef, recall, qps, avgLat.Round(time.Microsecond))
	}
	tw.Flush()

	// ---- Memory ----
	fmt.Println("\nMemory per vector:")
	f32 := quantize.Float32Bytes(*dim)
	// Quantize one representative vector to get the int8 footprint.
	q8 := quantize.Quantize(data[0])
	fmt.Printf("  float32:      %d bytes\n", f32)
	fmt.Printf("  int8 (quant): %d bytes  (%.2fx smaller)\n", q8.MemoryBytes(), float64(f32)/float64(q8.MemoryBytes()))
	fmt.Printf("  graph edges:  ~%d bytes  (est. M*2 uint32 at layer 0)\n", *m*2*4)

	// Quantization recall sanity check: does int8 preserve the ranking?
	qRecall := quantizedRecall(data, qs, truth, *k)
	fmt.Printf("\nQuantized (int8) exact-scan recall@%d vs float32 truth: %.4f\n", *k, qRecall)
}

// measure runs all queries at a given ef and returns recall, QPS, and average
// latency.
func measure(idx *hnsw.Index, qs [][]float32, truth []map[uint64]bool, k, ef int) (recall, qps float64, avgLat time.Duration) {
	var totalRecall float64
	start := time.Now()
	for i, q := range qs {
		res, _ := idx.Search(q, k, ef)
		hits := 0
		for _, r := range res {
			if truth[i][r.ID] {
				hits++
			}
		}
		totalRecall += float64(hits) / float64(k)
	}
	elapsed := time.Since(start)
	recall = totalRecall / float64(len(qs))
	qps = float64(len(qs)) / elapsed.Seconds()
	avgLat = elapsed / time.Duration(len(qs))
	return
}

// quantizedRecall scores every vector with int8 quantized dot products and
// compares the top-k to the float32 ground truth — the memory-vs-accuracy story.
func quantizedRecall(data, qs [][]float32, truth []map[uint64]bool, k int) float64 {
	qdata := make([]quantize.QuantizedVector, len(data))
	for i, v := range data {
		qdata[i] = quantize.Quantize(v)
	}
	type scored struct {
		id  uint64
		val float32
	}
	var total float64
	for qi, q := range qs {
		qq := quantize.Quantize(q)
		scores := make([]scored, len(qdata))
		for i := range qdata {
			// Score with the int8 quantized dot product and compare the top-k to
			// the float32 dot-product ground truth: a fair measure of how much
			// recall we give up for 4x less memory.
			scores[i] = scored{uint64(i), quantize.Dot(qq, qdata[i])}
		}
		sort.Slice(scores, func(a, b int) bool { return scores[a].val > scores[b].val })
		hits := 0
		for i := 0; i < k && i < len(scores); i++ {
			if truth[qi][scores[i].id] {
				hits++
			}
		}
		total += float64(hits) / float64(k)
	}
	return total / float64(len(qs))
}

func randomVectors(r *rand.Rand, n, dim int) [][]float32 {
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

func itoa(i int) string { return fmt.Sprintf("%d", i) }
