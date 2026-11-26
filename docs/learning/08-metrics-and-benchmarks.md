# Learning Guide 08 — Metrics & Benchmarks

**Packages:** `internal/metrics`, `cmd/vectordb-bench`

You cannot operate — or tune — what you cannot measure. This guide covers the
two kinds of measurement an infrastructure service needs: **live metrics** (what
is happening right now, in production) and **benchmarks** (how the index behaves
under controlled load).

## Part A — Live metrics (`internal/metrics`)

We expose Prometheus metrics, the de-facto standard for infrastructure. Four
signals, straight from the spec:

| Metric | Type | Why |
|--------|------|-----|
| `vectordb_request_duration_seconds` | histogram | latency by method + code |
| `vectordb_requests_total` | counter | throughput + error rate |
| `vectordb_vectors_total` | gauge | dataset size per collection |
| `vectordb_index_build_seconds` | histogram | build cost |

### Why a histogram, not an average

`request_duration_seconds` is a **histogram**, so we can compute p50/p95/**p99**.
Averages lie: a service can have a 5ms average while 1% of users wait 2s. Tail
latency is what users feel, and only a histogram exposes it. The buckets are
hand-picked around a search service's real range (100µs to a couple of seconds)
— **always set buckets near your SLOs**, because the default buckets are rarely
right for your workload.

### One interceptor instruments every RPC

Rather than sprinkle timing code into each handler, `UnaryServerInterceptor`
wraps every RPC (gRPC's version of middleware). It records duration and outcome
in one place, labeled by `method` and gRPC `code`. `main.go` installs it with
`grpc.UnaryInterceptor(...)`. Add a method to the service and it's instrumented
automatically.

### The endpoint

`metrics.Serve` runs a plain HTTP server exposing `/metrics` in Prometheus text
format. A Prometheus server scrapes that URL every few seconds; Grafana graphs
it. The server binary starts it on `--metrics-addr` (default `:9090`).

## Part B — Benchmarks (`cmd/vectordb-bench`)

The benchmark measures the four numbers that decide if an ANN index is any good.
Run it:

```
go run ./cmd/vectordb-bench --n 10000 --dim 128 --k 10 --queries 200
```

### 1. Recall@k vs ef_search — the master trade-off

Every query's HNSW top-k is compared to the **exact brute-force top-k** (the only
honest ground truth). A representative run:

```
ef_search  recall@10  QPS    avg latency
10         0.21       11299  88µs
20         0.33       6646   150µs
50         0.56       2723   367µs
100        0.76       1435   697µs
200        0.92       875    1.14ms
400        0.99       557    1.80ms
```

This curve is the single most important thing to understand about a vector
database: **`ef_search` buys recall with latency.** You choose the point on this
curve that matches your product — a chatbot might accept 0.9 recall for 4× the
throughput; a legal-search tool might pay for 0.99. There is no single "correct"
setting; there is a curve, and you pick.

> **Read the numbers honestly:** these are **random Gaussian vectors**, which are
> a *worst case* for ANN. In high dimensions random points are all nearly
> equidistant (the "curse of dimensionality"), so the true top-10 are barely
> closer than ranks 11–100 and easy to miss. **Real embeddings cluster** by
> meaning, giving the graph real structure to exploit — production recall at a
> given ef is typically much higher than on random data. The *shape* of the
> curve (rising recall, falling QPS) is what to internalize, not the absolute
> values on synthetic data.

### 2. QPS and latency

Measured in the same loop as recall, so every recall number has its throughput
cost attached. QPS is single-threaded here; a real deployment fans queries
across cores (search takes only the read lock, so it scales well with readers).

### 3. Index build time

Timed over all inserts, reported as vectors/sec. Build cost grows with `M` and
`ef_construction` — the knobs that also raise query recall. Our build is
deliberately un-optimized (per-search `visited` map allocation, guide 02); the
number to watch is how it *scales*, and where to optimize (the `visited` set) if
you needed to.

### 4. Memory: float32 vs int8

```
float32:      512 bytes
int8 (quant): 132 bytes  (3.88x smaller)
```

Plus the quantized-recall line: scoring every vector with int8 dot products and
comparing to the float32 ground truth gives **recall ≈ 0.98** — you keep
essentially the same answers at a quarter of the memory (guide 05). Because the
benchmark normalizes vectors and uses the dot metric throughout, that comparison
is apples-to-apples against the same ground truth.

## How to use these together

1. Run the benchmark to find the `ef_search` that hits your target recall.
2. Deploy with that default; watch `request_duration_seconds` p99 and
   `requests_total` error rate in production.
3. If p99 creeps up as `vectors_total` grows, re-run the benchmark at the new
   scale and re-pick `ef_search` — or add quantization to fit more vectors in
   RAM.

## Takeaways

- Histograms, not averages: tail latency is the truth, and buckets go near your
  SLOs.
- One interceptor instruments every RPC; add methods freely.
- Recall vs ef_search is a *curve you choose a point on*, not a fixed setting.
- Always measure recall against exact brute force.
- Random vectors are an ANN worst case; real embeddings do better — learn the
  curve's shape, not the synthetic absolute numbers.
- int8 keeps ~0.98 recall at 4× less memory.

This is the last package guide. See the repo `README.md` for how to run the
whole thing, and `docs/learning/00-overview.md` for the map.
