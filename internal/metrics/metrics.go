// Package metrics exposes Prometheus instrumentation for the server: per-RPC
// latency and counts, plus vector-count and index-build gauges. Observability is
// not optional for infrastructure — you cannot operate what you cannot measure.
//
// See docs/learning/08-metrics-and-benchmarks.md.
package metrics

import (
	"context"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/status"
)

var (
	// RequestDuration is a histogram of RPC latency, labeled by method and
	// status code. Histograms (not just averages) let us compute p50/p95/p99 —
	// tail latency is what users actually feel, and an average hides it.
	//
	// The buckets are tuned for a vector search service: sub-millisecond to a few
	// hundred ms. Pick buckets around your SLOs, not the defaults.
	RequestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "vectordb",
		Name:      "request_duration_seconds",
		Help:      "Duration of gRPC requests by method and status code.",
		Buckets:   []float64{0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5},
	}, []string{"method", "code"})

	// RequestsTotal counts RPCs by method and status code.
	RequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "vectordb",
		Name:      "requests_total",
		Help:      "Total gRPC requests by method and status code.",
	}, []string{"method", "code"})

	// VectorsTotal is the live vector count per collection.
	VectorsTotal = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "vectordb",
		Name:      "vectors_total",
		Help:      "Number of live vectors per collection.",
	}, []string{"collection"})

	// IndexBuildSeconds records how long a full index build/rebuild took.
	IndexBuildSeconds = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "vectordb",
		Name:      "index_build_seconds",
		Help:      "Time to build/rebuild a collection's index.",
		Buckets:   prometheus.DefBuckets,
	}, []string{"collection"})
)

// Serve starts an HTTP server exposing /metrics in Prometheus text format.
// It blocks; run it in a goroutine.
func Serve(addr string) error {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	srv := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	return srv.ListenAndServe()
}

// UnaryServerInterceptor returns a gRPC interceptor that records the duration
// and outcome of every unary RPC. An interceptor is gRPC's middleware: it wraps
// each handler so we instrument all methods in one place instead of sprinkling
// timing code through the service.
func UnaryServerInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		start := time.Now()
		resp, err := handler(ctx, req)
		code := status.Code(err).String()
		RequestDuration.WithLabelValues(info.FullMethod, code).Observe(time.Since(start).Seconds())
		RequestsTotal.WithLabelValues(info.FullMethod, code).Inc()
		return resp, err
	}
}

// SetVectorCount updates the live vector gauge for a collection.
func SetVectorCount(collection string, n int) {
	VectorsTotal.WithLabelValues(collection).Set(float64(n))
}

// ObserveIndexBuild records an index build duration for a collection.
func ObserveIndexBuild(collection string, d time.Duration) {
	IndexBuildSeconds.WithLabelValues(collection).Observe(d.Seconds())
}
