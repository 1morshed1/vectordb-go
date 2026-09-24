// Command vectordb-server runs the gRPC vector search service.
//
// Usage:
//
//	vectordb-server --addr :50051 --data ./data
//
// It opens (recovering) a Store at --data, registers the gRPC service, and
// serves until interrupted, taking a final snapshot of every collection on a
// clean shutdown so restarts are fast.
package main

import (
	"context"
	"flag"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	vectordbpb "github.com/1morshed1/vectordb-go/api/gen"
	"github.com/1morshed1/vectordb-go/internal/metrics"
	"github.com/1morshed1/vectordb-go/internal/server"
	"github.com/1morshed1/vectordb-go/internal/storage"

	"google.golang.org/grpc"
)

func main() {
	addr := flag.String("addr", ":50051", "gRPC listen address")
	dataDir := flag.String("data", "./data", "data directory for persistence")
	metricsAddr := flag.String("metrics-addr", ":9090", "Prometheus metrics HTTP address (empty to disable)")
	flag.Parse()

	store, err := storage.Open(*dataDir)
	if err != nil {
		log.Fatalf("open store: %v", err)
	}
	log.Printf("recovered collections: %v", store.Manager().List())

	// Start the Prometheus metrics endpoint (best-effort; a failure to bind is
	// logged but does not stop the main service).
	if *metricsAddr != "" {
		go func() {
			log.Printf("metrics listening on %s/metrics", *metricsAddr)
			if err := metrics.Serve(*metricsAddr); err != nil {
				log.Printf("metrics server stopped: %v", err)
			}
		}()
	}

	lis, err := net.Listen("tcp", *addr)
	if err != nil {
		log.Fatalf("listen %s: %v", *addr, err)
	}

	// The metrics interceptor records latency and counts for every RPC.
	grpcServer := grpc.NewServer(
		grpc.UnaryInterceptor(metrics.UnaryServerInterceptor()),
	)
	vectordbpb.RegisterVectorDBServer(grpcServer, server.New(store))

	// Graceful shutdown: on SIGINT/SIGTERM, stop accepting new RPCs, let
	// in-flight ones finish, snapshot everything, and close the store.
	go func() {
		sigc := make(chan os.Signal, 1)
		signal.Notify(sigc, syscall.SIGINT, syscall.SIGTERM)
		<-sigc
		log.Println("shutting down...")

		// GracefulStop drains in-flight RPCs. Guard it with a timeout so a stuck
		// stream can't hang shutdown forever.
		done := make(chan struct{})
		go func() { grpcServer.GracefulStop(); close(done) }()
		select {
		case <-done:
		case <-time.After(10 * time.Second):
			log.Println("graceful stop timed out; forcing")
			grpcServer.Stop()
		}

		for _, name := range store.Manager().List() {
			if err := store.Snapshot(name); err != nil {
				log.Printf("snapshot %s on shutdown: %v", name, err)
			}
		}
		if err := store.Close(); err != nil {
			log.Printf("close store: %v", err)
		}
		log.Println("bye")
	}()

	log.Printf("vectordb-server listening on %s (data=%s)", *addr, *dataDir)
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("serve: %v", err)
	}
	// Give the shutdown goroutine a moment to finish its snapshot work.
	_ = context.Background()
}
