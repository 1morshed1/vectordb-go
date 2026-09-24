package server

import (
	"context"
	"math/rand"
	"net"
	"testing"

	vectordbpb "github.com/1morshed1/vectordb-go/api/gen"
	"github.com/1morshed1/vectordb-go/internal/storage"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
)

// newTestClient spins up the full gRPC server over an in-memory bufconn (no real
// network) and returns a connected client. This exercises the real
// serialization + service code path end to end.
func newTestClient(t *testing.T) vectordbpb.VectorDBClient {
	t.Helper()
	store, err := storage.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	lis := bufconn.Listen(1 << 20)
	srv := grpc.NewServer()
	vectordbpb.RegisterVectorDBServer(srv, New(store))
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(func() { srv.Stop(); store.Close() })

	conn, err := grpc.NewClient(
		"passthrough:///bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) { return lis.DialContext(ctx) }),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close() })
	return vectordbpb.NewVectorDBClient(conn)
}

func mv(s string) *vectordbpb.MetadataValue {
	return &vectordbpb.MetadataValue{Value: &vectordbpb.MetadataValue_StringValue{StringValue: s}}
}
func mvNum(n float64) *vectordbpb.MetadataValue {
	return &vectordbpb.MetadataValue{Value: &vectordbpb.MetadataValue_NumberValue{NumberValue: n}}
}

func TestEndToEndInsertSearch(t *testing.T) {
	c := newTestClient(t)
	ctx := context.Background()

	if _, err := c.CreateCollection(ctx, &vectordbpb.CreateCollectionRequest{
		Name: "docs", Dim: 16, Metric: vectordbpb.DistanceMetric_DISTANCE_METRIC_COSINE,
	}); err != nil {
		t.Fatal(err)
	}

	r := rand.New(rand.NewSource(1))
	vecs := make([][]float32, 100)
	for i := range vecs {
		v := make([]float32, 16)
		for j := range v {
			v[j] = float32(r.NormFloat64())
		}
		vecs[i] = v
		cat := "common"
		if i%10 == 0 {
			cat = "rare"
		}
		if _, err := c.Insert(ctx, &vectordbpb.InsertRequest{
			Collection: "docs", Id: uint64(i), Vector: v,
			Metadata: map[string]*vectordbpb.MetadataValue{"category": mv(cat), "n": mvNum(float64(i))},
		}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}

	// Search for a stored vector; it should come back first with its metadata.
	resp, err := c.Search(ctx, &vectordbpb.SearchRequest{
		Collection: "docs", Vector: vecs[42], TopK: 5, EfSearch: 100,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Hits) == 0 || resp.Hits[0].Id != 42 {
		t.Fatalf("expected id 42 first, got %+v", resp.Hits)
	}
	if resp.Hits[0].Metadata["n"].GetNumberValue() != 42 {
		t.Errorf("metadata not round-tripped: %+v", resp.Hits[0].Metadata)
	}
}

func TestEndToEndFilteredSearch(t *testing.T) {
	c := newTestClient(t)
	ctx := context.Background()
	c.CreateCollection(ctx, &vectordbpb.CreateCollectionRequest{
		Name: "docs", Dim: 16, Metric: vectordbpb.DistanceMetric_DISTANCE_METRIC_COSINE, EfSearch: 200,
	})
	r := rand.New(rand.NewSource(2))
	var q []float32
	for i := 0; i < 200; i++ {
		v := make([]float32, 16)
		for j := range v {
			v[j] = float32(r.NormFloat64())
		}
		if i == 0 {
			q = v
		}
		cat := "common"
		if i%10 == 0 {
			cat = "rare"
		}
		c.Insert(ctx, &vectordbpb.InsertRequest{
			Collection: "docs", Id: uint64(i), Vector: v,
			Metadata: map[string]*vectordbpb.MetadataValue{"category": mv(cat)},
		})
	}

	resp, err := c.Search(ctx, &vectordbpb.SearchRequest{
		Collection: "docs", Vector: q, TopK: 5, EfSearch: 200,
		Filter: `category == "rare"`, Strategy: vectordbpb.FilterStrategy_FILTER_STRATEGY_AUTO,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Hits) == 0 {
		t.Fatal("no hits")
	}
	for _, h := range resp.Hits {
		if h.Metadata["category"].GetStringValue() != "rare" {
			t.Errorf("hit %d not rare: %v", h.Id, h.Metadata["category"])
		}
	}
	// A 10% filter should trigger the exact pre-filter under AUTO.
	if resp.StrategyUsed != vectordbpb.FilterStrategy_FILTER_STRATEGY_PRE {
		t.Errorf("expected PRE strategy, got %v", resp.StrategyUsed)
	}
}

func TestErrorCodes(t *testing.T) {
	c := newTestClient(t)
	ctx := context.Background()

	// Searching a missing collection => NotFound.
	_, err := c.Search(ctx, &vectordbpb.SearchRequest{Collection: "nope", Vector: []float32{1, 2}, TopK: 1})
	if status.Code(err) != codes.NotFound {
		t.Errorf("expected NotFound, got %v", err)
	}

	// Bad filter => InvalidArgument.
	c.CreateCollection(ctx, &vectordbpb.CreateCollectionRequest{Name: "c", Dim: 4})
	c.Insert(ctx, &vectordbpb.InsertRequest{Collection: "c", Id: 1, Vector: []float32{1, 2, 3, 4}})
	_, err = c.Search(ctx, &vectordbpb.SearchRequest{Collection: "c", Vector: []float32{1, 2, 3, 4}, TopK: 1, Filter: "bad =="})
	if status.Code(err) != codes.InvalidArgument {
		t.Errorf("expected InvalidArgument for bad filter, got %v", err)
	}

	// Duplicate id => AlreadyExists.
	_, err = c.Insert(ctx, &vectordbpb.InsertRequest{Collection: "c", Id: 1, Vector: []float32{1, 2, 3, 4}})
	if status.Code(err) != codes.AlreadyExists {
		t.Errorf("expected AlreadyExists, got %v", err)
	}
}
