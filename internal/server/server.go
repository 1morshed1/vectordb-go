// Package server implements the gRPC VectorDB service by translating wire
// requests into calls on the storage.Store (which owns durability and the
// collection manager). It is a thin adapter: no search logic lives here, only
// request/response conversion, validation, and error-code mapping.
//
// See docs/learning/07-grpc-api.md.
package server

import (
	"context"
	"errors"

	vectordbpb "github.com/1morshed1/vectordb-go/api/gen"
	"github.com/1morshed1/vectordb-go/internal/collection"
	"github.com/1morshed1/vectordb-go/internal/distance"
	"github.com/1morshed1/vectordb-go/internal/filter"
	"github.com/1morshed1/vectordb-go/internal/storage"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Service implements vectordbpb.VectorDBServer.
type Service struct {
	vectordbpb.UnimplementedVectorDBServer // forward-compat: new RPCs won't break old servers
	store                                  *storage.Store
}

// New wraps a Store in a gRPC service.
func New(store *storage.Store) *Service { return &Service{store: store} }

// --- collection management --------------------------------------------------

func (s *Service) CreateCollection(ctx context.Context, req *vectordbpb.CreateCollectionRequest) (*vectordbpb.CreateCollectionResponse, error) {
	if req.Name == "" {
		return nil, status.Error(codes.InvalidArgument, "name is required")
	}
	if req.Dim <= 0 {
		return nil, status.Error(codes.InvalidArgument, "dim must be positive")
	}
	cfg := collection.Config{
		Name:           req.Name,
		Dim:            int(req.Dim),
		Metric:         metricFromProto(req.Metric),
		M:              int(req.M),
		EfConstruction: int(req.EfConstruction),
		EfSearch:       int(req.EfSearch),
	}
	if _, err := s.store.CreateCollection(cfg); err != nil {
		return nil, toStatus(err)
	}
	return &vectordbpb.CreateCollectionResponse{Name: req.Name}, nil
}

func (s *Service) DropCollection(ctx context.Context, req *vectordbpb.DropCollectionRequest) (*vectordbpb.DropCollectionResponse, error) {
	if err := s.store.DropCollection(req.Name); err != nil {
		return nil, toStatus(err)
	}
	return &vectordbpb.DropCollectionResponse{}, nil
}

func (s *Service) ListCollections(ctx context.Context, req *vectordbpb.ListCollectionsRequest) (*vectordbpb.ListCollectionsResponse, error) {
	return &vectordbpb.ListCollectionsResponse{Names: s.store.Manager().List()}, nil
}

// --- data operations --------------------------------------------------------

func (s *Service) Insert(ctx context.Context, req *vectordbpb.InsertRequest) (*vectordbpb.InsertResponse, error) {
	if len(req.Vector) == 0 {
		return nil, status.Error(codes.InvalidArgument, "vector is required")
	}
	md := metadataFromProto(req.Metadata)
	if err := s.store.Insert(req.Collection, req.Id, req.Vector, md); err != nil {
		return nil, toStatus(err)
	}
	return &vectordbpb.InsertResponse{}, nil
}

func (s *Service) Delete(ctx context.Context, req *vectordbpb.DeleteRequest) (*vectordbpb.DeleteResponse, error) {
	if err := s.store.Delete(req.Collection, req.Id); err != nil {
		return nil, toStatus(err)
	}
	return &vectordbpb.DeleteResponse{}, nil
}

func (s *Service) Search(ctx context.Context, req *vectordbpb.SearchRequest) (*vectordbpb.SearchResponse, error) {
	if len(req.Vector) == 0 {
		return nil, status.Error(codes.InvalidArgument, "query vector is required")
	}
	topK := int(req.TopK)
	if topK <= 0 {
		topK = 10
	}
	col, err := s.store.Manager().Get(req.Collection)
	if err != nil {
		return nil, toStatus(err)
	}

	f, err := filter.Parse(req.Filter)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid filter: %v", err)
	}

	res, err := col.SearchFiltered(req.Vector, topK, int(req.EfSearch), f, strategyFromProto(req.Strategy))
	if err != nil {
		return nil, toStatus(err)
	}

	hits := make([]*vectordbpb.SearchHit, len(res.Results))
	for i, r := range res.Results {
		hits[i] = &vectordbpb.SearchHit{
			Id:       r.ID,
			Distance: r.Distance,
			Metadata: metadataToProto(r.Metadata),
		}
	}
	return &vectordbpb.SearchResponse{
		Hits:         hits,
		StrategyUsed: strategyToProto(res.Strategy),
		Matched:      int32(res.Matched),
	}, nil
}

func (s *Service) Snapshot(ctx context.Context, req *vectordbpb.SnapshotRequest) (*vectordbpb.SnapshotResponse, error) {
	if err := s.store.Snapshot(req.Collection); err != nil {
		return nil, toStatus(err)
	}
	return &vectordbpb.SnapshotResponse{}, nil
}

// --- conversions ------------------------------------------------------------

func metricFromProto(m vectordbpb.DistanceMetric) distance.Metric {
	switch m {
	case vectordbpb.DistanceMetric_DISTANCE_METRIC_DOT:
		return distance.DotProduct
	case vectordbpb.DistanceMetric_DISTANCE_METRIC_EUCLIDEAN:
		return distance.Euclidean
	default: // UNSPECIFIED or COSINE
		return distance.Cosine
	}
}

func strategyFromProto(s vectordbpb.FilterStrategy) collection.FilterStrategy {
	switch s {
	case vectordbpb.FilterStrategy_FILTER_STRATEGY_PRE:
		return collection.StrategyPre
	case vectordbpb.FilterStrategy_FILTER_STRATEGY_POST:
		return collection.StrategyPost
	default:
		return collection.StrategyAuto
	}
}

func strategyToProto(s collection.FilterStrategy) vectordbpb.FilterStrategy {
	switch s {
	case collection.StrategyPre:
		return vectordbpb.FilterStrategy_FILTER_STRATEGY_PRE
	case collection.StrategyPost:
		return vectordbpb.FilterStrategy_FILTER_STRATEGY_POST
	default:
		return vectordbpb.FilterStrategy_FILTER_STRATEGY_AUTO
	}
}

// metadataFromProto converts the wire oneof map into our Metadata map. Numbers
// arrive as double and are stored as float64, matching the filter engine.
func metadataFromProto(m map[string]*vectordbpb.MetadataValue) collection.Metadata {
	if len(m) == 0 {
		return nil
	}
	out := make(collection.Metadata, len(m))
	for k, v := range m {
		switch val := v.GetValue().(type) {
		case *vectordbpb.MetadataValue_StringValue:
			out[k] = val.StringValue
		case *vectordbpb.MetadataValue_NumberValue:
			out[k] = val.NumberValue
		case *vectordbpb.MetadataValue_BoolValue:
			out[k] = val.BoolValue
		}
	}
	return out
}

func metadataToProto(md collection.Metadata) map[string]*vectordbpb.MetadataValue {
	if len(md) == 0 {
		return nil
	}
	out := make(map[string]*vectordbpb.MetadataValue, len(md))
	for k, v := range md {
		switch val := v.(type) {
		case string:
			out[k] = &vectordbpb.MetadataValue{Value: &vectordbpb.MetadataValue_StringValue{StringValue: val}}
		case float64:
			out[k] = &vectordbpb.MetadataValue{Value: &vectordbpb.MetadataValue_NumberValue{NumberValue: val}}
		case bool:
			out[k] = &vectordbpb.MetadataValue{Value: &vectordbpb.MetadataValue_BoolValue{BoolValue: val}}
		}
	}
	return out
}

// toStatus maps our domain errors onto gRPC status codes so clients get
// meaningful, machine-readable failures instead of opaque strings.
func toStatus(err error) error {
	switch {
	case err == nil:
		return nil
	case errors.Is(err, collection.ErrCollectionNotFound), errors.Is(err, collection.ErrNotFound):
		return status.Error(codes.NotFound, err.Error())
	case errors.Is(err, collection.ErrCollectionExists), errors.Is(err, collection.ErrDuplicateID):
		return status.Error(codes.AlreadyExists, err.Error())
	case errors.Is(err, collection.ErrDimMismatch):
		return status.Error(codes.InvalidArgument, err.Error())
	default:
		return status.Error(codes.Internal, err.Error())
	}
}
