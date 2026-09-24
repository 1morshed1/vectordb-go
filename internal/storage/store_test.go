package storage

import (
	"math/rand"
	"os"
	"testing"

	"github.com/1morshed1/vectordb-go/internal/collection"
	"github.com/1morshed1/vectordb-go/internal/distance"
)

func randVec(r *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = float32(r.NormFloat64())
	}
	return v
}

func TestWALRecordRoundTrip(t *testing.T) {
	rec := Record{
		Op:       OpInsert,
		ID:       123,
		Vector:   []float32{1.5, -2.25, 0.0, 3.75},
		Metadata: collection.Metadata{"category": "finance", "year": float64(2024), "public": true},
	}
	payload, err := rec.encode()
	if err != nil {
		t.Fatal(err)
	}
	got, err := decodeRecord(payload)
	if err != nil {
		t.Fatal(err)
	}
	if got.ID != rec.ID || got.Op != rec.Op {
		t.Fatalf("header mismatch: %+v", got)
	}
	for i := range rec.Vector {
		if got.Vector[i] != rec.Vector[i] {
			t.Fatalf("vector mismatch at %d: %v vs %v", i, got.Vector[i], rec.Vector[i])
		}
	}
	if got.Metadata["category"] != "finance" || got.Metadata["year"].(float64) != 2024 || got.Metadata["public"] != true {
		t.Fatalf("metadata mismatch: %+v", got.Metadata)
	}
}

// TestRecoverFromWAL is the core durability test: write data through a Store,
// "crash" (drop the Store without snapshotting), reopen, and verify every insert
// and delete survived via WAL replay.
func TestRecoverFromWAL(t *testing.T) {
	dir := t.TempDir()
	r := rand.New(rand.NewSource(1))

	s, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	cfg := collection.Config{Name: "docs", Dim: 16, Metric: distance.Cosine, EfSearch: 100}
	if _, err := s.CreateCollection(cfg); err != nil {
		t.Fatal(err)
	}
	vecs := make([][]float32, 200)
	for i := range vecs {
		vecs[i] = randVec(r, 16)
		if err := s.Insert("docs", uint64(i), vecs[i], collection.Metadata{"n": float64(i)}); err != nil {
			t.Fatal(err)
		}
	}
	// Delete a few.
	for _, id := range []uint64{5, 10, 15} {
		if err := s.Delete("docs", id); err != nil {
			t.Fatal(err)
		}
	}
	// Simulate a crash: close WALs but never snapshot.
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}

	// Reopen and verify recovery.
	s2, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer s2.Close()
	col, err := s2.Manager().Get("docs")
	if err != nil {
		t.Fatal(err)
	}
	if col.Len() != 197 { // 200 - 3 deleted
		t.Fatalf("recovered Len = %d, want 197", col.Len())
	}
	// A surviving vector must still be findable.
	res, _ := col.Search(vecs[42], 1, 100)
	if len(res) == 0 || res[0].ID != 42 {
		t.Fatalf("expected id 42 after recovery, got %+v", res)
	}
	// A deleted vector must be gone.
	if _, err := col.GetMetadata(5); err != collection.ErrNotFound {
		t.Errorf("deleted id 5 came back after recovery")
	}
}

// TestRecoverFromSnapshotPlusWAL checks the compaction path: snapshot, then more
// mutations into a fresh WAL, then recover from snapshot + WAL together.
func TestRecoverFromSnapshotPlusWAL(t *testing.T) {
	dir := t.TempDir()
	r := rand.New(rand.NewSource(2))

	s, _ := Open(dir)
	cfg := collection.Config{Name: "c", Dim: 8, Metric: distance.Euclidean}
	s.CreateCollection(cfg)
	for i := 0; i < 100; i++ {
		s.Insert("c", uint64(i), randVec(r, 8), nil)
	}
	// Compact: snapshot then truncate WAL.
	if err := s.Snapshot("c"); err != nil {
		t.Fatal(err)
	}
	// More mutations after the snapshot go into the now-empty WAL.
	for i := 100; i < 150; i++ {
		s.Insert("c", uint64(i), randVec(r, 8), nil)
	}
	s.Delete("c", 0)
	s.Close()

	s2, _ := Open(dir)
	defer s2.Close()
	col, _ := s2.Manager().Get("c")
	if col.Len() != 149 { // 150 inserted - 1 deleted
		t.Fatalf("recovered Len = %d, want 149", col.Len())
	}
	if _, err := col.GetMetadata(0); err != collection.ErrNotFound {
		t.Error("deleted id 0 survived recovery")
	}
	if _, err := col.GetMetadata(149); err != nil {
		t.Error("post-snapshot insert 149 lost")
	}
}

func TestTornWALTailIsDiscarded(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/wal.log"
	wal, _ := OpenWAL(path)
	_ = wal.Append(Record{Op: OpInsert, ID: 1, Vector: []float32{1, 2}})
	_ = wal.Append(Record{Op: OpInsert, ID: 2, Vector: []float32{3, 4}})
	wal.Close()

	// Corrupt the file by appending a garbage partial record (simulating a crash
	// mid-append).
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	f.Write([]byte{0xFF, 0xFF, 0xFF}) // 3 bytes: a torn header
	f.Close()

	var ids []uint64
	if err := ReplayWAL(path, func(r Record) error {
		ids = append(ids, r.ID)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(ids) != 2 || ids[0] != 1 || ids[1] != 2 {
		t.Fatalf("expected the two intact records, got %v", ids)
	}
}
