package storage

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/1morshed1/vectordb-go/internal/collection"
)

// Store adds durability to a collection.Manager. It owns a data directory with
// one subdirectory per collection:
//
//	<dataDir>/<collection>/snapshot.dat   full state as of the last snapshot
//	<dataDir>/<collection>/wal.log        mutations since that snapshot
//
// Every mutating operation goes through the Store so it can be logged to the WAL
// *before* being applied in memory (write-ahead). Reads go straight to the
// Manager, which the Store exposes.
//
// Recovery on Open is: for each collection, load its snapshot, then replay its
// WAL on top. Snapshot + WAL together reconstruct the exact acknowledged state.
type Store struct {
	dataDir string
	mgr     *collection.Manager

	mu   sync.Mutex
	wals map[string]*WAL // collection name -> its WAL
}

const (
	snapshotFile = "snapshot.dat"
	walFile      = "wal.log"
)

// Open opens (creating if needed) a Store rooted at dataDir, recovering any
// collections found there.
func Open(dataDir string) (*Store, error) {
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return nil, err
	}
	s := &Store{
		dataDir: dataDir,
		mgr:     collection.NewManager(),
		wals:    make(map[string]*WAL),
	}
	if err := s.recover(); err != nil {
		return nil, err
	}
	return s, nil
}

// Manager exposes the underlying manager for read operations (Get/List/Search).
func (s *Store) Manager() *collection.Manager { return s.mgr }

// recover loads every collection subdirectory: snapshot first, then WAL replay.
func (s *Store) recover() error {
	entries, err := os.ReadDir(s.dataDir)
	if err != nil {
		return err
	}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		name := e.Name()
		dir := filepath.Join(s.dataDir, name)

		cfg, items, err := ReadSnapshot(filepath.Join(dir, snapshotFile))
		if err != nil {
			return fmt.Errorf("storage: recover %s snapshot: %w", name, err)
		}
		if cfg.Name == "" {
			// No usable snapshot (empty dir); skip.
			continue
		}

		col, err := collection.New(cfg)
		if err != nil {
			return fmt.Errorf("storage: recover %s new: %w", name, err)
		}
		// Rebuild the index from the snapshot's vectors. We rebuild rather than
		// persisting the graph edges — see docs/learning/06 for why.
		for _, it := range items {
			if err := col.Insert(it.ID, it.Vector, it.Metadata); err != nil {
				return fmt.Errorf("storage: recover %s insert %d: %w", name, it.ID, err)
			}
		}
		// Replay the WAL on top of the snapshot to apply mutations that happened
		// after it was taken.
		err = ReplayWAL(filepath.Join(dir, walFile), func(r Record) error {
			switch r.Op {
			case OpInsert:
				// A re-inserted id (snapshot + later delete + re-insert) must not
				// collide; deletes in the WAL are applied in order, so by the time
				// we see this insert any prior delete has already run.
				if err := col.Insert(r.ID, r.Vector, r.Metadata); err != nil && !errors.Is(err, collection.ErrDuplicateID) {
					return err
				}
				return nil
			case OpDelete:
				if err := col.Delete(r.ID); err != nil && !errors.Is(err, collection.ErrNotFound) {
					return err
				}
				return nil
			default:
				return fmt.Errorf("storage: unknown wal op %d", r.Op)
			}
		})
		if err != nil {
			return fmt.Errorf("storage: recover %s wal: %w", name, err)
		}

		if err := s.mgr.Register(col); err != nil {
			return err
		}
		wal, err := OpenWAL(filepath.Join(dir, walFile))
		if err != nil {
			return err
		}
		s.wals[name] = wal
	}
	return nil
}

// CreateCollection creates a collection, its directory, an initial snapshot
// (which records the config), and its WAL.
func (s *Store) CreateCollection(cfg collection.Config) (*collection.Collection, error) {
	col, err := s.mgr.Create(cfg)
	if err != nil {
		return nil, err
	}
	dir := filepath.Join(s.dataDir, cfg.Name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	// Write an initial snapshot so the config is durable even before any inserts.
	if err := WriteSnapshot(filepath.Join(dir, snapshotFile), cfg, nil); err != nil {
		return nil, err
	}
	wal, err := OpenWAL(filepath.Join(dir, walFile))
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	s.wals[cfg.Name] = wal
	s.mu.Unlock()
	return col, nil
}

// Insert logs then applies an insert. The WAL append is durable before the
// in-memory mutation, so a crash can never lose an acknowledged insert.
func (s *Store) Insert(coll string, id uint64, vector []float32, md collection.Metadata) error {
	col, err := s.mgr.Get(coll)
	if err != nil {
		return err
	}
	wal, err := s.walFor(coll)
	if err != nil {
		return err
	}
	// Pre-validate against the in-memory state so we don't append a record that
	// would fail to apply (e.g. a duplicate id). This keeps the WAL replayable:
	// every record in it is one that did apply.
	if _, err := col.GetMetadata(id); err == nil {
		return collection.ErrDuplicateID
	}
	if err := wal.Append(Record{Op: OpInsert, ID: id, Vector: vector, Metadata: md}); err != nil {
		return err
	}
	return col.Insert(id, vector, md)
}

// Delete logs then applies a delete.
func (s *Store) Delete(coll string, id uint64) error {
	col, err := s.mgr.Get(coll)
	if err != nil {
		return err
	}
	wal, err := s.walFor(coll)
	if err != nil {
		return err
	}
	if _, err := col.GetMetadata(id); err != nil {
		return collection.ErrNotFound
	}
	if err := wal.Append(Record{Op: OpDelete, ID: id}); err != nil {
		return err
	}
	return col.Delete(id)
}

// Snapshot compacts a collection: it writes the collection's full current state
// to a fresh snapshot, then truncates the WAL. After this the WAL is empty and
// the snapshot alone reflects all data, which bounds recovery time and disk use.
func (s *Store) Snapshot(coll string) error {
	col, err := s.mgr.Get(coll)
	if err != nil {
		return err
	}
	wal, err := s.walFor(coll)
	if err != nil {
		return err
	}
	dir := filepath.Join(s.dataDir, coll)
	// Order is critical for crash-safety: write the new snapshot durably FIRST,
	// only then truncate the WAL. If we crash between the two, recovery loads the
	// new snapshot and replays a WAL that still holds those mutations — they are
	// applied twice, which is safe because insert/delete replay is idempotent
	// (duplicate-insert and missing-delete errors are ignored on replay). If we
	// truncated first and crashed before writing the snapshot, we would lose
	// data — so never do it in that order.
	if err := WriteSnapshot(filepath.Join(dir, snapshotFile), col.Config(), col.Items()); err != nil {
		return err
	}
	return wal.Truncate()
}

// DropCollection removes a collection from memory and deletes its files.
func (s *Store) DropCollection(coll string) error {
	if err := s.mgr.Drop(coll); err != nil {
		return err
	}
	s.mu.Lock()
	if wal, ok := s.wals[coll]; ok {
		wal.Close()
		delete(s.wals, coll)
	}
	s.mu.Unlock()
	return os.RemoveAll(filepath.Join(s.dataDir, coll))
}

// Close flushes and closes all WAL files.
func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	var firstErr error
	for _, wal := range s.wals {
		if err := wal.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

func (s *Store) walFor(coll string) (*WAL, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	wal, ok := s.wals[coll]
	if !ok {
		return nil, fmt.Errorf("storage: no wal for collection %q", coll)
	}
	return wal, nil
}
