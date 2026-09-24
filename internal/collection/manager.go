package collection

import (
	"errors"
	"fmt"
	"sort"
	"sync"
)

// Manager owns the set of named collections in a server. It is the top-level
// object the gRPC layer talks to: create/drop/list collections, and route
// insert/search/delete to the right one. All access is guarded so collections
// can be created and used concurrently.
type Manager struct {
	mu          sync.RWMutex
	collections map[string]*Collection
}

var (
	// ErrCollectionExists is returned by Create when the name is taken.
	ErrCollectionExists = errors.New("collection: already exists")
	// ErrCollectionNotFound is returned when a named collection is absent.
	ErrCollectionNotFound = errors.New("collection: not found")
)

// NewManager returns an empty manager.
func NewManager() *Manager {
	return &Manager{collections: make(map[string]*Collection)}
}

// Create builds and registers a new collection. Fails if the name is taken.
func (m *Manager) Create(cfg Config) (*Collection, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.collections[cfg.Name]; exists {
		return nil, fmt.Errorf("%w: %q", ErrCollectionExists, cfg.Name)
	}
	col, err := New(cfg)
	if err != nil {
		return nil, err
	}
	m.collections[cfg.Name] = col
	return col, nil
}

// Get returns a collection by name, or ErrCollectionNotFound.
func (m *Manager) Get(name string) (*Collection, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	col, ok := m.collections[name]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrCollectionNotFound, name)
	}
	return col, nil
}

// Drop removes a collection. Returns ErrCollectionNotFound if absent.
func (m *Manager) Drop(name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.collections[name]; !ok {
		return fmt.Errorf("%w: %q", ErrCollectionNotFound, name)
	}
	delete(m.collections, name)
	return nil
}

// List returns the names of all collections, sorted for stable output.
func (m *Manager) List() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	names := make([]string, 0, len(m.collections))
	for name := range m.collections {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Register inserts an already-built collection under its configured name. Used
// by the storage layer when loading collections from disk at startup.
func (m *Manager) Register(col *Collection) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	name := col.cfg.Name
	if _, exists := m.collections[name]; exists {
		return fmt.Errorf("%w: %q", ErrCollectionExists, name)
	}
	m.collections[name] = col
	return nil
}
