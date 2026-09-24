package storage

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/1morshed1/vectordb-go/internal/collection"
	"github.com/1morshed1/vectordb-go/internal/distance"
)

// snapshotMagic identifies our snapshot files and their version. Bumping the
// trailing digit lets a future reader reject/upgrade older formats.
var snapshotMagic = [4]byte{'V', 'D', 'B', '1'}

// configJSON is the on-disk form of a collection.Config. Metric is stored as its
// human string ("cosine"/"dot"/"euclidean") rather than a raw number so the file
// stays readable and stable if the enum values ever change.
type configJSON struct {
	Name           string `json:"name"`
	Dim            int    `json:"dim"`
	Metric         string `json:"metric"`
	M              int    `json:"m"`
	EfConstruction int    `json:"ef_construction"`
	EfSearch       int    `json:"ef_search"`
}

func toConfigJSON(c collection.Config) configJSON {
	return configJSON{
		Name:           c.Name,
		Dim:            c.Dim,
		Metric:         c.Metric.String(),
		M:              c.M,
		EfConstruction: c.EfConstruction,
		EfSearch:       c.EfSearch,
	}
}

func (cj configJSON) toConfig() (collection.Config, error) {
	m, err := distance.ParseMetric(cj.Metric)
	if err != nil {
		return collection.Config{}, err
	}
	return collection.Config{
		Name:           cj.Name,
		Dim:            cj.Dim,
		Metric:         m,
		M:              cj.M,
		EfConstruction: cj.EfConstruction,
		EfSearch:       cj.EfSearch,
	}, nil
}

// WriteSnapshot writes the full state of a collection to path atomically.
//
// Atomicity matters: if we crashed halfway through overwriting the snapshot, we
// would lose *both* the old and new state. So we write to a temp file in the
// same directory, fsync it, then rename it over the target. rename(2) is atomic
// on POSIX filesystems, so a reader always sees either the complete old file or
// the complete new one.
//
// Layout:
//
//	magic     : 4 bytes
//	configLen : 4 bytes   + config JSON
//	itemCount : 4 bytes
//	items     : itemCount × ([4-byte len][record payload])   (Op = OpInsert)
func WriteSnapshot(path string, cfg collection.Config, items []collection.Item) (err error) {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	// Ensure we clean up the temp file on any error path.
	defer func() {
		if err != nil {
			f.Close()
			os.Remove(tmp)
		}
	}()

	w := bufio.NewWriter(f)
	if _, err = w.Write(snapshotMagic[:]); err != nil {
		return err
	}

	cfgBytes, err := json.Marshal(toConfigJSON(cfg))
	if err != nil {
		return err
	}
	if err = writeUint32(w, uint32(len(cfgBytes))); err != nil {
		return err
	}
	if _, err = w.Write(cfgBytes); err != nil {
		return err
	}

	if err = writeUint32(w, uint32(len(items))); err != nil {
		return err
	}
	for _, it := range items {
		payload, encErr := Record{Op: OpInsert, ID: it.ID, Vector: it.Vector, Metadata: it.Metadata}.encode()
		if encErr != nil {
			return encErr
		}
		if err = writeUint32(w, uint32(len(payload))); err != nil {
			return err
		}
		if _, err = w.Write(payload); err != nil {
			return err
		}
	}

	if err = w.Flush(); err != nil {
		return err
	}
	if err = f.Sync(); err != nil {
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	// Atomic replace.
	if err = os.Rename(tmp, path); err != nil {
		return err
	}
	// fsync the directory so the rename itself is durable.
	return fsyncDir(filepath.Dir(path))
}

// ReadSnapshot loads a snapshot written by WriteSnapshot. A missing file yields
// a zero config and nil items with no error, so callers can treat "no snapshot
// yet" as "empty".
func ReadSnapshot(path string) (collection.Config, []collection.Item, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return collection.Config{}, nil, nil
		}
		return collection.Config{}, nil, err
	}
	defer f.Close()
	r := bufio.NewReader(f)

	var magic [4]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return collection.Config{}, nil, fmt.Errorf("storage: read snapshot magic: %w", err)
	}
	if magic != snapshotMagic {
		return collection.Config{}, nil, fmt.Errorf("storage: bad snapshot magic %v", magic)
	}

	cfgLen, err := readUint32(r)
	if err != nil {
		return collection.Config{}, nil, err
	}
	cfgBytes := make([]byte, cfgLen)
	if _, err := io.ReadFull(r, cfgBytes); err != nil {
		return collection.Config{}, nil, err
	}
	var cj configJSON
	if err := json.Unmarshal(cfgBytes, &cj); err != nil {
		return collection.Config{}, nil, err
	}
	cfg, err := cj.toConfig()
	if err != nil {
		return collection.Config{}, nil, err
	}

	count, err := readUint32(r)
	if err != nil {
		return collection.Config{}, nil, err
	}
	items := make([]collection.Item, 0, count)
	for i := uint32(0); i < count; i++ {
		plen, err := readUint32(r)
		if err != nil {
			return collection.Config{}, nil, err
		}
		payload := make([]byte, plen)
		if _, err := io.ReadFull(r, payload); err != nil {
			return collection.Config{}, nil, err
		}
		rec, err := decodeRecord(payload)
		if err != nil {
			return collection.Config{}, nil, err
		}
		items = append(items, collection.Item{ID: rec.ID, Vector: rec.Vector, Metadata: rec.Metadata})
	}
	return cfg, items, nil
}

func writeUint32(w io.Writer, v uint32) error {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	_, err := w.Write(b[:])
	return err
}

func readUint32(r io.Reader) (uint32, error) {
	var b [4]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b[:]), nil
}

// fsyncDir flushes a directory entry so a create/rename within it is durable.
func fsyncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}
