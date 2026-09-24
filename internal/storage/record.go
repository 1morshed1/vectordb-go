// Package storage gives collections durability: their data survives a process
// crash or restart. It has two cooperating pieces, the classic database pair:
//
//   - A write-ahead log (WAL): every mutation is appended to an on-disk log
//     *before* it is applied in memory, so a crash can be recovered by replaying
//     the log. See wal.go and docs/learning/06.
//   - Snapshots: the full state of a collection serialized to one file, so the
//     WAL can be truncated instead of growing forever. See snapshot.go.
//
// This file defines the record type both pieces serialize, and the binary
// encoding they share.
package storage

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"

	"github.com/1morshed1/vectordb-go/internal/collection"
)

// OpType tags a mutation record.
type OpType uint8

const (
	// OpInsert adds a vector (+ metadata) under an id.
	OpInsert OpType = 1
	// OpDelete removes an id.
	OpDelete OpType = 2
)

// Record is one durable mutation. For OpDelete, Vector and Metadata are empty.
type Record struct {
	Op       OpType
	ID       uint64
	Vector   []float32
	Metadata collection.Metadata
}

// encode serializes a Record into a compact binary payload:
//
//	op        : 1 byte
//	id        : 8 bytes  (uint64, little-endian)
//	vecLen    : 4 bytes  (uint32)  number of float32s
//	vector    : vecLen * 4 bytes   (each float32 as its IEEE-754 bits)
//	metaLen   : 4 bytes  (uint32)  length of the JSON metadata blob
//	metadata  : metaLen bytes      (JSON; naturally handles string/number/bool)
//
// We hand-roll the vector as raw float32 bits (not JSON) because vectors are the
// bulk of the data and JSON would bloat them ~5x and cost parse time. Metadata
// is small and irregular, so JSON's flexibility is worth its overhead there.
func (r Record) encode() ([]byte, error) {
	metaJSON := []byte("null")
	if r.Metadata != nil {
		var err error
		metaJSON, err = json.Marshal(r.Metadata)
		if err != nil {
			return nil, fmt.Errorf("storage: marshal metadata: %w", err)
		}
	}

	size := 1 + 8 + 4 + len(r.Vector)*4 + 4 + len(metaJSON)
	buf := make([]byte, size)
	pos := 0
	buf[pos] = byte(r.Op)
	pos++
	binary.LittleEndian.PutUint64(buf[pos:], r.ID)
	pos += 8
	binary.LittleEndian.PutUint32(buf[pos:], uint32(len(r.Vector)))
	pos += 4
	for _, f := range r.Vector {
		binary.LittleEndian.PutUint32(buf[pos:], math.Float32bits(f))
		pos += 4
	}
	binary.LittleEndian.PutUint32(buf[pos:], uint32(len(metaJSON)))
	pos += 4
	copy(buf[pos:], metaJSON)
	return buf, nil
}

// decodeRecord parses a payload produced by encode. It returns an error if the
// buffer is too short at any field, which the WAL replay uses to detect a record
// torn by a mid-write crash.
func decodeRecord(buf []byte) (Record, error) {
	var r Record
	pos := 0
	need := func(n int) error {
		if pos+n > len(buf) {
			return fmt.Errorf("storage: truncated record (need %d at %d, have %d)", n, pos, len(buf))
		}
		return nil
	}

	if err := need(1); err != nil {
		return r, err
	}
	r.Op = OpType(buf[pos])
	pos++

	if err := need(8); err != nil {
		return r, err
	}
	r.ID = binary.LittleEndian.Uint64(buf[pos:])
	pos += 8

	if err := need(4); err != nil {
		return r, err
	}
	vecLen := int(binary.LittleEndian.Uint32(buf[pos:]))
	pos += 4
	if err := need(vecLen * 4); err != nil {
		return r, err
	}
	if vecLen > 0 {
		r.Vector = make([]float32, vecLen)
		for i := 0; i < vecLen; i++ {
			r.Vector[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[pos:]))
			pos += 4
		}
	}

	if err := need(4); err != nil {
		return r, err
	}
	metaLen := int(binary.LittleEndian.Uint32(buf[pos:]))
	pos += 4
	if err := need(metaLen); err != nil {
		return r, err
	}
	metaJSON := buf[pos : pos+metaLen]
	pos += metaLen
	if string(metaJSON) != "null" {
		if err := json.Unmarshal(metaJSON, &r.Metadata); err != nil {
			return r, fmt.Errorf("storage: unmarshal metadata: %w", err)
		}
		r.Metadata = normalizeMetadata(r.Metadata)
	}
	return r, nil
}

// normalizeMetadata ensures decoded numbers are float64 (JSON already does this)
// and is the single place to enforce the collection's allowed value types after
// a round-trip through JSON. It drops any value that isn't string/float64/bool.
func normalizeMetadata(md collection.Metadata) collection.Metadata {
	for k, v := range md {
		switch v.(type) {
		case string, float64, bool:
			// ok
		default:
			delete(md, k)
		}
	}
	return md
}
