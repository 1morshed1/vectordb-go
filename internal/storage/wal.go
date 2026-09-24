package storage

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"sync"
)

// WAL is an append-only write-ahead log for one collection.
//
// The contract that makes it useful: a mutation is durably on disk (fsync'd)
// *before* we tell the caller it succeeded and *before* it is applied in memory.
// So if the process dies at any instant, replaying the log reconstructs exactly
// the acknowledged mutations — no more (torn writes are discarded), no fewer.
//
// On-disk framing, one record after another:
//
//	[uint32 payloadLen][uint32 crc32(payload)][payload bytes...]
//
// The length prefix lets us read one record at a time and detect a *torn* final
// record (a crash mid-append leaves a short or missing tail). The CRC detects
// silent corruption of a record's bytes. On replay we stop at the first record
// that is short or fails its CRC, and truncate the file there — everything
// before it was acknowledged and is intact.
type WAL struct {
	mu   sync.Mutex
	f    *os.File
	path string
}

const walCRCTable = crc32.Castagnoli // stronger than IEEE, cheap on modern CPUs

var castagnoli = crc32.MakeTable(walCRCTable)

// OpenWAL opens (creating if needed) the WAL at path for appending.
func OpenWAL(path string) (*WAL, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0o644)
	if err != nil {
		return nil, fmt.Errorf("storage: open wal %s: %w", path, err)
	}
	return &WAL{f: f, path: path}, nil
}

// Append writes one record durably. It encodes, frames, writes, and fsyncs
// before returning, so a successful return means the record survives a crash.
func (w *WAL) Append(r Record) error {
	payload, err := r.encode()
	if err != nil {
		return err
	}
	w.mu.Lock()
	defer w.mu.Unlock()

	var header [8]byte
	binary.LittleEndian.PutUint32(header[0:], uint32(len(payload)))
	binary.LittleEndian.PutUint32(header[4:], crc32.Checksum(payload, castagnoli))

	if _, err := w.f.Write(header[:]); err != nil {
		return fmt.Errorf("storage: wal write header: %w", err)
	}
	if _, err := w.f.Write(payload); err != nil {
		return fmt.Errorf("storage: wal write payload: %w", err)
	}
	// fsync: the whole point of a WAL. Without this, the OS may buffer the write
	// and a power loss would lose an "acknowledged" mutation.
	if err := w.f.Sync(); err != nil {
		return fmt.Errorf("storage: wal fsync: %w", err)
	}
	return nil
}

// Close closes the underlying file.
func (w *WAL) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.f.Close()
}

// Truncate empties the log. Called after a snapshot captures all prior state, so
// the WAL only ever holds mutations since the last snapshot.
func (w *WAL) Truncate() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if err := w.f.Truncate(0); err != nil {
		return err
	}
	_, err := w.f.Seek(0, io.SeekStart)
	return err
}

// ReplayWAL reads every intact record from the WAL at path, in order, calling fn
// for each. It tolerates a torn or corrupt trailing record (the signature of a
// crash mid-append) by stopping there and truncating the file to the last good
// record. A missing file replays as empty.
func ReplayWAL(path string, fn func(Record) error) error {
	f, err := os.OpenFile(path, os.O_RDWR, 0o644)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer f.Close()

	r := bufio.NewReader(f)
	var goodOffset int64 // bytes of intact records seen so far

	for {
		var header [8]byte
		n, err := io.ReadFull(r, header[:])
		if err == io.EOF {
			break // clean end
		}
		if err == io.ErrUnexpectedEOF || (err == nil && n < 8) {
			// Torn header: crash happened between records. Stop and truncate.
			break
		}
		if err != nil {
			return err
		}

		payloadLen := binary.LittleEndian.Uint32(header[0:])
		wantCRC := binary.LittleEndian.Uint32(header[4:])

		payload := make([]byte, payloadLen)
		if _, err := io.ReadFull(r, payload); err != nil {
			// Torn payload: the record wasn't fully written. Stop.
			break
		}
		if crc32.Checksum(payload, castagnoli) != wantCRC {
			// Corrupt record. Stop here; earlier records are trustworthy.
			break
		}

		rec, err := decodeRecord(payload)
		if err != nil {
			break
		}
		if err := fn(rec); err != nil {
			return err
		}
		goodOffset += int64(8 + payloadLen)
	}

	// Discard any torn/corrupt tail so future appends start from clean state.
	if err := f.Truncate(goodOffset); err != nil {
		return err
	}
	return nil
}
