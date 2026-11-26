# Learning Guide 06 — Persistence & the Write-Ahead Log

**Package:** `internal/storage`

Everything so far lives in RAM. Restart the process and it's gone. A database
must **survive a crash** and come back with exactly the data it acknowledged.
This package adds that with the two mechanisms every durable store uses: a
**write-ahead log** and **snapshots**.

## 1. The durability contract

When a client's `Insert` returns success, that insert must survive a power loss
one nanosecond later. To promise that, we must get the mutation onto stable
storage *before* we return — and *before* we even apply it in memory. That
ordering is the whole idea, and it's the "write-ahead" in write-ahead log:

```
1. append the mutation to the WAL   (and fsync it — now it's on disk)
2. apply the mutation in memory
3. return success to the client
```

If we crash after step 1, replay finds the record and reapplies it. If we crash
before step 1, the client never got success, so losing it is correct. There is
no ordering that loses an acknowledged write. `Store.Insert`/`Store.Delete`
implement exactly this order.

## 2. Why `fsync` is the point

`os.File.Write` only hands bytes to the OS page cache; the OS may not put them on
the physical disk for seconds. A power loss in that window loses a write we
"succeeded." `WAL.Append` calls `f.Sync()` (fsync) after every record, forcing
the bytes to durable storage before returning. Without the fsync, the WAL is
theatre. (Real systems batch many writes into one fsync — "group commit" — to
amortize its cost; we fsync per record for clarity.)

## 3. The on-disk framing (`wal.go`)

Each record is written as:

```
[uint32 payloadLen][uint32 crc32c(payload)][payload bytes]
```

Two guards, for two different failure modes:

- **The length prefix** lets us read one record at a time and recognize a
  **torn record** — a crash mid-append leaves a short or missing tail. On replay
  we read until a header or payload comes up short, then **stop and truncate** the
  file to the last complete record. `TestTornWALTailIsDiscarded` appends 3 junk
  bytes (a half-written header) and confirms replay keeps the two good records
  and drops the garbage.
- **The CRC** (Castagnoli/CRC32C, hardware-accelerated) detects **silent
  corruption** — a flipped bit in a record's bytes. A record that fails its CRC
  is treated as the end of trustworthy data.

## 4. The record encoding (`record.go`)

```
op(1) id(8) vecLen(4) vector(vecLen*4, raw float32 bits) metaLen(4) metadata(JSON)
```

A deliberate **hybrid**: the vector is raw little-endian float32 bits (it's the
bulk of the bytes; JSON would bloat it ~5× and cost parse time), while metadata
is JSON (small, irregular, and JSON handles `string`/`number`/`bool` for free).
After decoding, `normalizeMetadata` drops any value that isn't one of our three
allowed types, so a hand-edited file can't smuggle in an unsupported type.

## 5. Snapshots: so the WAL doesn't grow forever (`snapshot.go`)

A WAL that only ever appends grows without bound, and replaying millions of
records at startup is slow. A **snapshot** writes the collection's *entire
current state* to one file; afterward the WAL can be truncated, because the
snapshot already contains everything up to that point.

### Atomic snapshot writes

If we overwrote `snapshot.dat` in place and crashed halfway, we'd lose **both**
the old and the new state — catastrophic. So `WriteSnapshot`:

1. writes to `snapshot.dat.tmp`,
2. fsyncs it,
3. `rename`s it over `snapshot.dat` (atomic on POSIX),
4. fsyncs the directory so the rename itself is durable.

A reader therefore always sees either the complete old snapshot or the complete
new one — never a half-written file.

## 6. The compaction ordering rule (`Store.Snapshot`)

Compaction is "snapshot, then truncate the WAL." The **order is a crash-safety
decision**:

- **Snapshot first, then truncate.** If we crash between them, recovery loads the
  new snapshot *and* replays a WAL that still holds those same mutations — they
  apply **twice**. That's safe because replay is **idempotent**: a duplicate
  insert and a delete-of-missing are both ignored on replay. No data lost.
- **Truncate first (WRONG).** Crash before the snapshot is written → the
  mutations are gone from the WAL and never made it to a snapshot → **data
  loss.**

So the code writes the snapshot durably first and only then truncates. Reason
through the crash window before choosing an order; the safe one usually relies on
idempotent replay.

## 7. Recovery = snapshot + WAL replay (`Store.recover`)

On `Open`, for each collection directory:

1. `ReadSnapshot` → config + all items as of the last snapshot.
2. Rebuild the collection and insert those items.
3. `ReplayWAL` → apply every mutation logged since the snapshot, in order.

`TestRecoverFromWAL` proves the pure-WAL path (crash with no snapshot), and
`TestRecoverFromSnapshotPlusWAL` proves the combined path (snapshot, then more
mutations, then recover).

## 8. The big simplification: we rebuild the graph, we don't persist it

We snapshot the **vectors and metadata**, not the HNSW **graph edges**. On
recovery we re-insert the vectors, which rebuilds the index from scratch.

- **Pro:** simple, and the recovered graph is guaranteed consistent.
- **Con:** startup cost is a full index build (O(N log N)), which is slow for
  large collections.

Production engines persist the graph itself — typically the adjacency lists in a
**memory-mapped file** (`mmap`), so the graph is "loaded" by mapping the file and
paging in on demand, with no rebuild. That's the natural next step and the reason
the architecture diagram in the original spec shows `vectors.dat` and `hnsw.dat` as
mmap'd files. We chose rebuild-on-load to keep this guide about *durability*
rather than *serialization format*, but the WAL/snapshot machinery here is
exactly what you'd keep.

## Takeaways

- Write-ahead + fsync before ack = no acknowledged write is ever lost.
- Length framing catches torn tails; CRC catches bit-rot; replay truncates to
  the last good record.
- Snapshot atomically (temp + rename + dir fsync) so you never destroy old and
  new at once.
- Compaction order (snapshot before truncate) leans on idempotent replay to stay
  crash-safe.
- Rebuild-on-load is the simple choice; mmap'd graph persistence is the
  production sequel.

Next: `07-grpc-api.md` — exposing all of this over the network.
