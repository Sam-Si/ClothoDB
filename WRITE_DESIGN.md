# ClothoDB Write Path Design: Implementation Options & Trade-offs

## Overview

Designing the write path for a distributed database involves balancing **durability**, **latency**, **consistency**, and **throughput**. This document explores all major implementation approaches with their pros and cons.

---

## 1. Simple In-Memory Write (Current Implementation)

### How It Works
```python
async def write_simple(key, value):
    # 1. Route to primary node
    primary = router.get_primary_node(key)
    
    # 2. Store in memory dict
    storage[key] = (value, vector_clock)
    
    # 3. Replicate asynchronously (fire-and-forget)
    for replica in get_replicas(key):
        asyncio.create_task(replicate_async(replica, key, value))
    
    # 4. Return immediately
    return {"success": True}
```

### Pros
- ✅ **Ultra-low latency** (< 1ms)
- ✅ **Simple implementation**
- ✅ **High throughput** (no disk I/O)
- ✅ No write amplification

### Cons
- ❌ **Data loss on crash** (no durability)
- ❌ **No persistence** across restarts
- ❌ Replication may fail silently
- ❌ Inconsistent on node failure

### Best For
- Caches, session stores, temporary data
- Prototyping and testing

---

## 2. Write-Ahead Log (WAL) with Group Commit

### How It Works
```python
async def write_wal(key, value):
    # 1. Serialize write operation
    log_entry = serialize_write(key, value, vector_clock)
    
    # 2. Append to WAL (buffered, fsync every N ms or entries)
    await wal.append(log_entry)
    
    # 3. Apply to in-memory structure
    memtable[key] = (value, vector_clock)
    
    # 4. Return after WAL fsync
    return {"success": True}
```

### Variants

#### 2a. Synchronous WAL (fsync per write)
```python
await wal.append(entry)
await wal.fsync()  # Wait for disk
```

#### 2b. Group Commit (fsync every N ms)
```python
# Background task periodically fsyncs
async def group_commit_loop():
    while True:
        await asyncio.sleep(0.01)  # 10ms
        await wal.fsync()
```

#### 2c. Async WAL (no fsync, rely on OS)
```python
await wal.append(entry)
# No fsync - fast but may lose last N seconds of data
```

### Pros
- ✅ **Durability** (survives crashes)
- ✅ **Fast recovery** (replay WAL)
- ✅ **Sequential writes** (fast on SSD/HDD)
- ✅ Can batch/group commits

### Cons
- ❌ **Write amplification** (write to WAL + memtable + SSTable)
- ❌ **Disk I/O latency** (fsync ~1-10ms)
- ❌ WAL files need management (rotation, cleanup)
- ❌ Recovery time grows with WAL size

### Best For
- Production systems requiring durability
- When data loss is unacceptable

---

## 3. Quorum-Based Write (DynamoDB-Style)

### How It Works
```python
async def write_quorum(key, value, w=2, n=3):
    # 1. Get N replicas
    replicas = router.get_replica_nodes(key, n=n)
    
    # 2. Send writes concurrently
    write_tasks = [
        send_write(replica, key, value, vector_clock)
        for replica in replicas
    ]
    
    # 3. Wait for W acknowledgments
    done, pending = await asyncio.wait(
        write_tasks,
        return_when=asyncio.FIRST_COMPLETED,
        timeout=timeout_ms
    )
    
    # 4. Check if quorum reached
    successes = sum(1 for t in done if not t.exception())
    
    if successes >= w:
        # 5. Store hints for failed nodes
        for task in pending:
            if task.exception():
                hinted_handoff.store_hint(failed_node, key, value, clock)
        
        return {"success": True, "writes": successes}
    else:
        return {"success": False, "error": "Quorum not reached"}
```

### Variants

#### 3a. Synchronous Quorum (Wait for W)
- Wait for W acknowledgments before returning
- Strong consistency guarantee
- Higher latency

#### 3b. Asynchronous Quorum (Write to coordinator)
- Write to coordinator, return immediately
- Coordinator replicates asynchronously
- Lower latency, eventual consistency

#### 3c. Sloppy Quorum (During Partitions)
- Accept writes from ANY available nodes
- Not just designated replicas
- Hinted handoff for later reconciliation

### Pros
- ✅ **High availability** (tolerates N-W failures)
- ✅ **Tunable consistency** (W=1 fast, W=N strong)
- ✅ **Natural load balancing**
- ✅ Hinted handoff handles temporary failures

### Cons
- ❌ **Higher latency** (network round-trips)
- ❌ **Conflict resolution needed** (concurrent writes)
- ❌ **Complex failure handling**
- ❌ W < N means read repair needed

### Best For
- Distributed systems requiring high availability
- When network partitions are expected

---

## 4. Two-Phase Commit (2PC)

### How It Works
```python
async def write_2pc(key, value):
    coordinator = router.get_primary_node(key)
    participants = router.get_replica_nodes(key)
    
    # Phase 1: Prepare
    prepare_tasks = [
        send_prepare(participant, key, value, vector_clock)
        for participant in participants
    ]
    prepare_results = await asyncio.gather(*prepare_tasks)
    
    if all(r.success for r in prepare_results):
        # Phase 2a: Commit
        commit_tasks = [
            send_commit(participant, key, vector_clock)
            for participant in participants
        ]
        await asyncio.gather(*commit_tasks)
        return {"success": True}
    else:
        # Phase 2b: Abort
        abort_tasks = [
            send_abort(participant, key)
            for participant in participants
        ]
        await asyncio.gather(*abort_tasks)
        return {"success": False, "error": "Prepare failed"}
```

### Pros
- ✅ **Strong consistency** (ACID)
- ✅ **No conflicts** (all nodes agree)
- ✅ Well-understood protocol

### Cons
- ❌ **Blocking** (locks held during commit)
- ❌ **Coordinator failure** requires recovery
- ❌ **High latency** (2 round-trips minimum)
- ❌ **Not partition-tolerant** (CAP theorem)

### Best For
- Financial transactions
- When strong consistency is mandatory
- Small, stable clusters

---

## 5. Optimistic Write with Vector Clocks

### How It Works
```python
async def write_optimistic(key, value, context=None):
    # 1. Read current version (if updating)
    current = await read(key)
    
    # 2. Check causality
    if context and current:
        if context.happens_before(current.vector_clock):
            raise ConflictError("Write based on stale data")
    
    # 3. Increment vector clock
    new_clock = (context or VectorClock.new(node_id)).increment(node_id)
    
    # 4. Write locally
    storage[key] = (value, new_clock)
    
    # 5. Replicate asynchronously
    for replica in get_replicas(key):
        asyncio.create_task(replicate_with_clock(replica, key, value, new_clock))
    
    return {"success": True, "vector_clock": new_clock.to_dict()}
```

### Conflict Resolution
```python
def resolve_conflict(versions):
    # Last-Write-Wins (by timestamp)
    return max(versions, key=lambda v: v.timestamp)
    
    # OR: Vector Clock merge
    merged_clock = versions[0].clock
    for v in versions[1:]:
        merged_clock = merged_clock.merge(v.clock)
    return merged_value, merged_clock
    
    # OR: Client-side resolution
    return {"conflict": True, "versions": versions}
```

### Pros
- ✅ **Low latency** (local write)
- ✅ **Always available** (accepts writes during partitions)
- ✅ **Causality tracking** (vector clocks)
- ✅ **Flexible conflict resolution**

### Cons
- ❌ **Conflicts possible** (requires resolution)
- ❌ **Eventual consistency** (not immediate)
- ❌ **Client complexity** (must handle conflicts)
- ❌ **Storage overhead** (multiple versions)

### Best For
- High-write workloads
- Mobile/offline-first applications
- When availability > strong consistency

---

## 6. Batch Write with Buffering

### How It Works
```python
class BatchWriter:
    def __init__(self, max_batch_size=100, max_latency_ms=10):
        self.buffer = []
        self.max_batch_size = max_batch_size
        self.max_latency_ms = max_latency_ms
        
    async def write(self, key, value):
        future = asyncio.Future()
        self.buffer.append((key, value, future))
        
        if len(self.buffer) >= self.max_batch_size:
            await self.flush()
        
        return await future
    
    async def flush(self):
        batch = self.buffer[:self.max_batch_size]
        self.buffer = self.buffer[self.max_batch_size:]
        
        # Write batch as single operation
        await write_batch_to_wal(batch)
        
        # Notify all waiters
        for _, _, future in batch:
            future.set_result({"success": True})
```

### Pros
- ✅ **High throughput** (amortize I/O cost)
- ✅ **Efficient disk usage** (sequential writes)
- ✅ **Compression** (compress batches)
- ✅ **Reduced write amplification**

### Cons
- ❌ **Higher latency** (wait for batch fill)
- ❌ **Data loss on crash** (unflushed buffer)
- ❌ **Complex error handling** (partial batch failure)
- ❌ **Ordering challenges**

### Best For
- High-throughput logging
- Analytics workloads
- When latency is less critical

---

## 7. Tiered Storage Write (Hot/Cold)

### How It Works
```python
async def write_tiered(key, value):
    # 1. Always write to hot tier (memory/WAL)
    hot_storage[key] = (value, vector_clock, timestamp)
    
    # 2. Check if should migrate to cold
    if should_migrate_to_cold(key, timestamp):
        await migrate_to_cold_tier(key)
    
    # 3. Async replicate
    asyncio.create_task(replicate_to_tier(key, value, "hot"))
    
    return {"success": True}

def should_migrate_to_cold(key, timestamp):
    # LRU, LFU, or time-based policy
    age = time.now() - timestamp
    return age > HOT_TIER_MAX_AGE
```

### Storage Tiers
1. **Hot**: Memory (fastest, most expensive)
2. **Warm**: SSD (fast, durable)
3. **Cold**: HDD/S3 (slow, cheap)

### Pros
- ✅ **Cost optimization** (hot data in memory)
- ✅ **Automatic lifecycle management**
- ✅ **Scalable** (unlimited cold storage)

### Cons
- ❌ **Complexity** (multiple storage backends)
- ❌ **Migration overhead**
- ❌ **Consistency across tiers**
- ❌ **Cold start latency**

### Best For
- Large datasets (TB+)
- Variable access patterns
- Cost-sensitive applications

---

## 8. Pipeline Write (Primary-Backup Chain)

### How It Works
```python
async def write_pipeline(key, value):
    replicas = router.get_replica_nodes(key)
    primary = replicas[0]
    backups = replicas[1:]
    
    # 1. Write to primary
    await primary.write(key, value, vector_clock)
    
    # 2. Primary forwards to next in chain
    if backups:
        asyncio.create_task(
            primary.forward_to(backups[0], key, value, vector_clock)
        )
    
    # 3. Return after primary confirms
    return {"success": True}
```

### Chain Replication
```
Client → Primary → Backup-1 → Backup-2
         (ack)     (async)    (async)
```

### Pros
- ✅ **Ordered writes** (sequential in chain)
- ✅ **Load distribution** (each node does less work)
- ✅ **Simple consistency model**

### Cons
- ❌ **Primary bottleneck**
- ❌ **Latency accumulation** (chain length)
- ❌ **Failure handling** (chain breaks)

### Best For
- Ordered event streams
- When write order matters
- HDFS-style workloads

---

## 9. Multi-Version Write (MVCC)

### How It Works
```python
async def write_mvcc(key, value, context=None):
    # 1. Get current version
    versions = storage.get(key, [])
    
    # 2. Determine new version number
    new_version = max(v.version for v in versions) + 1 if versions else 1
    
    # 3. Check for conflicts
    if context:
        base_version = context.version
        if not any(v.version == base_version for v in versions):
            raise ConflictError("Base version no longer exists")
    
    # 4. Append new version
    versions.append(VersionedValue(
        value=value,
        version=new_version,
        vector_clock=clock,
        timestamp=time.now()
    ))
    
    storage[key] = versions
    
    # 5. Schedule old version cleanup
    asyncio.create_task(cleanup_old_versions(key))
    
    return {"success": True, "version": new_version}
```

### Pros
- ✅ **Snapshot isolation** (read old versions)
- ✅ **Time-travel queries** (read historical data)
- ✅ **No read locks** (readers don't block writers)
- ✅ **Automatic conflict detection**

### Cons
- ❌ **Storage overhead** (multiple versions)
- ❌ **Cleanup complexity** (when to delete old versions)
- ❌ **Read amplification** (filter versions)

### Best For
- Time-series data
- Audit trails
- Concurrent read/write workloads

---

## 10. Append-Only Write (Log-Structured)

### How It Works
```python
async def write_append_only(key, value):
    # 1. Create immutable log entry
    entry = LogEntry(
        op="WRITE",
        key=key,
        value=value,
        vector_clock=clock,
        timestamp=time.now()
    )
    
    # 2. Append to global log
    offset = await global_log.append(entry.serialize())
    
    # 3. Update index (key -> [log offsets])
    index[key].append(offset)
    
    # 4. Periodic compaction merges entries
    
    return {"success": True, "offset": offset}
```

### Compaction
```python
async def compact(key):
    # Read all log entries for key
    entries = [global_log.read(offset) for offset in index[key]]
    
    # Merge into single value
    merged = merge_entries(entries)
    
    # Write new compacted entry
    new_offset = await global_log.append(merged)
    
    # Update index
    index[key] = [new_offset]
```

### Pros
- ✅ **Sequential writes** (maximum disk throughput)
- ✅ **Immutable data** (easy replication)
- ✅ **Crash recovery** (replay log)
- ✅ **Time-travel** (any point in time)

### Cons
- ❌ **Read amplification** (reconstruct from log)
- ❌ **Compaction overhead** (background I/O)
- ❌ **Space overhead** (until compaction)
- ❌ **Write amplification** (compaction rewrites)

### Best For
- High-write workloads
- Event sourcing
- Kafka-style log processing

---

## Comparison Matrix

| Approach | Latency | Throughput | Durability | Consistency | Complexity |
|----------|---------|------------|------------|-------------|------------|
| Simple In-Memory | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐ | ⭐ |
| WAL + Group Commit | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Quorum-Based | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| 2PC | ⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Optimistic (VCC) | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐ |
| Batch Write | ⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Tiered Storage | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| Pipeline | ⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| MVCC | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Append-Only | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

---

## Recommendation for ClothoDB

### Recommended: **Hybrid Approach (Quorum + WAL + Optimistic)**

```
┌─────────────────────────────────────────────────────────────┐
│                     Write Path Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Client                                                      │
│    │                                                         │
│    ▼                                                         │
│  ┌──────────────────┐                                        │
│  │ 1. Route to N    │  Consistent Hashing                    │
│  │    Replicas      │                                        │
│  └────────┬─────────┘                                        │
│           │                                                  │
│    ┌──────┴──────┬────────┬────────┐                        │
│    ▼             ▼        ▼        ▼                        │
│  ┌────┐      ┌────┐   ┌────┐   ┌────┐                      │
│  │ R1 │      │ R2 │   │ R3 │   │ R4 │  (N=3, W=2)          │
│  └─┬──┘      └─┬──┘   └─┬──┘   └────┘                      │
│    │           │        │                                   │
│    ▼           ▼        ▼                                   │
│  ┌──────────────────────────────────────┐                  │
│  │ 2. Write to WAL (fsync every 10ms)   │  Durability      │
│  └──────────────────┬───────────────────┘                  │
│                     │                                       │
│                     ▼                                       │
│  ┌──────────────────────────────────────┐                  │
│  │ 3. Apply to Memtable                 │  Fast Reads      │
│  └──────────────────┬───────────────────┘                  │
│                     │                                       │
│                     ▼                                       │
│  ┌──────────────────────────────────────┐                  │
│  │ 4. Return to Client after W acks     │  Quorum          │
│  └──────────────────────────────────────┘                  │
│                                                              │
│  Background:                                                 │
│  - Flush memtable to SSTable when full                      │
│  - Compact SSTables periodically                            │
│  - Replicate to remaining N-W nodes                         │
│  - Hinted handoff for failed nodes                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Plan

```python
class WriteCoordinator:
    def __init__(self, router, wal, storage):
        self.router = router
        self.wal = wal
        self.storage = storage
        self.n = 3  # Replication factor
        self.w = 2  # Write quorum
    
    async def write(self, key, value, context=None):
        # 1. Get replicas
        replicas = self.router.get_replica_nodes(key, n=self.n)
        
        # 2. Prepare write with vector clock
        clock = self._prepare_clock(context)
        write_op = WriteOperation(key, value, clock)
        
        # 3. Send to all N replicas concurrently
        tasks = [
            self._write_to_replica(replica, write_op)
            for replica in replicas
        ]
        
        # 4. Wait for W successes
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.ALL_COMPLETED,
            timeout=5000  # 5 second timeout
        )
        
        successes = sum(1 for t in done if not t.exception())
        
        # 5. Handle results
        if successes >= self.w:
            # Store hints for failed replicas
            for task in pending:
                if task.exception():
                    failed_node = task.get_coro().cr_frame.f_locals['replica']
                    self.hinted_handoff.store_hint(failed_node.node_id, key, value, clock)
            
            return {
                "success": True,
                "vector_clock": clock.to_dict(),
                "replicas_written": successes
            }
        else:
            return {
                "success": False,
                "error": f"Quorum not reached: {successes}/{self.w}"
            }
    
    async def _write_to_replica(self, replica, write_op):
        # Each replica:
        # 1. Append to WAL
        await self.wal.append(write_op.serialize())
        # 2. Update memtable
        self.storage.memtable[write_op.key] = (write_op.value, write_op.clock)
        # 3. Return ack
        return {"ack": True}
```

### Why This Approach?

1. **Quorum (W=2, N=3)**: Survives 1 node failure without blocking
2. **WAL**: Durability across restarts
3. **Group Commit**: Amortize fsync cost (10ms batches)
4. **Vector Clocks**: Automatic conflict detection
5. **Hinted Handoff**: Handle temporary failures gracefully
6. **Memtable + SSTable**: Fast reads, efficient disk usage

### Trade-offs Accepted

- **Latency**: ~10ms (group commit window) + network RTT
- **Consistency**: Eventual (W < N), conflicts possible
- **Complexity**: Medium (WAL + quorum + compaction)

This gives us the best balance for a DynamoDB-like system.
