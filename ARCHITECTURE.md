# ClothoDB Architecture & Edge Case Handling

## Current Architecture Overview

### 1. Data Layer - Vector Clocks
**What we have:**
- Immutable vector clocks for causality tracking
- Happens-before/concurrent detection
- Merge operations for clock synchronization

**DynamoDB's approach:**
- Uses vector clocks (similar to ours)
- Prunes clocks when they grow too large (10 node limit)
- Falls back to timestamp-based reconciliation when clocks are pruned

### 2. Routing Layer - Consistent Hashing
**What we have:**
- 150 virtual nodes per physical node
- MD5-based hashing
- Replication factor N=3
- O(log n) lookup with binary search

**DynamoDB's approach:**
- Partition-based (not consistent hashing)
- Dynamo (predecessor) used consistent hashing
- Uses "preference list" - more than N nodes to handle failures

### 3. Storage Layer
**What we have:**
- In-memory dictionary
- No persistence
- No compression

**DynamoDB's approach:**
- LSM-tree based storage (RocksDB-like)
- SSTables with bloom filters
- Write-ahead log (WAL) for durability
- Automatic compaction
- Cold data to S3

---

## Critical Edge Cases & DynamoDB Solutions

### 1. Node Failures (Temporary vs Permanent)

**Edge Case:** Node goes down temporarily (network blip vs hardware failure)

**Our Current State:** ❌ Not handled

**DynamoDB's Solution:**
- **Hinted Handoff:** If a node is down, store the write on a temporary node with a hint
- When the target node comes back, the temporary node forwards the data
- Uses "sloppy quorum" - W writes to any N nodes, not necessarily the designated N

**Implementation needed:**
```python
class HintedHandoffManager:
    def store_hint(self, target_node, key, value, vector_clock):
        # Store on local node with metadata
        hint = {
            "target_node": target_node,
            "key": key,
            "value": value,
            "vector_clock": vector_clock,
            "timestamp": time.now(),
            "retry_count": 0
        }
        self.hints[target_node].append(hint)
    
    def forward_hints(self, target_node):
        # When target is back, forward all hints
        for hint in self.hints[target_node]:
            self.send_write(hint)
```

### 2. Read Repair

**Edge Case:** Replica nodes have different versions of the same key

**Our Current State:** ❌ Not handled - we return first value found

**DynamoDB's Solution:**
- During read, fetch from R replicas (R < N)
- Compare vector clocks
- If divergent, reconcile and write back the merged version
- Called "read repair"

**Implementation needed:**
```python
class ReadRepairHandler:
    async def read_with_repair(self, key, r=2):
        # Read from R nodes
        responses = await asyncio.gather(
            [self.read_from_node(node, key) for node in self.get_preference_list(key)[:r]]
        )
        
        # Find all versions
        versions = [r for r in responses if r.found]
        
        if len(versions) == 0:
            return None
        
        # Check for conflicts
        if self.has_conflicts(versions):
            # Reconcile
            merged = self.reconcile_versions(versions)
            # Async write back to all nodes
            asyncio.create_task(self.write_repaired(key, merged))
            return merged
        
        return versions[0]
```

### 3. Handling Slow Responses

**Edge Case:** Some replicas are slow (stragglers)

**Our Current State:** ❌ Wait for all responses

**DynamoDB's Solution:**
- Use "speculative execution" - send to N nodes, return first R responses
- Cancel remaining requests
- Configurable timeout (e.g., 99.9th percentile latency)

**Implementation needed:**
```python
class SpeculativeExecution:
    async def read_with_speculation(self, key, r=2, timeout_ms=10):
        nodes = self.get_preference_list(key)
        
        # Send to all N nodes
        tasks = [self.read_with_timeout(node, key, timeout_ms) for node in nodes]
        
        # Return first R successful responses
        responses = []
        for completed in asyncio.as_completed(tasks):
            try:
                result = await completed
                responses.append(result)
                if len(responses) >= r:
                    # Cancel remaining
                    for task in tasks:
                        task.cancel()
                    break
            except TimeoutError:
                continue
        
        return responses
```

### 4. Vector Clock Growth

**Edge Case:** Vector clocks grow unbounded as cluster size increases

**Our Current State:** ❌ No limit on clock size

**DynamoDB's Solution:**
- Prune oldest entries when clock exceeds threshold (e.g., 10 nodes)
- Remove entries with lowest timestamps
- Fall back to "last-write-wins" for pruned entries

**Implementation needed:**
```python
class VectorClockPruner:
    MAX_CLOCK_SIZE = 10
    
    def prune_if_needed(self, vector_clock):
        if len(vector_clock) <= self.MAX_CLOCK_SIZE:
            return vector_clock
        
        # Sort by timestamp (oldest first) and remove oldest
        sorted_entries = sorted(
            vector_clock.clock.items(),
            key=lambda x: x[1]  # timestamp
        )
        
        # Keep only MAX_CLOCK_SIZE newest
        pruned = dict(sorted_entries[-self.MAX_CLOCK_SIZE:])
        return VectorClock(pruned)
```

### 5. Network Partitions

**Edge Case:** Split-brain scenarios where network partitions the cluster

**Our Current State:** ⚠️ Partial - vector clocks detect conflicts but don't resolve

**DynamoDB's Solution:**
- Sacrifice availability for consistency during partitions (configurable)
- Or use "sloppy quorum" - accept writes from any available nodes
- Conflict resolution on reconciliation

**Implementation needed:**
```python
class PartitionHandler:
    def handle_partitioned_write(self, key, value, context):
        available_nodes = self.get_available_nodes()
        preference_list = self.get_preference_list(key)
        
        # Sloppy quorum: write to any W available nodes
        writable_nodes = [n for n in available_nodes if n in preference_list]
        
        if len(writable_nodes) < self.W:
            # Not enough nodes - reject write or write to non-preferred nodes
            if self.allow_sloppy_quorum:
                # Write to any available nodes
                writable_nodes = available_nodes[:self.W]
            else:
                raise QuorumNotMetError()
        
        return self.write_to_nodes(writable_nodes, key, value, context)
```

### 6. Hot Keys

**Edge Case:** One key gets massive traffic (celebrity problem)

**Our Current State:** ❌ No handling

**DynamoDB's Solution:**
- Adaptive capacity - automatically split hot partitions
- Request throttling with exponential backoff
- Load shedding when overloaded
- Separate "burst capacity" pools

**Implementation needed:**
```python
class HotKeyHandler:
    def __init__(self):
        self.key_access_counts = defaultdict(lambda: deque(maxlen=1000))
        self.hot_key_threshold = 1000  # accesses per second
        self.key_locks = {}
    
    def is_hot_key(self, key):
        recent_accesses = len(self.key_access_counts[key])
        return recent_accesses > self.hot_key_threshold
    
    async def handle_hot_read(self, key):
        if self.is_hot_key(key):
            # Use more replicas for hot keys
            r = min(self.N, self.R + 2)
            # Add jitter to spread load
            await asyncio.sleep(random.uniform(0, 0.001))
        else:
            r = self.R
        
        return await self.read_with_repair(key, r=r)
```

### 7. Request Timeouts & Retries

**Edge Case:** Requests timing out, need to retry without causing thundering herd

**Our Current State:** ❌ No retry logic

**DynamoDB's Solution:**
- Exponential backoff with jitter
- Circuit breaker pattern for failing nodes
- Separate fast/slow path
- Request hedging (send to multiple, use fastest)

**Implementation needed:**
```python
class ResilientRequestHandler:
    def __init__(self):
        self.circuit_breakers = {}
        self.retry_policy = ExponentialBackoff(
            base_delay=0.01,
            max_delay=1.0,
            max_retries=3
        )
    
    async def execute_with_retry(self, operation, key):
        for attempt in range(self.retry_policy.max_retries):
            try:
                node = self.select_node(key)
                
                if self.circuit_breakers[node].is_open():
                    continue  # Skip failed node
                
                return await asyncio.wait_for(
                    operation(node),
                    timeout=self.calculate_timeout(attempt)
                )
                
            except asyncio.TimeoutError:
                self.circuit_breakers[node].record_failure()
                delay = self.retry_policy.get_delay(attempt)
                await asyncio.sleep(delay)
        
        raise MaxRetriesExceeded()
```

### 8. Merkle Trees for Anti-Entropy

**Edge Case:** Detecting divergent replicas efficiently

**Our Current State:** ❌ No background reconciliation

**DynamoDB's Solution:**
- Use Merkle trees (hash trees) to compare replicas
- Only transfer differing ranges
- Background anti-entropy process

**Implementation needed:**
```python
class MerkleTree:
    def __init__(self):
        self.tree = {}
    
    def build_tree(self, key_range):
        # Build tree from key hashes
        # Leaf nodes = individual key hashes
        # Internal nodes = hash of children
        pass
    
    def compare_trees(self, other_tree):
        # Return list of differing key ranges
        differences = []
        self._compare_nodes(self.tree, other_tree.tree, differences)
        return differences
```

### 9. Gossip Protocol

**Edge Case:** Nodes need to discover membership changes and failures

**Our Current State:** ❌ Static membership

**DynamoDB's Solution:**
- Gossip-based membership (SWIM protocol or similar)
- Each node gossips to 3 random nodes every second
- Failure detection via indirect pings
- Automatic node removal after timeout

**Implementation needed:**
```python
class GossipProtocol:
    def __init__(self):
        self.membership = {}
        self.failure_detector = FailureDetector()
    
    async def gossip_loop(self):
        while True:
            # Pick 3 random nodes
            targets = random.sample(self.get_live_nodes(), 3)
            
            for target in targets:
                try:
                    # Send gossip
                    response = await self.send_gossip(target, self.membership)
                    # Merge received state
                    self.merge_membership(response)
                except TimeoutError:
                    self.failure_detector.suspect(target)
            
            await asyncio.sleep(1)
    
    def merge_membership(self, remote_state):
        # Keep most recent heartbeat for each node
        for node_id, info in remote_state.items():
            if node_id not in self.membership:
                self.membership[node_id] = info
            elif info['heartbeat'] > self.membership[node_id]['heartbeat']:
                self.membership[node_id] = info
```

### 10. Conflict Resolution Strategies

**Edge Case:** Multiple conflicting versions of same data

**Our Current State:** ⚠️ Detects but doesn't auto-resolve

**DynamoDB's Solution:**
- Configurable: Last-Write-Wins (LWW) vs Vector Clock reconciliation
- For LWW: use timestamp with clock skew correction
- For VC: return multiple versions to client (siblings)
- Application must resolve siblings

**Implementation needed:**
```python
class ConflictResolver:
    def resolve(self, versions, strategy="vector_clock"):
        if strategy == "lww":
            return max(versions, key=lambda v: v.timestamp)
        
        elif strategy == "vector_clock":
            # Check if one dominates
            for v1 in versions:
                if all(v1.dominates(v2) or v1 == v2 for v2 in versions):
                    return v1
            
            # Conflict - return all versions (siblings)
            return versions  # Client must resolve
        
        elif strategy == "merge":
            # Automatic merge for specific data types (CRDTs)
            return self.crdt_merge(versions)
```

---

## Implementation Status

### ✅ Phase 1: Critical (COMPLETE)
| Feature | Status | Location |
|---------|--------|----------|
| **Hinted Handoff** | ✅ Implemented | `clotho/resilience.py` - `HintedHandoffManager` |
| **Read Repair** | ✅ Implemented | `clotho/resilience.py` - `ReadRepairHandler` |
| **Request Timeouts & Retries** | ✅ Implemented | `clotho/resilience.py` - `RetryPolicy`, `ResilientOperation` |
| **Vector Clock Pruning** | ✅ Implemented | `clotho/resilience.py` - `VectorClockPruner` |

### ✅ Phase 2: Important (MOSTLY COMPLETE)
| Feature | Status | Location |
|---------|--------|----------|
| **Sloppy Quorum** | ✅ Implemented | `clotho/resilience.py` - `QuorumManager` with `QuorumType.SLOPPY` |
| **Gossip Protocol** | ⚠️ Partial | `clotho/gossip.py` - Phi-accrual failure detector |
| **Speculative Execution** | ⚠️ Partial | Built into `ReadRepairHandler` (first R responses) |
| **Circuit Breakers** | ✅ Implemented | `clotho/resilience.py` - `CircuitBreaker`, `CircuitBreakerManager` |

### ⏳ Phase 3: Advanced (PENDING)
| Feature | Status | Notes |
|---------|--------|-------|
| **Merkle Trees** | ⏳ Pending | Efficient anti-entropy for large datasets |
| **Hot Key Handling** | ⏳ Pending | Adaptive capacity for celebrity keys |
| **CRDT Support** | ⏳ Pending | Automatic conflict resolution for specific types |
| **Persistence** | ⏳ Pending | Write-ahead logging and SSTables |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Request                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Request Router / Load Balancer                 │
│         (Consistent Hashing → Preference List)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Coordinator Node                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Read      │  │   Write     │  │   Delete    │         │
│  │   Handler   │  │   Handler   │  │   Handler   │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                │
│         ▼                ▼                ▼                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Request Dispatcher                     │   │
│  │  (Sloppy Quorum / Speculative Execution)           │   │
│  └─────────────────────┬───────────────────────────────┘   │
└────────────────────────┼────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                     Replica Nodes (N=3)                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Node 1   │◄──►│ Node 2   │◄──►│ Node 3   │              │
│  │(Primary) │    │(Replica) │    │(Replica) │              │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘              │
│       │               │               │                     │
│       ▼               ▼               ▼                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Storage Engine (with Vector Clocks & Read Repair)  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Background Processes                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Gossip  │  │  Hinted  │  │  Merkle  │  │Compaction│   │
│  │ Protocol │  │ Handoff  │  │  Trees   │  │          │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Testing Edge Cases

### Test Matrix

| Edge Case | Test Strategy | Expected Behavior |
|-----------|--------------|-------------------|
| Node failure during write | Kill node mid-write | Hinted handoff stores write |
| Slow replica | Add 100ms delay to one node | Speculative execution ignores slow node |
| Concurrent writes | Two clients write same key | Vector clocks detect conflict |
| Partition | Block traffic between nodes | Sloppy quorum allows writes |
| Clock growth | 1000 different writers | Clock pruned to 10 entries |
| Hot key | 10k req/s to single key | Load spread across more replicas |
| All nodes fail | Stop all replicas | Write rejected with clear error |
| Node rejoin | Restart failed node | Hints forwarded, read repair fixes |
| Large values | 10MB values | Chunking or rejection |
| Expired data | TTL expiration | Background cleanup |
