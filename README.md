# ClothoDB

## Why Clotho?

In Greek mythology, Clotho is the youngest of the Moirai (the Fates). Her job was to spin the thread of human destiny, weaving the timeline of events. 

Since the core of this datastore is managing causal histories—weaving concurrent events together and determining their absolute ordering with vector clocks—Clotho is a fitting namesake. The database has no single "master" thread of time (no leader); instead, nodes weave independent threads that must be mathematically reconciled.

## Causality Tracking with Vector Clocks

The fundamental problem in distributed systems: **How do we know if event A happened before event B?**

In a leaderless database like DynamoDB:
- Each node has its own local time (logical clock)
- There is no global clock
- Events can happen concurrently (independently on different nodes)

### The Solution: Vector Clocks

Vector clocks solve this by tracking a vector of counters, one per node in the system.

**Key Operations:**
1. **Increment**: When a node performs a local event, it increments its own counter
2. **Merge**: When nodes communicate, they merge clocks by taking the component-wise maximum
3. **Compare**: Compare two clocks to determine causality:
   - `A → B` (A happens before B): All components of A ≤ B, at least one strictly <
   - `A || B` (concurrent): Neither A ≤ B nor B ≤ A (incomparable)
   - `A == B` (equal): All components equal

### Example Scenario

```
Time →

Node A: [1,0,0] → [2,0,0] → [3,2,1] (receives from B and C)
                  ↘
Node B: [0,1,0] → [1,2,0] → [1,3,0]
                  ↗
Node C: [0,0,1] ───────────→ [0,0,2]
```

At clock `[3,2,1]` on Node A:
- A has performed 3 events
- A has received 2 events from B
- A has received 1 event from C

## Project Structure

```
clotho/
├── __init__.py           # Package exports
├── vector_clock.py       # Core vector clock implementation
├── node.py               # Distributed node simulation
└── visualization.py      # Debugging and visualization tools

tests/
├── test_vector_clock.py      # Unit tests for vector clocks
├── test_integration.py       # Multi-node integration tests
├── test_property_based.py    # Property-based/fuzzing tests
└── conftest.py               # Pytest configuration
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_vector_clock.py -v

# Run with coverage
pytest tests/ --cov=clotho --cov-report=html

# Run property-based tests with more examples
pytest tests/test_property_based.py --hypothesis-profile=thorough
```

## Usage Example

```python
from clotho.node import Node, Cluster
from clotho.vector_clock import VectorClock

# Create two nodes
node_a = Node("Alice")
node_b = Node("Bob")

# Each node writes independently (concurrent events)
event_a = node_a.write("x", 1)  # Alice writes x=1
event_b = node_b.write("y", 2)  # Bob writes y=2

# These events are concurrent (no causal relationship)
assert event_a.vector_clock.is_concurrent_with(event_b.vector_clock)

# Alice sends message to Bob
node_a.send_message(node_b, "x")

# Now Bob's clock reflects Alice's events
assert node_b.vector_clock.happens_after(event_a.vector_clock)

# Detect conflicts in a cluster
cluster = Cluster()
cluster.add_node("A")
cluster.add_node("B")
# ... perform operations ...
conflicts = cluster.find_concurrent_conflicts()
```

## Key Properties Verified by Tests

### Mathematical Properties
1. **Merge is commutative**: `A.merge(B) == B.merge(A)`
2. **Merge is associative**: `(A.merge(B)).merge(C) == A.merge(B.merge(C))`
3. **Merge is idempotent**: `A.merge(A) == A`
4. **Happens-before is transitive**: If A → B and B → C, then A → C
5. **Concurrency is symmetric**: If A || B, then B || A

### Distributed System Properties
1. **Causal consistency**: If A causally precedes B, all nodes see A before B
2. **Conflict detection**: Concurrent writes to the same key are detected
3. **Clock monotonicity**: A node's own counter never decreases
4. **Convergence**: Nodes that communicate eventually agree on causality

## Consistent Hashing & Routing

ClothoDB implements consistent hashing for distributed data routing across multiple servers.

### Features

1. **Virtual Nodes**: Each physical node is mapped to 150 virtual nodes for even distribution
2. **Replication Factor N=3**: Each key is replicated to 3 nodes for high availability
3. **Minimal Reorganization**: When nodes are added/removed, only 1/N keys are remapped
4. **O(log n) Lookup**: Binary search for efficient node discovery

### API: Get Top 3 Nodes for a Key

```python
from clotho.server import get_top3_nodes_api, get_top3_nodes_for_key
from clotho.consistent_hash import create_5_server_cluster

# Method 1: Simple API (creates 5-server cluster automatically)
result = get_top3_nodes_api("my-key")
print(result)
# {
#     "key": "my-key",
#     "top_3_nodes": [
#         {"rank": 1, "node_id": "server-3", "host": "127.0.0.1", "port": 8002, "is_primary": True},
#         {"rank": 2, "node_id": "server-1", "host": "127.0.0.1", "port": 8000, "is_primary": False},
#         {"rank": 3, "node_id": "server-5", "host": "127.0.0.1", "port": 8004, "is_primary": False}
#     ]
# }

# Method 2: Using router directly
router = create_5_server_cluster(base_port=8000)
nodes = get_top3_nodes_for_key("my-key", router)
for node in nodes:
    print(f"{node.node_id} @ {node.address}")
```

### HTTP Server API

Each server runs on its own port and provides REST endpoints:

```python
from clotho.server import Cluster

# Create and start 5 servers on ports 8000-8004
cluster = Cluster(base_port=8000, num_servers=5, replication_factor=3)
cluster.print_routing_table()

# Run a specific server (blocking)
cluster.run_server("server-1")
```

**Available Endpoints:**

| Endpoint | Description |
|----------|-------------|
| `GET /get/{key}` | Get value for a key |
| `POST /put` | Store a key-value pair |
| `DELETE /delete/{key}` | Delete a key |
| `GET /route/{key}` | Get routing info for a key |
| `GET /top3/{key}` | **Get top 3 nodes for a key** |
| `GET /status` | Node status and vector clock |
| `GET /cluster/status` | Full cluster status |

**Example Usage:**

```bash
# Get top 3 nodes for key "user:123"
curl http://127.0.0.1:8000/top3/user:123

# Store a value
curl -X POST http://127.0.0.1:8000/put \
  -H "Content-Type: application/json" \
  -d '{"key": "user:123", "value": {"name": "Alice"}}'

# Get routing info
curl http://127.0.0.1:8000/route/user:123
```

### How Consistent Hashing Works

```
Hash Ring (0 to 2^128-1):

    server-1 (vn0) ---- server-2 (vn0) ---- server-3 (vn0)
         |                    |                    |
    server-1 (vn1)      server-2 (vn1)      server-3 (vn1)
         |                    |                    |
    server-1 (vn2)      server-2 (vn2)      server-3 (vn2)
         ...                  ...                  ...

Key "user:123" hashes to position X.
The first 3 unique nodes clockwise from X store replicas.
```

### Demo

```bash
# Run the routing demo
python demo_routing.py
```

This demonstrates:
- Key routing across 5 servers
- Top 3 nodes API
- Key distribution statistics
- Minimal reorganization on node removal

## Project Structure

```
clotho/
├── __init__.py           # Package exports
├── vector_clock.py       # Core vector clock implementation
├── node.py               # Distributed node simulation
├── consistent_hash.py    # Consistent hashing ring
├── server.py             # HTTP server with routing API
└── visualization.py      # Debugging and visualization tools

tests/
├── test_vector_clock.py      # Vector clock unit tests
├── test_integration.py       # Multi-node integration tests
├── test_property_based.py    # Property-based/fuzzing tests
├── test_consistent_hash.py   # Consistent hashing tests
└── conftest.py               # Pytest configuration
```

## Edge Case Handling (DynamoDB-style)

ClothoDB implements production-grade edge case handling inspired by DynamoDB:

| Edge Case | Solution | Status |
|-----------|----------|--------|
| Temporary node failure | **Hinted Handoff** | ✅ Store writes on alternate nodes, forward when target recovers |
| Cascading failures | **Circuit Breaker** | ✅ Open circuit after threshold failures, fast-fail until recovery |
| Transient errors | **Retry + Backoff** | ✅ Exponential backoff with jitter prevents thundering herd |
| Network partitions | **Sloppy Quorum** | ✅ Write to any available W nodes, not just designated replicas |
| Clock growth | **Pruning** | ✅ Auto-prune vector clocks to 10 entries (like DynamoDB) |
| Inconsistent replicas | **Read Repair** | ✅ Compare R replicas during reads, fix divergent versions |

### Demo

```bash
# Run the edge case handling demo
python demo_edge_cases.py
```

This demonstrates all 6 edge case handling mechanisms with real examples.

### Usage Example: Resilient Storage

```python
import asyncio
from clotho import ResilientStorage, create_5_server_cluster

async def main():
    router = create_5_server_cluster()
    storage = ResilientStorage(router, node_id="client-1")
    await storage.start()  # Start background hint forwarding
    
    # Write with automatic resilience
    result = await storage.write("key", {"data": "value"})
    print(f"Success: {result['writes_succeeded']}/{result['writes_succeeded'] + result['writes_failed']}")
    print(f"Hints stored for failed nodes: {result['hints_stored']}")
    
    # Read with automatic repair
    result = await storage.read("key")
    print(f"Value: {result['value']}")

asyncio.run(main())
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for:
- Detailed design decisions
- Edge case analysis and DynamoDB comparison
- Implementation roadmap
- Testing matrix

## Test Summary

```
176 tests passing ✅
├── test_vector_clock.py      # 54 tests (causality, merge, compare)
├── test_consistent_hash.py   # 32 tests (routing, replication)
├── test_resilience.py        # 46 tests (edge case handling)
├── test_integration.py       # 25 tests (multi-node scenarios)
└── test_property_based.py    # 19 tests (fuzzing)
```

## References

- **Dynamo: Amazon's Highly Available Key-value Store** (DeCandia et al., 2007)
- **Time, Clocks, and the Ordering of Events in a Distributed System** (Lamport, 1978)
- **Conflict-free Replicated Data Types** (Shapiro et al., 2011)
